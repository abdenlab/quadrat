"""Tests for the grouping behind the sparse window cache.

The contract is that the grouped form describes the same signal a
coordinate read describes. So the central assertion here is a
cross-check rather than a golden array: densify what the grouping
returns and compare it against `quadrat.stores.TrackStore`, window by
window. One path reads a contig at a time through polars, the other
groups through the engine, and agreement is a statement about both.

Upstream pinned a second, hand-rolled grouping engine against this one
byte for byte. That engine is not ported, so the store carries the
oracle instead -- which is the stronger of the two, since it compares
against a reader rather than against another writer of the same
layout.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import gc
import multiprocessing
import os
import pickle

import numpy
import pytest

from quadrat import cache, stores
from tests import TWO_CONTIGS

# Channel 0 puts two rows inside one window and one on a second
# contig; channel 1 interleaves with channel 0 inside that window and
# records nothing on the second contig.
CHANNEL_0 = [("chr1", 10, 1.0), ("chr1", 12, 3.0), ("chr1", 400, 7.0), ("chr2", 5, 2.0)]


CHANNEL_1 = [("chr1", 11, 5.0)]


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    return corpus_factory(
        "cache", TWO_CONTIGS, tracks=[CHANNEL_0, CHANNEL_1], in_window=60, out_window=30
    )


@pytest.fixture(scope="module")
def three_channel(corpus_factory):
    return corpus_factory(
        "cache-three",
        TWO_CONTIGS,
        tracks=[CHANNEL_0, CHANNEL_1, [("chr1", 13, 9.0)]],
    )


@pytest.fixture
def tracks(corpus):
    pytest.importorskip("polars")
    pytest.importorskip("polars_bio")
    return corpus.tracks


def frame(rows, columns=("sig_start", "sig_end")):
    """A window frame from ``(chrom, start, end)`` rows."""

    polars = pytest.importorskip("polars")
    lo, hi = columns
    return polars.DataFrame(
        {
            "chrom": [str(row[0]) for row in rows],
            lo: [int(row[1]) for row in rows],
            hi: [int(row[2]) for row in rows],
        },
        schema={"chrom": polars.Utf8, lo: polars.Int64, hi: polars.Int64},
    )


def densify(grouped, windows, columns=("sig_start", "sig_end")):
    """The dense array per window that a grouped result describes."""

    offsets, pos, val, chan, n_channels = grouped
    lo, hi = columns
    out = []
    for index, (start, end) in enumerate(zip(windows[lo], windows[hi])):
        window = numpy.zeros((n_channels, int(end) - int(start)), dtype=numpy.float32)
        a, b = int(offsets[index]), int(offsets[index + 1])
        window[chan[a:b], pos[a:b]] = val[a:b]
        out.append(window)
    return out


def test_group_track_should_index_its_offsets_by_window_alone(tracks):
    """Test the shape and dtype of the grouped layout.

    Given:
        Two channels and two windows.
    When:
        They are grouped.
    Then:
        ``offsets`` should be one-dimensional over windows with one
        more entry than there are windows, and the three row arrays
        should carry the narrow dtypes the layout specifies.

    There is no channel axis in ``offsets``: every channel's rows for a
    window live in that window's one run, with the channel index beside
    them. Upstream's docstring said otherwise for both engines and the
    code never matched it.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8)])

    # Act
    offsets, pos, val, chan, n_channels = cache.group_track(windows, tracks)

    # Assert
    assert offsets.ndim == 1
    assert len(offsets) == len(windows) + 1
    assert offsets.dtype == numpy.int64
    assert (pos.dtype, val.dtype, chan.dtype) == (
        numpy.int32,
        numpy.float32,
        numpy.int16,
    )
    assert n_channels == 2


def test_group_track_should_densify_to_what_a_coordinate_read_returns(tracks):
    """Test the grouping against the reader, which is the whole contract.

    Given:
        Windows over both contigs, one holding rows from both channels
        and one holding rows from only the first.
    When:
        They are grouped and each window's run is scattered back into a
        dense array.
    Then:
        Every window should equal what `TrackStore` reads over the same
        interval, so the grouped form and the coordinate form describe
        one signal.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8), ("chr1", 395, 405)])
    store = stores.TrackStore(tracks)

    # Act
    dense = densify(cache.group_track(windows, tracks), windows)

    # Assert
    for index, (chrom, start, end) in enumerate(
        zip(windows["chrom"], windows["sig_start"], windows["sig_end"])
    ):
        expected = store.window(chrom, start, end).numpy()
        assert numpy.array_equal(dense[index], expected), index


def test_group_track_should_offset_positions_from_the_windows_own_start(tracks):
    """Test what ``pos`` is measured from.

    Given:
        A window starting well inside a contig, covering one known row.
    When:
        It is grouped.
    Then:
        The stored position should be the distance from the window's
        start, not the contig coordinate, so a read needs no arithmetic
        against the window it came from.
    """

    # Arrange
    windows = frame([("chr1", 396, 404)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, tracks)

    # Assert
    assert pos.tolist() == [400 - 396]
    assert val.tolist() == [7.0]
    assert chan.tolist() == [0]


def test_group_track_should_keep_only_what_the_windows_cover(tracks):
    """Test that unwindowed rows are not read into the layout.

    Given:
        A window covering one of channel 0's three rows on a contig.
    When:
        It is grouped.
    Then:
        It should hold that row alone, because keeping only what
        training can ask for is what makes the cache affordable at all.
    """

    # Arrange
    windows = frame([("chr1", 11, 13)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, tracks)

    # Assert
    assert int(offsets[-1]) == 2
    assert sorted(zip(chan.tolist(), val.tolist())) == [(0, 3.0), (1, 5.0)]


def test_group_track_should_store_a_position_once_per_window_that_covers_it(tracks):
    """Test what overlapping windows do to a shared position.

    Given:
        Two windows that both cover the same row.
    When:
        They are grouped.
    Then:
        Each window's run should carry its own copy, so a read is a
        slice of one window's rows rather than a search across windows.

    Duplication is the stated price of an addressed read: upstream
    measured about 2.4x on the signal span for a real locus set.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr1", 9, 17)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, tracks)

    # Assert
    assert [int(offsets[1] - offsets[0]), int(offsets[2] - offsets[1])] == [3, 3]
    assert int(offsets[-1]) == 6


def test_group_track_should_group_over_the_columns_bounds_names(tracks):
    """Test that the bounds parameter chooses the span.

    Given:
        One window frame carrying a wide input span and a narrow output
        span under different column names.
    When:
        It is grouped over each pair in turn.
    Then:
        The wider pair should cover rows the narrower one excludes, so
        a control can be grouped over the input window while the signal
        is grouped over the output window.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    windows = polars.DataFrame(
        {
            "chrom": ["chr1"],
            "in_start": [8],
            "in_end": [16],
            "sig_start": [11],
            "sig_end": [13],
        }
    )

    # Act
    wide = cache.group_track(windows, tracks, bounds=("in_start", "in_end"))
    narrow = cache.group_track(windows, tracks, bounds=("sig_start", "sig_end"))

    # Assert
    assert int(wide[0][-1]) == 3
    assert int(narrow[0][-1]) == 2


def test_group_track_should_count_a_trailing_channel_that_recorded_nothing(corpus):
    """Test the channel count when the last channel is empty.

    Given:
        A channel set whose final channel has no rows in any window.
    When:
        It is grouped.
    Then:
        It should still report that channel, because the arrays cannot
        recover it: a channel with no rows leaves no trace in the
        channel index, so a reader would build a dense array one row
        short.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    windows = frame([("chr2", 0, 8)])

    # Act
    grouped = cache.group_track(windows, corpus.tracks)

    # Assert
    assert grouped[4] == 2
    assert densify(grouped, windows)[0].shape == (2, 8)


def test_group_track_should_answer_empty_arrays_for_no_windows(tracks):
    """Test the grouping of an empty window set.

    Given:
        No windows.
    When:
        They are grouped.
    Then:
        It should answer a single zero offset and three empty arrays,
        so a locus set that filtered down to nothing is a shape rather
        than an error.
    """

    # Act
    offsets, pos, val, chan, n_channels = cache.group_track(frame([]), tracks)

    # Assert
    assert offsets.tolist() == [0]
    assert (len(pos), len(val), len(chan)) == (0, 0, 0)
    assert n_channels == 2


def test_group_track_should_refuse_more_channels_than_the_index_holds():
    """Test the refusal of a channel set the layout cannot index.

    Given:
        More channels than an ``int16`` channel index can distinguish.
    When:
        They are grouped.
    Then:
        It should raise `ValueError` naming both counts, before reading
        anything, because the channel index would otherwise wrap and
        label rows with the wrong channel.

    Checked above the scans rather than inside one engine's branch, as
    upstream had it: the limit is the layout's, not any one way of
    filling it.
    """

    # Arrange
    too_many = ["unread"] * (numpy.iinfo(numpy.int16).max + 1)

    # Act & assert
    with pytest.raises(ValueError, match="more channels than"):
        cache.group_track(frame([("chr1", 8, 16)]), too_many)


def test_group_track_should_release_the_tables_it_registers(tracks):
    """Test that the grouping leaves nothing bound in the session.

    Given:
        A grouping that registers a window frame and a track frame per
        contig and per channel.
    When:
        It finishes.
    Then:
        Neither table should still be bound, because the session is
        process-global and the next caller of either name would read
        this one's rows rather than failing.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")

    # Act
    cache.group_track(frame([("chr1", 8, 16)]), tracks)

    # Assert
    for name in ("quadrat_windows", "quadrat_track"):
        with pytest.raises(BaseException):
            pb.ctx.table(name)


def test_window_group_sums_should_total_each_group_over_its_channels(tracks):
    """Test the per-group totals against the rows they come from.

    Given:
        Two channels pooled into one group, and two windows.
    When:
        The group sums are taken.
    Then:
        Each window should total every row both channels recorded
        inside it, so pooling is a sum over channels rather than a
        separate read.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8)])

    # Act
    totals = cache.window_group_sums(windows, tracks, [2])

    # Assert
    assert totals.tolist() == [[9.0], [2.0]]


def test_window_group_sums_should_keep_the_groups_apart(tracks):
    """Test that two groups are totalled separately.

    Given:
        Two channels, each its own group.
    When:
        The group sums are taken.
    Then:
        Each column should carry only its own channel's rows, so the
        grouping decides what is pooled and nothing else does.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])

    # Act
    totals = cache.window_group_sums(windows, tracks, [1, 1])

    # Assert
    assert totals.tolist() == [[4.0, 5.0]]


def test_window_group_sums_should_refuse_a_grouping_that_misses_a_channel(tracks):
    """Test the refusal of a grouping that does not cover the channels.

    Given:
        Two channels and a grouping summing to one.
    When:
        The group sums are taken.
    Then:
        It should raise `ValueError` naming both counts, because every
        channel belongs to exactly one group and a short grouping would
        silently drop the rest.
    """

    # Act & assert
    with pytest.raises(ValueError, match="group_sizes sums to 1"):
        cache.window_group_sums(frame([("chr1", 8, 16)]), tracks, [1])


def test_window_group_sums_should_refuse_a_wrong_grouping_with_no_windows(tracks):
    """Test that an empty window set does not excuse a wrong grouping.

    Given:
        A grouping that does not cover the channels, and no windows.
    When:
        The group sums are taken.
    Then:
        It should still raise, because an inconsistent grouping is the
        caller's mistake whether or not there is a window to apply it
        to.

    Upstream checked below its empty-window return, so the same
    argument was accepted on an empty locus set and refused on a
    non-empty one.
    """

    # Act & assert
    with pytest.raises(ValueError, match="group_sizes sums to 1"):
        cache.window_group_sums(frame([]), tracks, [1])


class TestWindowSignalCache:
    """The grouped cache, read by draw."""

    @pytest.mark.parametrize("residency", ["memory", "scratch"])
    def test_window_should_read_back_what_a_coordinate_read_returns(
        self, tracks, residency
    ):
        """Test the cache against the reader, over every window.

        Given:
            A cache built for three windows, in each residency.
        When:
            Each window is read at no jitter.
        Then:
            It should equal what `TrackStore` reads over the same
            interval, so the cache and a coordinate read describe one
            signal whichever residency holds the arrays.
        """

        # Arrange
        windows = frame([("chr1", 8, 16), ("chr2", 0, 8), ("chr1", 396, 404)])
        store = stores.TrackStore(tracks)

        # Act
        built = cache.WindowSignalCache.build(windows, tracks, residency=residency)

        # Assert
        for index, (chrom, start, end) in enumerate(
            zip(windows["chrom"], windows["sig_start"], windows["sig_end"])
        ):
            expected = store.window(chrom, start, end).numpy()
            read = built.window(index, 0, int(end) - int(start)).numpy()
            assert numpy.array_equal(read, expected), index

    def test_window_should_answer_identically_from_either_residency(self, tracks):
        """Test that residency is invisible in the answer.

        Given:
            Two caches over the same windows, one in memory and one
            mapped from scratch files.
        When:
            The same draw is read from each.
        Then:
            They should agree exactly, because residency is a choice
            about where the bytes live and not about what they say.
        """

        # Arrange
        windows = frame([("chr1", 8, 16)])

        # Act
        held = cache.WindowSignalCache.build(windows, tracks, residency="memory")
        mapped = cache.WindowSignalCache.build(windows, tracks, residency="scratch")

        # Assert
        assert held.window(0, 0, 8).tolist() == mapped.window(0, 0, 8).tolist()

    def test_window_should_shift_the_draw_by_the_jitter(self, tracks):
        """Test what the jitter offset does to a read.

        Given:
            A cache over one window with rows at known offsets.
        When:
            The window is read at a jitter of two.
        Then:
            Every row should move two columns toward the start, because
            the jitter is the draw's displacement within the window
            rather than a second coordinate to search by.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        read = built.window(0, 2, 8).numpy()

        # Assert
        assert read[0].tolist() == [1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert read[1].tolist() == [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_window_should_drop_a_row_the_jitter_moves_before_the_start(self, tracks):
        """Test a draw jittered past the low edge.

        Given:
            A cache whose window holds a row two columns in.
        When:
            It is read at a jitter of three, moving that row before the
            start.
        Then:
            That row should be dropped and the rest kept, so a draw
            near a contig edge reads as zeros there rather than raising
            or wrapping.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        read = built.window(0, 3, 8).numpy()

        # Assert
        assert read[0].tolist() == [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert read[1].tolist() == [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_window_should_drop_a_row_that_falls_past_the_length(self, tracks):
        """Test a draw shorter than the window it was grouped over.

        Given:
            A cache grouped over an eight-base window.
        When:
            Three bases of it are read.
        Then:
            Only the rows inside those three should appear, so the
            length asked for bounds the answer rather than the span the
            grouping used.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        read = built.window(0, 0, 3).numpy()

        # Assert
        assert tuple(read.shape) == (2, 3)
        assert read[0].tolist() == [0.0, 0.0, 1.0]
        assert read[1].sum() == 0

    def test_window_should_mirror_the_draw_when_reverse_complemented(self, tracks):
        """Test the mirrored read against gathering and flipping.

        Given:
            A cache over one window.
        When:
            The draw is read mirrored, and read plainly and flipped.
        Then:
            The two should agree, so applying the mirror during the
            scatter is the same answer as the two passes over the dense
            tensor it replaces.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        mirrored = built.window(0, 0, 8, reverse_complement=True).numpy()

        # Assert
        plain = built.window(0, 0, 8).numpy()
        assert numpy.array_equal(mirrored, numpy.flip(plain, axis=1))

    def test_window_should_permute_the_channels_of_a_mirrored_draw(self, tracks):
        """Test the channel permutation a mirror applies.

        Given:
            A cache over two channels and a permutation swapping them.
        When:
            The draw is read mirrored with that permutation.
        Then:
            It should equal flipping the plain read and then swapping
            its rows, because a stranded pair exchanges roles under
            reverse-complement while the positions mirror.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )
        swap = numpy.array([1, 0])

        # Act
        mirrored = built.window(0, 0, 8, reverse_complement=True, perm=swap).numpy()

        # Assert
        plain = numpy.flip(built.window(0, 0, 8).numpy(), axis=1)
        assert numpy.array_equal(mirrored, plain[swap])

    def test_window_should_apply_a_permutation_that_is_not_an_involution(
        self, three_channel
    ):
        """Test a permutation that is not its own inverse.

        Given:
            Three channels and a three-cycle permutation.
        When:
            The draw is read mirrored with it.
        Then:
            Each channel's row should land where the permutation sends
            it rather than where its inverse would, which a swap of two
            channels cannot distinguish.
        """

        # Arrange
        pytest.importorskip("polars_bio")
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), three_channel.tracks, residency="memory"
        )
        cycle = numpy.array([1, 2, 0])

        # Act
        mirrored = built.window(0, 0, 8, reverse_complement=True, perm=cycle).numpy()

        # Assert
        plain = numpy.flip(built.window(0, 0, 8).numpy(), axis=1)
        assert numpy.array_equal(mirrored[cycle], plain)

    def test_n_channels_should_report_what_the_arrays_cannot_recover(self, corpus):
        """Test the channel count of a cache whose last channel is empty.

        Given:
            A cache whose final channel recorded nothing in any window.
        When:
            Its channel count is read and a window taken.
        Then:
            Both should include that channel, because a channel with no
            rows leaves no trace in the channel index and a reader
            would otherwise build a dense array one row short.
        """

        # Arrange
        pytest.importorskip("polars_bio")

        # Act
        built = cache.WindowSignalCache.build(
            frame([("chr2", 0, 8)]), corpus.tracks, residency="memory"
        )

        # Assert
        assert built.n_channels == 2
        assert tuple(built.window(0, 0, 8).shape) == (2, 8)

    def test_nbytes_should_total_the_four_arrays(self, tracks):
        """Test what a cache reports for its own size.

        Given:
            A cache over one window.
        When:
            Its byte count is taken.
        Then:
            It should be the sum over the four arrays, which is what
            makes the sparse layout's cost measurable rather than
            asserted.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act & assert
        assert built.nbytes() == 2 * 8 + 3 * (4 + 4 + 2)

    def test_window_should_answer_the_same_after_a_pickle_round_trip(self, tracks):
        """Test the crossing a spawned worker makes.

        Given:
            A mapped cache that has already read a window, so its
            arrays are open.
        When:
            It is pickled, restored, and asked again.
        Then:
            It should answer identically, which is the whole point of
            the mapped residency: under spawn the arrays are not
            inherited, and mapping them keeps the cost of reaching the
            first batch flat in cache size instead of linear.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="scratch"
        )
        expected = built.window(0, 0, 8).tolist()

        # Act
        restored = pickle.loads(pickle.dumps(built))

        # Assert
        assert restored.window(0, 0, 8).tolist() == expected

    def test_from_arrays_should_keep_the_arrays_in_memory_under_fork(
        self, mocker, tracks
    ):
        """Test the residency chosen when workers are forked.

        Given:
            A start method of ``fork`` and no residency asked for.
        When:
            A cache is built.
        Then:
            It should hold the arrays in memory, because a forked
            worker already shares them copy-on-write and a mapping
            would only add the page-fault indirection.
        """

        # Arrange
        mocker.patch.object(multiprocessing, "get_start_method", return_value="fork")

        # Act
        built = cache.WindowSignalCache.build(frame([("chr1", 8, 16)]), tracks)

        # Assert
        assert built.scratch is None

    def test_from_arrays_should_map_the_arrays_under_spawn(self, mocker, tracks):
        """Test the residency chosen when workers are spawned.

        Given:
            A start method of ``spawn`` and no residency asked for.
        When:
            A cache is built.
        Then:
            It should write the arrays for mapping, because a spawned
            worker inherits no sharing and would otherwise be sent one
            pickled copy each.
        """

        # Arrange
        mocker.patch.object(multiprocessing, "get_start_method", return_value="spawn")

        # Act
        built = cache.WindowSignalCache.build(frame([("chr1", 8, 16)]), tracks)

        # Assert
        assert built.scratch is not None

    def test_from_arrays_should_map_the_arrays_when_no_start_method_answers(
        self, mocker, tracks
    ):
        """Test the residency chosen where the start method is unavailable.

        Given:
            A start method that raises.
        When:
            A cache is built with no residency asked for.
        Then:
            It should write the arrays for mapping, which is correct
            everywhere and only slower, rather than refusing to build.
        """

        # Arrange
        mocker.patch.object(
            multiprocessing,
            "get_start_method",
            side_effect=RuntimeError("no start method"),
        )

        # Act
        built = cache.WindowSignalCache.build(frame([("chr1", 8, 16)]), tracks)

        # Assert
        assert built.scratch is not None

    def test_from_arrays_should_refuse_an_unknown_residency(self, tracks):
        """Test the refusal of a residency that is neither.

        Given:
            A residency this does not implement.
        When:
            A cache is built.
        Then:
            It should raise `ValueError` naming both choices and
            ``None``, rather than silently taking one of them.
        """

        # Act & assert
        with pytest.raises(ValueError, match="residency must be"):
            cache.WindowSignalCache.build(
                frame([("chr1", 8, 16)]), tracks, residency="mmap"
            )

    def test_build_should_remove_its_scratch_when_the_cache_is_dropped(self, tracks):
        """Test that a cache cleans up the directory it made.

        Given:
            A mapped cache, which made its own scratch directory.
        When:
            The last reference to it is dropped.
        Then:
            The directory should be gone, because nothing else knows
            about it and a training run that built a hundred would
            otherwise fill the disk.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="scratch"
        )
        root = built.scratch
        assert os.path.isdir(root)

        # Act
        del built
        gc.collect()

        # Assert
        assert not os.path.exists(root)

    def test_adopt_should_leave_the_directory_when_the_cache_is_dropped(
        self, tmp_path, tracks
    ):
        """Test that an adopted entry outlives the cache reading it.

        Given:
            Arrays on disk, adopted rather than built.
        When:
            The cache is dropped.
        Then:
            The directory should remain, because a persistent entry
            outlives every cache that reads it and the first run to
            finish would otherwise delete what every later run is
            counting on.
        """

        # Arrange
        grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)
        root = str(tmp_path / "entry")
        os.makedirs(root)
        cache.write_cache_arrays(root, *grouped[:4])
        adopted = cache.WindowSignalCache.adopt(root, grouped[4])
        assert adopted.window(0, 0, 8).numpy().sum() > 0

        # Act
        del adopted
        gc.collect()

        # Assert
        assert os.path.isdir(root)

    def test_adopt_should_never_own_its_files_across_a_pickle(self, tmp_path, tracks):
        """Test what a worker's copy of an adopted cache owns.

        Given:
            An adopted cache, pickled and restored as a worker's copy
            would be.
        When:
            The restored copy is asked whether it owns its files.
        Then:
            It should not, because ownership travels with the pickle
            and a worker that exited first would otherwise delete the
            arrays every other worker is still reading.
        """

        # Arrange
        grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)
        root = str(tmp_path / "shared")
        os.makedirs(root)
        cache.write_cache_arrays(root, *grouped[:4])
        adopted = cache.WindowSignalCache.adopt(root, grouped[4])

        # Act
        restored = pickle.loads(pickle.dumps(adopted))

        # Assert
        assert restored.owns_scratch is False
        assert os.path.isdir(root)
