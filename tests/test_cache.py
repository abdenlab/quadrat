"""Tests for the sparse window cache: grouping, draws and persistence.

Three things in order. The grouping, which turns a track set and a
window set into the layout; the cache over that layout, which answers
a draw; and the persisted entry, which commits one and reads it back
or misses. The array writer and the counts read off the arrays come
just ahead of the cache, whose `adopt` tests arrange with them.

The contract running through all three is that the grouped form
describes the same signal a coordinate read describes. So the central
assertion is a cross-check rather than a golden array: densify what
the grouping returns and compare it against `quadrat.stores.TrackStore`,
window by window. One path reads a contig at a time through polars,
the other groups through the engine, and agreement is a statement
about both.

The store is the oracle, rather than a second writer of the same
layout, because two writers can share a mistake and agree on it; a
reader that never sees the layout cannot.

Two things are easy to get wrong here. A window's bounds are
half-open, and no row in the corpus above sits on an edge of any
window the rest of the suite asks for, so `EDGE_ROWS` and the two
tests over it are what keep either bound from moving a base unnoticed.
And a draw takes a jitter, a length and a mirror; each is covered
alone, and the combinations are covered by the mirroring property
among the `window` tests of `TestWindowSignalCache`, because the
mirror absorbing the jitter a second time is a mistake no
single-input example can see.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import gc
import inspect
import json
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile

import numpy
import polars
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import quadrat
from quadrat import cache, stores
from tests import FILE_BACKED, TWO_CONTIGS, builders, quadrat_tables, run_child

# Channel 0 puts two rows inside one window and one on a second
# contig; channel 1 interleaves with channel 0 inside that window and
# records nothing on the second contig.
CHANNEL_0 = [("chr1", 10, 1.0), ("chr1", 12, 3.0), ("chr1", 400, 7.0), ("chr2", 5, 2.0)]


CHANNEL_1 = [("chr1", 11, 5.0)]


# Rows sitting exactly on a window's bounds. The span is half-open, so
# the row at the start is grouped and the row at the end is not, and no
# row in the channels above touches an edge of any window the suite
# asks for -- which leaves the interval join and both of
# `window_group_sums`' searches free to move a base unnoticed.
EDGE_ROWS = [("chr1", 8, 11.0), ("chr1", 12, 3.0), ("chr1", 16, 13.0)]


# One position recorded twice, which the format does not promise and
# nothing rejects. Every reader of a track has to answer one total for
# it, so this is the corpus each reader is checked over.
DUPLICATE_ROWS = [("chr1", 10, 1.0), ("chr1", 10, 4.0), ("chr1", 12, 3.0)]


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    return corpus_factory(
        "cache",
        TWO_CONTIGS,
        tracks=[CHANNEL_0, CHANNEL_1],
        in_window=60,
        out_window=30,
        max_jitter=6,
    )


@pytest.fixture(scope="module")
def three_channel(corpus_factory):
    return corpus_factory(
        "cache-three",
        TWO_CONTIGS,
        tracks=[CHANNEL_0, CHANNEL_1, [("chr1", 13, 9.0)]],
    )


@pytest.fixture(scope="module")
def edges(corpus_factory):
    return corpus_factory("cache-edges", TWO_CONTIGS, tracks=[EDGE_ROWS])


@pytest.fixture(scope="module")
def duplicates(corpus_factory):
    return corpus_factory("cache-duplicates", TWO_CONTIGS, tracks=[DUPLICATE_ROWS])


@pytest.fixture
def duplicate_tracks(duplicates):
    pytest.importorskip("polars_bio")
    return duplicates.tracks


@pytest.fixture
def edge_tracks(edges):
    pytest.importorskip("polars_bio")
    return edges.tracks


@pytest.fixture
def tracks(corpus):
    pytest.importorskip("polars")
    pytest.importorskip("polars_bio")
    return corpus.tracks


def shaped_track(shape, path, rows):
    """Return one channel's rows in one of the four shapes a reader takes."""

    if shape == "directory":
        return builders.write_track(str(path) + quadrat.TRACK_SUFFIX, rows)
    flat = builders.write_flat_track(str(path) + ".parquet", rows)
    if shape == "single_file":
        return flat
    eager = polars.read_parquet(flat)
    return eager if shape == "eager" else eager.lazy()


def frame(rows, columns=("sig_start", "sig_end")):
    """Return a window frame built from ``(chrom, start, end)`` rows."""

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


def held_track(rows):
    """Return ``(chrom, pos, value)`` rows as an in-memory track, in the order given."""

    return polars.DataFrame(
        {
            "chrom": [str(row[0]) for row in rows],
            "pos": [int(row[1]) for row in rows],
            "value": [float(row[2]) for row in rows],
        },
        schema={"chrom": polars.Utf8, "pos": polars.Int64, "value": polars.Float64},
    )


def densify(grouped, windows, columns=("sig_start", "sig_end")):
    """Return the dense array per window that a grouped result describes."""

    offsets, pos, val, chan, n_channels = grouped
    lo, hi = columns
    out = []
    for index, (start, end) in enumerate(zip(windows[lo], windows[hi])):
        window = numpy.zeros((n_channels, int(end) - int(start)), dtype=numpy.float32)
        a, b = int(offsets[index]), int(offsets[index + 1])
        numpy.add.at(window, (chan[a:b], pos[a:b]), val[a:b])
        out.append(window)
    return out


# Windows inside the shared corpus's first contig, as a locus set will
# present them: any count including none, any span including empty.
window_sets = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=600), st.integers(min_value=0, max_value=600)
    ),
    min_size=0,
    max_size=8,
)


def unfiled_inputs(**overrides):
    """Return a key's input set that touches no file, with ``overrides`` applied.

    No tracks and no locus files, and a reference at an absolute path
    where nothing exists, which hashes as absent: so the key depends on
    the arguments alone and not on where the suite runs, and a property
    can take hundreds of keys without writing anything.
    """

    return dict(
        {
            "signals": [],
            "sequences": "/quadrat-golden-key/absent.fa",
            "peaks": [],
            "negatives": [],
        },
        **overrides,
    )


# Contig names a subset of which a key can be taken over.
contig_subsets = st.lists(
    st.sampled_from(["chr1", "chr2", "chr3", "chrX", "chrM"]), unique=True
).map(sorted)


# Channel group sizes, as a grouping argument takes them.
group_sizes = st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=4)


# Count thresholds a key hashes: any finite float.
thresholds = st.floats(allow_nan=False, allow_infinity=False)


def inputs(root, *, peaks=(("chr1", 8, 16),), track_rows=None):
    """Return a key's full input set of a track, a reference and two locus files.

    The reference is indexed, because the key hashes the index rather
    than the sequence: without one, every spelling of the path hashes
    as absent and a test comparing two spellings would agree for the
    wrong reason.
    """

    pytest.importorskip("polars")
    pytest.importorskip("pysam")
    root = str(root)
    return {
        "signals": [
            builders.write_track(
                os.path.join(root, "s0.track.parquet"),
                CHANNEL_0 if track_rows is None else track_rows,
            )
        ],
        "sequences": builders.write_fasta(
            os.path.join(root, "genome.fa"), {"chr1": 600, "chr2": 500}, index=True
        ),
        "peaks": builders.write_bed(os.path.join(root, "peaks.bed"), peaks),
        "negatives": builders.write_bed(
            os.path.join(root, "negatives.bed"), [("chr1", 100, 108)]
        ),
    }


def test_group_track_should_index_its_offsets_by_window_alone(tracks):
    """Test the shape and dtype of the grouped layout.

    Given:
        Two channels and three windows, so that a count over windows
        and a count over channels cannot be mistaken for each other.
    When:
        They are grouped.
    Then:
        ``offsets`` should be one-dimensional over windows with one
        more entry than there are windows, and the three row arrays
        should carry the narrow dtypes the layout specifies.

    There is no channel axis in ``offsets``: every channel's rows for a
    window live in that window's one run, with the channel index beside
    them.

    The window count has to differ from the channel count for this to
    say anything. At two and two, both candidate lengths are three, so
    ``offsets`` sized by channel would pass a test written to rule
    exactly that out.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8), ("chr1", 395, 405)])

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
        A window covering one of channel 0's three rows on a contig and
        the one row channel 1 records there.
    When:
        It is grouped.
    Then:
        It should hold those two rows alone, one from each channel,
        because keeping only what training can ask for is what makes
        the cache affordable at all.
    """

    # Arrange
    windows = frame([("chr1", 11, 13)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, tracks)

    # Assert
    assert int(offsets[-1]) == 2
    assert sorted(zip(chan.tolist(), val.tolist())) == [(0, 3.0), (1, 5.0)]


def test_group_track_should_cover_a_window_closed_at_its_start_and_open_at_its_end(
    edge_tracks,
):
    """Test the bounds of the span a window covers.

    Given:
        A track whose rows include one at exactly a window's start and
        one at exactly its end.
    When:
        The window is grouped.
    Then:
        It should hold the row at the start and the one inside it, and
        not the row at the end, because the span is half-open -- so a
        row on a boundary belongs to one window rather than to both.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, edge_tracks)

    # Assert
    assert sorted(zip(pos.tolist(), val.tolist())) == [(0, 11.0), (4, 3.0)]
    assert int(offsets[-1]) == 2


def test_group_track_should_answer_the_same_arrays_however_the_rows_are_ordered():
    """Test that the grouped arrays are a function of the rows alone.

    Given:
        Three channels recording the same positions inside two
        overlapping windows, one of which records a position twice
        with different values, supplied once as written and once with
        every channel's rows reversed.
    When:
        Both are grouped over the same windows.
    Then:
        All four arrays should be identical, because a rebuild has to
        reproduce a cache byte for byte, and rows tied on window and
        offset -- one per channel at a shared position, two for the
        repeated one -- otherwise come out in whatever order the engine
        returned them.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    channels = [
        [("chr1", 10, 1.0), ("chr1", 12, 3.0), ("chr1", 10, 4.0), ("chr1", 14, 2.0)],
        [("chr1", 10, 5.0), ("chr1", 12, 6.0), ("chr1", 14, 9.0)],
        [("chr1", 12, 8.0), ("chr1", 10, 7.0), ("chr1", 11, 0.5)],
    ]
    windows = frame([("chr1", 8, 16), ("chr1", 9, 17)])

    # Act
    forward = cache.group_track(windows, [held_track(rows) for rows in channels])
    backward = cache.group_track(windows, [held_track(rows[::-1]) for rows in channels])

    # Assert
    for name, one, other in zip(("offsets", "pos", "val", "chan"), forward, backward):
        assert numpy.array_equal(one, other), name
    assert forward[4] == backward[4]


# One channel's rows on the first contig: any positions, a position
# repeated allowed, every value distinct so no two rows are identical.
channel_rows = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=600),
        st.floats(width=32, allow_nan=False, allow_infinity=False),
    ),
    max_size=10,
    unique_by=lambda row: row[1],
)


# No deadline: every example plans engine queries, and how long that
# takes is not this property's concern.
@settings(deadline=None, max_examples=25)
@given(
    spans=window_sets,
    channels=st.lists(channel_rows, min_size=1, max_size=3),
    data=st.data(),
)
def test_group_track_should_answer_the_same_arrays_for_any_order_of_the_rows(
    spans, channels, data
):
    """Test that the grouped arrays are a function of the rows alone, over the domain.

    Given:
        Any window set over one contig and up to three channels of rows
        on it, a position repeated allowed, supplied once as drawn and
        once with each channel's rows in a drawn permutation.
    When:
        Both are grouped over the same windows.
    Then:
        All four arrays should be identical, because a rebuild has to
        reproduce a cache byte for byte whatever order the rows arrive
        in.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    windows = frame([("chr1", min(a, b), max(a, b)) for a, b in spans])
    drawn = [[("chr1", pos, value) for pos, value in rows] for rows in channels]
    permuted = [data.draw(st.permutations(rows)) for rows in drawn]

    # Act
    forward = cache.group_track(windows, [held_track(rows) for rows in drawn])
    shuffled = cache.group_track(windows, [held_track(rows) for rows in permuted])

    # Assert
    for name, one, other in zip(("offsets", "pos", "val", "chan"), forward, shuffled):
        assert numpy.array_equal(one, other), name


def test_group_track_should_store_a_position_once_per_window_that_covers_it(tracks):
    """Test what overlapping windows do to a shared position.

    Given:
        Two windows that both cover the same row.
    When:
        They are grouped.
    Then:
        Each window's run should carry its own copy, so a read is a
        slice of one window's rows rather than a search across windows.

    Duplication is the stated price of an addressed read, and it grows
    with how much the windows of a locus set overlap.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr1", 9, 17)])

    # Act
    offsets, pos, val, chan, _ = cache.group_track(windows, tracks)

    # Assert
    assert [int(offsets[1] - offsets[0]), int(offsets[2] - offsets[1])] == [3, 3]
    assert int(offsets[-1]) == 6
    # The rows themselves, each measured from its own window's start,
    # because the lengths above hold for any two runs of three -- with
    # every stored position zeroed, or with a copy in the wrong run,
    # which is what this test exists to rule out. Sorted, so the check
    # does not rest on the order within a run.
    runs = [
        sorted(
            zip(
                pos[int(offsets[index]) : int(offsets[index + 1])].tolist(),
                chan[int(offsets[index]) : int(offsets[index + 1])].tolist(),
                val[int(offsets[index]) : int(offsets[index + 1])].tolist(),
            )
        )
        for index in range(2)
    ]
    assert runs[0] == [(2, 0, 1.0), (3, 1, 5.0), (4, 0, 3.0)]
    assert runs[1] == [(1, 0, 1.0), (2, 1, 5.0), (3, 0, 3.0)]


def test_group_track_should_hold_an_empty_run_for_a_zero_length_window(tracks):
    """Test a window whose start and end are the same position.

    Given:
        A zero-length window between two windows that hold rows.
    When:
        They are grouped.
    Then:
        Its run should be empty and both neighbours' runs unchanged, so
        a degenerate window is a shape rather than an error or a
        shifted index for everything after it.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr1", 12, 12), ("chr1", 11, 13)])

    # Act
    offsets, _, _, _, _ = cache.group_track(windows, tracks)

    # Assert
    assert int(offsets[2] - offsets[1]) == 0
    assert int(offsets[1] - offsets[0]) == 3
    assert int(offsets[3] - offsets[2]) == 2


def test_group_track_should_hold_an_empty_run_for_a_contig_no_track_has(tracks):
    """Test a window on a contig absent from every track.

    Given:
        A window set naming a contig no track holds, beside one that
        holds rows.
    When:
        They are grouped.
    Then:
        The absent contig's run should be empty and the other's its own
        rows, because a contig with no signal is routine for the small
        unplaced ones and reads as no signal rather than as an error.
    """

    # Arrange
    windows = frame([("chr3", 0, 8), ("chr1", 8, 16)])

    # Act
    offsets, _, _, _, _ = cache.group_track(windows, tracks)

    # Assert
    assert int(offsets[1] - offsets[0]) == 0
    assert int(offsets[2] - offsets[1]) == 3


def test_group_track_should_read_a_track_it_is_handed_in_memory(tracks):
    """Test grouping a track that was never written to disk.

    Given:
        One track's rows as a polars frame, and the same rows on disk.
    When:
        Both are grouped over the same windows.
    Then:
        They should produce the same arrays, because the whole-table
        reader accepts an in-memory frame and that is what lets a
        caller go from alignments to tensors without writing a track.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    held = stores.track_frame(tracks[0]).collect()

    # Act
    from_disk = cache.group_track(windows, [tracks[0]])
    from_memory = cache.group_track(windows, [held])

    # Assert
    for one, other in zip(from_disk[:4], from_memory[:4]):
        assert numpy.array_equal(one, other)
    assert from_disk[4] == from_memory[4]


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
    wide = cache.group_track(windows, tracks, bound_columns=("in_start", "in_end"))
    narrow = cache.group_track(windows, tracks, bound_columns=("sig_start", "sig_end"))

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


def test_group_track_should_group_a_contig_only_some_channels_record(tmp_path):
    """Test a contig where one channel has rows and another has none.

    Given:
        Three channels, of which only the middle one records anything
        on the second contig, and the last records nothing anywhere.
    When:
        Windows on both contigs are grouped.
    Then:
        Every row should land under the channel that recorded it, and
        the channel count should still be three, because a channel
        that recorded nothing is still a channel the layout reserves.

    One query answers for every channel on a contig, so the parts
    with nothing in them have to be left out of it rather than joined
    as empty -- and leaving out the wrong one would silently shift
    every later channel's rows onto its neighbour.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    tracks = [
        shaped_track("directory", tmp_path / "t0", [("chr1", 10, 1.0)]),
        shaped_track(
            "directory", tmp_path / "t1", [("chr1", 11, 5.0), ("chr2", 5, 2.0)]
        ),
        shaped_track("directory", tmp_path / "t2", [("chr1", 900, 9.0)]),
    ]
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8)])

    # Act
    offsets, pos, val, chan, n_channels = cache.group_track(windows, tracks)

    # Assert
    assert n_channels == 3
    first = sorted(
        zip(
            chan[int(offsets[0]) : int(offsets[1])].tolist(),
            val[int(offsets[0]) : int(offsets[1])].tolist(),
        )
    )
    second = sorted(
        zip(
            chan[int(offsets[1]) : int(offsets[2])].tolist(),
            val[int(offsets[1]) : int(offsets[2])].tolist(),
        )
    )
    assert first == [(0, 1.0), (1, 5.0)]
    assert second == [(1, 2.0)]


@settings(FILE_BACKED, max_examples=25)
@given(spans=window_sets)
def test_group_track_should_hold_the_layout_invariants_for_any_window_set(tracks, spans):
    """Test the layout's own invariants over arbitrary window sets.

    Given:
        Any number of windows over one contig, at any spans including
        empty ones.
    When:
        They are grouped.
    Then:
        The offsets should start at zero, never decrease, and end at
        the length of each row array, and every channel index should
        be one of the channels -- because every read of this layout
        slices on those offsets and indexes a dense row by that
        channel.
    """

    # Arrange
    windows = frame([("chr1", min(a, b), max(a, b)) for a, b in spans])

    # Act
    offsets, pos, val, chan, n_channels = cache.group_track(windows, tracks)

    # Assert
    assert int(offsets[0]) == 0
    assert offsets.tolist() == sorted(offsets.tolist())
    assert int(offsets[-1]) == len(pos) == len(val) == len(chan)
    assert len(offsets) == len(windows) + 1
    if len(chan):
        assert 0 <= int(chan.min()) and int(chan.max()) < n_channels


@settings(FILE_BACKED, max_examples=25)
@given(spans=window_sets)
def test_group_track_should_densify_to_the_reader_for_any_window_set(tracks, spans):
    """Test the grouping against the reader over arbitrary window sets.

    Given:
        Any number of windows over one contig, at any spans.
    When:
        Each grouped run is scattered back into a dense array.
    Then:
        Every window should equal what the coordinate reader answers
        for the same span, which is the contract the example-based
        oracle checks over five geometries and this checks over the
        domain.
    """

    # Arrange
    windows = frame([("chr1", min(a, b), max(a, b)) for a, b in spans])
    store = stores.TrackStore(tracks)

    # Act
    dense = densify(cache.group_track(windows, tracks), windows)

    # Assert
    for index, (chrom, start, end) in enumerate(
        zip(windows["chrom"], windows["sig_start"], windows["sig_end"])
    ):
        expected = store.window(chrom, start, end).numpy()
        assert numpy.allclose(dense[index], expected, rtol=1e-5, atol=1e-5), index


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

    Checked above the scans rather than inside one way of filling the
    layout, because the limit is the layout's: a check that lived in
    one branch would leave every other branch to wrap the index.
    """

    # Arrange
    too_many = ["unread"] * (numpy.iinfo(numpy.int16).max + 1)

    # Act & assert
    with pytest.raises(
        ValueError, match=r"more channels than .* 32768 against a limit of 32767"
    ):
        cache.group_track(frame([("chr1", 8, 16)]), too_many)


def test_group_track_should_allow_as_many_channels_as_the_index_holds():
    """Test the channel count the layout can just about index.

    Given:
        Exactly as many channels as an ``int16`` channel index can
        distinguish.
    When:
        They are grouped.
    Then:
        It should get past the channel limit and fail on reading the
        first track, with `FileNotFoundError` naming the missing path,
        because the largest index the layout holds is a usable channel.

    Without this the guard's comparison is unpinned in one direction:
    it can be tightened to refuse the last usable channel and the
    one-past-the-limit test above cannot tell.
    """

    # Arrange
    at_the_limit = ["unread"] * numpy.iinfo(numpy.int16).max

    # Act & assert
    with pytest.raises(FileNotFoundError, match="unread"):
        cache.group_track(frame([("chr1", 8, 16)]), at_the_limit)


def test_group_track_should_refuse_a_windows_frame_it_cannot_persist(tracks):
    """Test the grouping's answer for a windows frame of another library.

    Given:
        A pandas frame of windows, carrying the same three columns.
    When:
        It is grouped.
    Then:
        It should be refused by type, because grouping it succeeds and
        persisting the result does not: the entry writer needs polars
        to write the window tables, and it answers a miss rather than
        raising, so the cache would silently never persist.

    The grouping is what makes this worth refusing rather than
    leaving. A pandas frame indexes and iterates enough like a polars
    one that the arrays come out identical, so nothing downstream of
    the grouping can tell the two apart -- and the failure surfaces
    once per run, as a cache that is rebuilt every time.
    """

    # Arrange
    pandas = pytest.importorskip("pandas")
    windows = pandas.DataFrame({"chrom": ["chr1"], "sig_start": [8], "sig_end": [16]})

    # Act & assert
    with pytest.raises(TypeError, match="polars DataFrame"):
        cache.group_track(windows, tracks)


def test_group_track_should_refuse_a_track_it_cannot_verify_complete(tmp_path):
    """Test the grouping's answer for a track that does not verify.

    Given:
        A track directory whose manifest has been removed, so the
        package cannot verify it complete.
    When:
        A window set is grouped over it.
    Then:
        It should raise, because a track missing a contig outright
        would otherwise group to a run of zeros for every window on
        that contig and say nothing.

    The refusal arrives transitively, from the reader this dispatches
    through rather than from here, so a change that moves the grouping
    onto a reader which does not verify would delete the guarantee
    with every other test still green.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    root = str(tmp_path / "partial.track.parquet")
    builders.write_track(root, [("chr1", 10, 1.0)])
    os.remove(quadrat.manifest_path(root))

    # Act & assert
    with pytest.raises(ValueError, match="cannot be verified complete"):
        cache.group_track(frame([("chr1", 8, 16)]), [root])


def test_group_track_should_release_the_tables_it_registers(tracks):
    """Test that the grouping leaves nothing bound in the session.

    Given:
        A grouping over two contigs, which registers a window frame and
        a track frame for each.
    When:
        It finishes.
    Then:
        The tables bound under the package's ``quadrat_`` prefix
        should be exactly those bound before it ran, because the
        session is process-global and the next caller of a name would
        read this one's rows rather than failing.

    Checked against the prefix rather than against the names the
    grouping uses, so the assertion holds whatever names those are and
    cannot pass by looking for the wrong ones. Compared with what was
    bound before rather than with nothing, so a table another test
    left bound is not this one's failure.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8)])
    before = quadrat_tables(pb)

    # Act
    cache.group_track(windows, tracks)

    # Assert
    assert quadrat_tables(pb) == before


def test_group_track_should_release_the_tables_it_registered_when_a_later_one_is_refused(
    mocker, tracks
):
    """Test that a registration refused partway through still unwinds.

    Given:
        An engine that binds the first frame a grouping registers and
        refuses the second.
    When:
        A window set is grouped.
    Then:
        It should propagate the refusal, and the tables bound under the
        package's ``quadrat_`` prefix should be exactly those bound
        before it ran, so the frame that did land is not left readable
        under a name the next query registers.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    windows = frame([("chr1", 8, 16)])
    before = quadrat_tables(pb)
    register = pb.from_polars
    calls = []

    def refuse_the_second(name, registered):
        calls.append(name)
        if len(calls) == 2:
            raise OSError("the provider refused this frame")
        return register(name, registered)

    mocker.patch.object(pb, "from_polars", side_effect=refuse_the_second)

    # Act
    with pytest.raises(OSError, match="refused this frame"):
        cache.group_track(windows, tracks)

    # Assert
    assert len(calls) == 2
    assert quadrat_tables(pb) == before


@pytest.mark.packaging
def test_group_track_should_leave_the_multiprocessing_start_method_unchosen():
    """Test whether a grouping settles a process-global choice.

    Given:
        A fresh interpreter that has not chosen a start method.
    When:
        A window set is grouped, which runs a query through the engine.
    Then:
        The start method should still be unchosen, so a program that
        calls ``set_start_method`` after its first pileup can still
        make the choice `quadrat.cache` derives a residency from.

    In a child interpreter because the start method is process-global
    and is settled once: an in-process test that bound it would decide
    the answer for every test that ran after it.

    Through a real query rather than through an import of the engine
    alone, because the bar is what binds the method and a bar is built
    per query rather than at import. ``tqdm`` builds its write lock in
    ``__new__``, before it consults ``disable``, and the default lock
    holds a `multiprocessing.RLock` -- so ``TQDM_DISABLE`` does not
    prevent this, and a `threading.RLock` installed ahead of it is
    what does.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    pytest.importorskip("giql")
    program = (
        "import multiprocessing, polars\n"
        "from quadrat import cache\n"
        "windows = polars.DataFrame("
        "{'chrom': ['chr1'], 'sig_start': [0], 'sig_end': [8]})\n"
        "rows = polars.DataFrame("
        "{'chrom': ['chr1'], 'pos': [1], 'value': [1.0]})\n"
        "cache.group_track(windows, [rows])\n"
        "print(multiprocessing.get_start_method(allow_none=True))\n"
    )

    # Act
    result = run_child([sys.executable, "-c", program], "the child interpreter failed")

    # Assert
    assert result.stdout.strip() == "None"


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


def test_window_group_sums_should_total_a_window_closed_at_its_start_and_open_at_its_end(
    edge_tracks,
):
    """Test the bounds of the span a total covers.

    Given:
        A track whose rows include one at exactly a window's start and
        one at exactly its end.
    When:
        The group sums are taken.
    Then:
        The total should carry the row at the start and the row inside
        it and not the row at the end, so a total covers the same
        half-open span a grouped read does.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])

    # Act
    totals = cache.window_group_sums(windows, edge_tracks, [1])

    # Assert
    assert totals.tolist() == [[14.0]]


def test_window_group_sums_should_answer_an_empty_total_for_no_windows(tracks):
    """Test a locus set that filtered down to nothing.

    Given:
        An empty window set and a grouping that covers the channels.
    When:
        The group sums are taken.
    Then:
        It should answer a table with no rows and one column per group,
        so an empty locus set is a shape a caller can go on to use.

    The refusal of a wrong grouping on an empty set is the other half;
    this is the return an empty set takes when the grouping is right.
    """

    # Act
    totals = cache.window_group_sums(frame([]), tracks, [2])

    # Assert
    assert tuple(totals.shape) == (0, 1)
    assert totals.dtype is torch.float32


def test_window_group_sums_should_pool_a_group_that_starts_past_channel_zero(
    three_channel,
):
    """Test a group whose channels do not begin at the first one.

    Given:
        Three channels pooled as one on its own and two together.
    When:
        The group sums are taken.
    Then:
        The first column should carry the first channel alone and the
        second the sum of the other two, so the channel offset a group
        starts at is read from the groups before it.

    A grouping of [2] or [1, 1] leaves this unexercised: both put every
    group's first channel where its index already is.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])

    # Act
    totals = cache.window_group_sums(windows, three_channel.tracks, [1, 2])

    # Assert
    assert totals.tolist() == [[4.0, 14.0]]


@pytest.mark.parametrize("shape", ["directory", "single_file", "eager", "lazy"])
def test_window_group_sums_should_agree_with_the_grouped_runs_it_totals(tmp_path, shape):
    """Test the two public readers of a track set against each other.

    Given:
        One window set, grouped into runs and totalled per group.
    When:
        Each window's run is summed by hand and compared with its
        total.
    Then:
        They should agree, because the sparse runs and the prefix-sum
        totals are two readings of one signal, and their agreement is a
        check neither gets from hand-written numbers alone.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    tracks = [
        shaped_track(shape, tmp_path / "t0", CHANNEL_0),
        shaped_track(shape, tmp_path / "t1", CHANNEL_1),
    ]
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8), ("chr1", 395, 405)])
    offsets, _, val, _, _ = cache.group_track(windows, tracks)

    # Act
    totals = cache.window_group_sums(windows, tracks, [2])

    # Assert
    for index in range(len(windows)):
        run = val[int(offsets[index]) : int(offsets[index + 1])]
        assert totals[index][0].item() == pytest.approx(float(run.sum()))


def test_window_group_sums_should_total_a_window_the_same_however_it_was_batched(
    tracks,
):
    """Test that a window's total does not depend on its neighbours.

    Given:
        One window, totalled alone and then as a row of a larger set
        spanning both contigs.
    When:
        The two totals for it are compared.
    Then:
        They should be identical, because the total is exact and a
        locus's answer cannot depend on how the windows were batched.
    """

    # Arrange
    alone = frame([("chr1", 8, 16)])
    batched = frame([("chr2", 0, 8), ("chr1", 8, 16), ("chr1", 395, 405)])

    # Act
    one = cache.window_group_sums(alone, tracks, [2])
    many = cache.window_group_sums(batched, tracks, [2])

    # Assert
    assert one[0].tolist() == many[1].tolist()


def test_window_group_sums_should_total_a_position_recorded_twice(duplicate_tracks):
    """Test what a duplicated position adds to a group total.

    Given:
        A track recording one position twice, with different values,
        beside one other row in the same window.
    When:
        The group sums are taken over that window.
    Then:
        The total should carry both values of the repeated position,
        because a track promises one value per position and nothing
        rejects one that breaks the promise -- so every reader of it
        has to answer the same sum, or the readers disagree silently.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])

    # Act
    totals = cache.window_group_sums(windows, duplicate_tracks, [1])

    # Assert
    assert totals.tolist() == [[8.0]]


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
    with pytest.raises(
        ValueError, match="channel_group_sizes sums to 1 and there are 2 tracks"
    ):
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

    A check placed below the empty-window return would accept on an
    empty locus set the same argument it refuses on a non-empty one.
    """

    # Act & assert
    with pytest.raises(ValueError, match="channel_group_sizes sums to 1"):
        cache.window_group_sums(frame([]), tracks, [1])


def test_window_group_sums_should_refuse_a_track_it_cannot_verify_complete(tmp_path):
    """Test the group totals' answer for a track that does not verify.

    Given:
        A track directory whose manifest has been removed, so the
        package cannot verify it complete.
    When:
        Its windows are totalled.
    Then:
        It should raise, for the reason the grouping raises: a total
        taken over a track that may be missing a contig is a plausible
        number with nothing wrong on its face.

    The two readers of a track have to accept the same thing, and a
    total of zero for a contig whose rows were never written is the
    failure that makes it matter.
    """

    # Arrange
    root = str(tmp_path / "partial.track.parquet")
    builders.write_track(root, [("chr1", 10, 1.0)])
    os.remove(quadrat.manifest_path(root))

    # Act & assert
    with pytest.raises(ValueError, match="cannot be verified complete"):
        cache.window_group_sums(frame([("chr1", 8, 16)]), [root], [1])


def test_array_meta_should_count_windows_from_the_offsets_and_sites_from_the_rows():
    """Test the three counts read off grouped arrays.

    Given:
        Grouped arrays over three windows holding five rows across two
        channels, so no two counts coincide.
    When:
        Their counts are taken.
    Then:
        It should report two channels, three windows -- one fewer than
        the offsets -- and five sites, so a reader can size a draw
        without opening an array.
    """

    # Arrange
    arrays = (
        numpy.asarray([0, 2, 2, 5], dtype=numpy.int64),
        numpy.asarray([0, 1, 0, 1, 2], dtype=numpy.int32),
        numpy.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=numpy.float32),
        numpy.asarray([0, 1, 0, 0, 1], dtype=numpy.int16),
        2,
    )

    # Act
    meta = cache.array_meta(arrays)

    # Assert
    assert meta == {"n_channels": 2, "n_windows": 3, "n_sites": 5}


def test_array_meta_should_count_no_windows_for_an_empty_window_set():
    """Test the counts of an empty grouping.

    Given:
        The arrays an empty window set groups to: one zero offset and
        no rows, over two channels.
    When:
        Their counts are taken.
    Then:
        It should report no windows and no sites and keep the channel
        count, so an entry for a locus set that filtered down to
        nothing records a shape rather than a negative count.
    """

    # Arrange
    arrays = (
        numpy.zeros(1, dtype=numpy.int64),
        numpy.zeros(0, dtype=numpy.int32),
        numpy.zeros(0, dtype=numpy.float32),
        numpy.zeros(0, dtype=numpy.int16),
        2,
    )

    # Act
    meta = cache.array_meta(arrays)

    # Assert
    assert meta == {"n_channels": 2, "n_windows": 0, "n_sites": 0}


def test_write_cache_arrays_should_write_the_four_the_layout_names(tmp_path, tracks):
    """Test the files a grouped result is written as.

    Given:
        A grouped window set and an empty directory.
    When:
        Its arrays are written there.
    Then:
        The four names the layout is read back by should be there as
        ``.npy`` files and nothing else, because a reader and an
        entry's record both assume exactly that set.
    """

    # Arrange
    grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)
    root = str(tmp_path / "arrays")
    os.makedirs(root)

    # Act
    cache.write_cache_arrays(root, *grouped[:4])

    # Assert
    assert sorted(os.listdir(root)) == sorted(
        name + ".npy" for name in ("offsets", "pos", "val", "chan")
    )


class TestWindowSignalCache:
    """The grouped cache, read by draw."""

    @pytest.mark.parametrize(
        ("arrays", "message"),
        [
            (([0, 3], [0, 1, 2], [1.0, 2.0], [0, 0, 0], 2), "one length"),
            (([0, 3], [0, 1, 2], [1.0, 2.0, 3.0], [0, 0], 2), "one length"),
            (([1, 3], [0, 1, 2], [1.0, 2.0, 3.0], [0, 0, 0], 2), "offsets must"),
            (([0, 2], [0, 1, 2], [1.0, 2.0, 3.0], [0, 0, 0], 2), "offsets must"),
            (([0, 2, 1, 3], [0, 1, 2], [1.0, 2.0, 3.0], [0, 0, 0], 2), "offsets must"),
            (([0, 3], [0, 1, 2], [1.0, 2.0, 3.0], [-1, 1, 1], 2), "chan must"),
            (([0, 3], [0, 1, 2], [1.0, 2.0, 3.0], [0, 1, 2], 2), "chan must"),
        ],
        ids=[
            "short_val",
            "short_chan",
            "offsets_not_from_zero",
            "offsets_short_of_the_rows",
            "offsets_decreasing",
            "negative_channel",
            "channel_past_the_count",
        ],
    )
    def test_from_arrays_should_refuse_arrays_a_draw_cannot_read(self, arrays, message):
        """Test the refusal of hand-assembled arrays that are inconsistent.

        Given:
            Five grouped values that break one cross-member invariant:
            unequal lengths, offsets that do not run non-decreasing
            from zero to the row count, or a channel outside the count.
        When:
            A cache is made from them.
        Then:
            It should raise `ValueError` naming the invariant, rather
            than answering a draw with signal on the wrong row.
        """

        # Arrange
        offsets, pos, val, chan, n_channels = arrays
        grouped = (
            numpy.asarray(offsets, dtype=numpy.int64),
            numpy.asarray(pos, dtype=numpy.int32),
            numpy.asarray(val, dtype=numpy.float32),
            numpy.asarray(chan, dtype=numpy.int16),
            n_channels,
        )

        # Act & assert
        with pytest.raises(ValueError, match=message):
            cache.WindowSignalCache.from_arrays(grouped, residency="memory")

    def test_from_arrays_should_put_its_scratch_under_the_scratch_variable(
        self, monkeypatch, tmp_path, tracks
    ):
        """Test where a scratch cache goes when only the environment names a place.

        Given:
            ``QUADRAT_SCRATCH_DIR`` naming a directory, and no scratch
            directory passed.
        When:
            A mapped cache is made from grouped arrays.
        Then:
            Its arrays should sit directly beneath that directory, so a
            cluster job can move scratch off a RAM-backed temporary
            directory without changing a call.
        """

        # Arrange
        monkeypatch.setenv("QUADRAT_SCRATCH_DIR", str(tmp_path))
        grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)

        # Act
        made = cache.WindowSignalCache.from_arrays(grouped, residency="mapped")

        # Assert
        assert os.path.dirname(made.mapped_dir) == str(tmp_path)

    def test_from_arrays_should_ignore_a_scratch_variable_set_to_nothing(
        self, monkeypatch, tracks
    ):
        """Test a scratch variable exported with an empty value.

        Given:
            ``QUADRAT_SCRATCH_DIR`` set to the empty string, and no
            scratch directory passed.
        When:
            A mapped cache is made from grouped arrays.
        Then:
            Its arrays should sit in the platform's temporary directory,
            because an empty variable names no directory.
        """

        # Arrange
        monkeypatch.setenv("QUADRAT_SCRATCH_DIR", "")
        grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)

        # Act
        made = cache.WindowSignalCache.from_arrays(grouped, residency="mapped")

        # Assert
        assert os.path.dirname(made.mapped_dir) == tempfile.gettempdir()

    def test_from_arrays_should_leave_its_scratch_when_a_forked_child_exits(self):
        """Test that a forked child exiting does not delete the arrays.

        Given:
            A cache that built its own scratch directory, in a process
            that then forks.
        When:
            The forked child drops its inherited copy and exits
            normally, running the handlers it inherited.
        Then:
            The directory should still be there, because the reap is
            scoped to the process that made it rather than to any
            process that inherited the registration.

        Through a real `os.fork` rather than a patched process id,
        because the mechanism under test is `atexit` inheritance: a
        child inherits the registration outright, and a patched pid in
        the parent exercises nothing, since nothing reaps on drop. A
        `multiprocessing` worker would not reach this -- it exits
        through ``os._exit`` and runs no handlers -- but that is a fact
        about one library's workers, not a guarantee about forked
        children.

        The arrays are assembled by hand rather than grouped. Grouping
        starts the engine's thread pools, and a forked child holds
        none of those threads, so its normal exit can wait forever on
        one of them. That is a hazard of forking after the engine has
        run and is not what this test is about.
        """

        # Arrange
        program = (
            "import os, sys, gc, numpy\n"
            "from quadrat import cache\n"
            "arrays = ("
            "numpy.asarray([0, 1], dtype=numpy.int64), "
            "numpy.asarray([2], dtype=numpy.int32), "
            "numpy.asarray([1.0], dtype=numpy.float32), "
            "numpy.asarray([0], dtype=numpy.int16), 1)\n"
            "built = cache.WindowSignalCache.from_arrays("
            "arrays, residency='mapped')\n"
            "root = built.mapped_dir\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    del built\n"
            "    gc.collect()\n"
            "    sys.exit(0)\n"
            "os.waitpid(pid, 0)\n"
            "print(os.path.isdir(root))\n"
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "True"

    def test_build_should_group_over_the_bounds_it_was_given(self, tracks):
        """Test that the bounds reach the grouping a cache is built from.

        Given:
            A window frame carrying a wide input span and a narrow
            output span under different column names.
        When:
            Caches are built over each pair.
        Then:
            The wider pair should hold rows the narrower one excludes,
            because a control is grouped over the input window while
            the signal is grouped over the output window.

        The grouping's own bounds test calls it directly and never
        goes through here, so without this an argument `build` drops on
        the way to the grouping goes unnoticed.
        """

        # Arrange
        windows = polars.DataFrame(
            {
                "chrom": ["chr1"],
                "in_start": [8],
                "in_end": [16],
                "out_start": [13],
                "out_end": [16],
            },
            schema={
                "chrom": polars.Utf8,
                "in_start": polars.Int64,
                "in_end": polars.Int64,
                "out_start": polars.Int64,
                "out_end": polars.Int64,
            },
        )

        # Act
        wide = cache.WindowSignalCache.build(
            windows, tracks, bound_columns=("in_start", "in_end"), residency="memory"
        )
        narrow = cache.WindowSignalCache.build(
            windows, tracks, bound_columns=("out_start", "out_end"), residency="memory"
        )

        # Assert
        assert wide.meta["n_sites"] > narrow.meta["n_sites"]
        assert narrow.meta["n_sites"] == 0

    def test_build_should_put_its_scratch_where_it_was_told(self, tracks, tmp_path):
        """Test the directory a scratch cache writes its arrays into.

        Given:
            A directory of the caller's choosing.
        When:
            A scratch cache is built with it.
        Then:
            The arrays should have been written beneath that directory,
            so a caller can put scratch on a volume that is fast or
            large enough for it rather than wherever the platform puts
            temporary files.
        """

        # Arrange
        chosen = tmp_path / "scratch"
        chosen.mkdir()

        # Act
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]),
            tracks,
            residency="mapped",
            scratch_dir=str(chosen),
        )
        root = built.mapped_dir

        # Assert
        assert os.path.dirname(root) == str(chosen)

    def test_build_should_put_its_scratch_under_the_scratch_variable(
        self, monkeypatch, tmp_path, tracks
    ):
        """Test that a built cache reaches the scratch variable too.

        Given:
            ``QUADRAT_SCRATCH_DIR`` naming a directory, and no scratch
            directory passed.
        When:
            A mapped cache is built.
        Then:
            Its arrays should sit directly beneath that directory, as
            they do for a cache made from arrays, because `build` is
            the call most callers make.
        """

        # Arrange
        monkeypatch.setenv("QUADRAT_SCRATCH_DIR", str(tmp_path))

        # Act
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="mapped"
        )

        # Assert
        assert os.path.dirname(built.mapped_dir) == str(tmp_path)

    def test_build_should_keep_the_arrays_in_memory_under_fork(self, mocker, tracks):
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
        assert built.mapped_dir is None

    @pytest.mark.parametrize("method", ["spawn", "forkserver"])
    def test_build_should_map_the_arrays_under_a_start_method_other_than_fork(
        self, mocker, tracks, method
    ):
        """Test the residency chosen when workers are not forked from the parent.

        Given:
            A start method of ``spawn`` or ``forkserver`` and no
            residency asked for.
        When:
            A cache is built.
        Then:
            It should write the arrays for mapping, because a worker
            started either way inherits none of the parent's arrays and
            would otherwise be sent one pickled copy each.
        """

        # Arrange
        mocker.patch.object(multiprocessing, "get_start_method", return_value=method)

        # Act
        built = cache.WindowSignalCache.build(frame([("chr1", 8, 16)]), tracks)

        # Assert
        assert built.mapped_dir is not None

    @pytest.mark.parametrize(
        ("platform_methods", "mapped"),
        [
            (["fork", "spawn", "forkserver"], False),
            (["spawn", "fork", "forkserver"], True),
            (["forkserver", "fork", "spawn"], True),
        ],
        ids=["fork_first", "spawn_first", "forkserver_first"],
    )
    def test_build_should_follow_the_platform_default_when_the_start_method_is_unchosen(
        self, mocker, tracks, platform_methods, mapped
    ):
        """Test the residency chosen before the program picks a start method.

        Given:
            No start method chosen yet, on a platform whose default is
            ``fork`` (Linux before 3.14), ``spawn`` (macOS) or
            ``forkserver`` (Linux from 3.14).
        When:
            A cache is built with no residency asked for.
        Then:
            It should hold the arrays in memory under a ``fork``
            default and write them for mapping under either other,
            because the platform default is how the program's workers
            will in fact start, and reading it binds nothing.
        """

        # Arrange
        mocker.patch.object(multiprocessing, "get_start_method", return_value=None)
        mocker.patch.object(
            multiprocessing, "get_all_start_methods", return_value=platform_methods
        )

        # Act
        built = cache.WindowSignalCache.build(frame([("chr1", 8, 16)]), tracks)

        # Assert
        assert (built.mapped_dir is not None) is mapped

    def test_build_should_refuse_an_unknown_residency(self, tracks):
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
        with pytest.raises(
            ValueError, match=r"residency must be one of 'memory', 'mapped' or None"
        ):
            cache.WindowSignalCache.build(
                frame([("chr1", 8, 16)]), tracks, residency="mmap"
            )

    @pytest.mark.packaging
    def test_build_should_keep_its_scratch_until_the_process_ends(self):
        """Test how long the directory a mapped cache made lasts.

        Given:
            A mapped cache in a fresh interpreter, which made its own
            scratch directory.
        When:
            The last reference to it is dropped, and then the process
            exits.
        Then:
            The directory should survive the drop and be gone after the
            exit, so a caller that pickles a cache into workers has
            nothing to keep alive -- and a run that builds a hundred
            still does not leave them behind.

        In a child interpreter because the second half only happens at
        interpreter exit: the reap is an `atexit` hook, which is what
        takes the directory's lifetime off the value and so off the
        parent's bookkeeping.
        """

        # Arrange
        program = (
            "import gc, os, polars\n"
            "from quadrat import cache\n"
            "windows = polars.DataFrame("
            "{'chrom': ['chr1'], 'sig_start': [8], 'sig_end': [16]})\n"
            "rows = polars.DataFrame("
            "{'chrom': ['chr1'], 'pos': [10], 'value': [1.0]})\n"
            "built = cache.WindowSignalCache.build("
            "windows, [rows], residency='mapped')\n"
            "root = built.mapped_dir\n"
            "del built\n"
            "gc.collect()\n"
            "print(os.path.isdir(root))\n"
            "print(root)\n"
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )
        survived, root = result.stdout.strip().splitlines()

        # Assert
        assert survived == "True"
        assert not os.path.exists(root)

    def test_build_should_leave_its_scratch_when_a_workers_copy_is_dropped(self, tracks):
        """Test what a worker's copy of a cache that owns its scratch deletes.

        Given:
            A cache that built its own scratch directory, pickled and
            restored as a worker's copy would be.
        When:
            The restored copy is dropped and collected.
        Then:
            The directory should still be there and the original should
            still read the same draw, because the copy owns nothing --
            every loader worker holds one of these, and the first to
            exit would otherwise delete the arrays the rest are
            mapping.

        A weak assertion on its own, because nothing deletes a
        scratch directory on drop in any process -- the reap is an
        `atexit` hook, so a copy that goes away is a non-event by
        construction rather than by a guard. What it still pins is the
        second half: the original keeps reading through the same files
        after a copy of it has been collected. The lifetime itself is
        pinned by the child-interpreter test that keeps the scratch
        until the process ends, and the pid that scopes it by the
        forked-child test.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="mapped"
        )
        root = built.mapped_dir
        expected = built.window(0, 0, 8).numpy()
        restored = pickle.loads(pickle.dumps(built))

        # Act
        del restored
        gc.collect()

        # Assert
        assert os.path.isdir(root)
        assert numpy.array_equal(built.window(0, 0, 8).numpy(), expected)

    @pytest.mark.packaging
    def test_build_should_leave_no_scratch_behind_when_the_write_fails(
        self, mocker, tmp_path, tracks
    ):
        """Test what a failed array write leaves on disk.

        Given:
            A scratch write that fails, as it does on a full disk.
        When:
            A cache is built.
        Then:
            It should raise, and leave no directory behind at once
            rather than at exit, because a full disk is the worst moment to
            go on holding a partly written set of arrays until the
            process ends.
        """

        # Arrange
        mocker.patch.object(numpy, "save", side_effect=OSError("disk full"))

        # Act
        with pytest.raises(OSError, match="disk full"):
            cache.WindowSignalCache.build(
                frame([("chr1", 8, 16)]),
                tracks,
                residency="mapped",
                scratch_dir=str(tmp_path),
            )

        # Assert
        assert os.listdir(tmp_path) == []

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
        adopted = cache.WindowSignalCache.adopt(
            root, grouped[4], cache.array_meta(grouped)
        )
        assert adopted.window(0, 0, 8).numpy().sum() > 0

        # Act
        del adopted
        gc.collect()

        # Assert
        assert os.path.isdir(root)

    def test_adopt_should_leave_its_files_when_a_workers_copy_is_dropped(
        self, tmp_path, tracks
    ):
        """Test what a worker's copy of an adopted cache deletes.

        Given:
            An adopted cache, pickled and restored as a worker's copy
            would be.
        When:
            The restored copy is dropped and collected.
        Then:
            The arrays should still be there and the original should
            still read the same draw, because a worker that exits first
            must not delete what every other worker is mapping.
        """

        # Arrange
        grouped = cache.group_track(frame([("chr1", 8, 16)]), tracks)
        root = str(tmp_path / "shared")
        os.makedirs(root)
        cache.write_cache_arrays(root, *grouped[:4])
        adopted = cache.WindowSignalCache.adopt(
            root, grouped[4], cache.array_meta(grouped)
        )
        expected = adopted.window(0, 0, 8).numpy()
        restored = pickle.loads(pickle.dumps(adopted))

        # Act
        del restored
        gc.collect()

        # Assert
        assert os.path.isdir(root)
        assert numpy.array_equal(adopted.window(0, 0, 8).numpy(), expected)

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

    @pytest.mark.parametrize("residency", ["memory", "mapped"])
    def test_meta_should_report_what_the_arrays_hold(self, tracks, residency):
        """Test the counts a cache carries beside its arrays.

        Given:
            A cache over two windows holding a known number of rows.
        When:
            Its counts are read.
        Then:
            They should report the channels, the windows and the sites,
            at either residency, so a reader can size a draw without
            touching a mapped array.
        """

        # Arrange
        windows = frame([("chr1", 8, 16), ("chr2", 0, 8)])
        grouped = cache.group_track(windows, tracks)

        # Act
        built = cache.WindowSignalCache.from_arrays(grouped, residency=residency)

        # Assert
        assert built.meta == {
            "n_channels": 2,
            "n_windows": 2,
            "n_sites": int(grouped[0][-1]),
        }

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

    @pytest.mark.parametrize("residency", ["memory", "mapped"])
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

    @pytest.mark.parametrize("residency", ["memory", "mapped"])
    def test_window_should_match_the_store_at_every_jitter(
        self, corpus, tracks, residency
    ):
        """Test every draw the corpus's geometry admits against the reader.

        Given:
            A window around every recorded row, each spanning the
            corpus's output window plus its jitter bound on both sides,
            cached in each residency.
        When:
            Every window is drawn at every jitter from zero to twice the
            bound, at the output window's length.
        Then:
            Each draw should equal what `TrackStore` reads over the span
            that jitter selects, so the shift a sampler applies lands
            every row where a coordinate read puts it.
        """

        # Arrange
        length, bound = corpus.out_window, corpus.max_jitter
        span = length + 2 * bound
        rows = [
            (chrom, pos)
            for channel in (CHANNEL_0, CHANNEL_1)
            for chrom, pos, _ in channel
        ]
        starts = sorted({(chrom, max(0, pos - span // 2)) for chrom, pos in rows})
        windows = frame([(chrom, start, start + span) for chrom, start in starts])
        store = stores.TrackStore(tracks)
        built = cache.WindowSignalCache.build(windows, tracks, residency=residency)

        # Act
        draws = {
            (index, jitter): built.window(index, jitter, length).numpy()
            for index in range(len(starts))
            for jitter in range(2 * bound + 1)
        }

        # Assert
        for (index, jitter), drawn in draws.items():
            chrom, start = starts[index]
            expected = store.window(chrom, start + jitter, start + jitter + length)
            assert numpy.array_equal(drawn, expected.numpy()), (index, jitter)

    def test_window_should_total_a_position_recorded_twice(self, duplicate_tracks):
        """Test what a duplicated position reads as in a draw.

        Given:
            A cache over a track recording one position twice, with
            different values.
        When:
            The window covering it is drawn.
        Then:
            That column should hold the sum of the two, because a track
            promises one value per position and nothing rejects one
            that breaks the promise -- so every reader of it has to
            answer the same sum, or the readers disagree silently.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), duplicate_tracks, residency="memory"
        )

        # Act
        drawn = built.window(0, 0, 8).numpy()

        # Assert
        assert drawn[0][2] == 5.0

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

    def test_window_should_mirror_a_jittered_draw_about_the_length(self, tracks):
        """Test the jitter and the mirror applied to one draw together.

        Given:
            A cache over one window and a non-zero jitter.
        When:
            The draw is read mirrored at that jitter, and read plainly
            at that jitter and flipped afterwards.
        Then:
            The two should agree, because the mirror is taken after the
            shift rather than mixed into it.

        Each of the two is covered alone, and a mirror that absorbed
        the jitter a second time would pass every test that sets only
        one of them.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        mirrored = built.window(0, 2, 8, reverse_complement=True).numpy()
        flipped = numpy.flip(built.window(0, 2, 8).numpy(), axis=1)

        # Assert
        assert numpy.array_equal(mirrored, flipped)

    def test_window_should_mirror_about_the_length_it_was_asked_for(self, tracks):
        """Test a mirrored draw whose length is not the grouped span.

        Given:
            A cache grouped over an eight-base window.
        When:
            A mirrored draw is read at twice that length.
        Then:
            It should be mirrored about the length asked for, so a draw
            padded past the span it was grouped over still reads as the
            reverse of the plain draw.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        mirrored = built.window(0, 0, 16, reverse_complement=True).numpy()
        flipped = numpy.flip(built.window(0, 0, 16).numpy(), axis=1)

        # Assert
        assert numpy.array_equal(mirrored, flipped)

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

    @pytest.mark.parametrize(
        "perm",
        [[1, 1], [-1, 0], [0, 1, 2], [0], [2, 0]],
        ids=["repeated", "negative", "too_long", "too_short", "out_of_range"],
    )
    def test_window_should_refuse_a_perm_that_is_not_a_permutation(self, tracks, perm):
        """Test what a mirrored draw does with a permutation it cannot use.

        Given:
            A cache over two channels, and a channel mapping that is
            not a permutation of its channels.
        When:
            A mirrored draw is read with it.
        Then:
            It should refuse, because every one of these otherwise
            answers a draw: a repeated entry sums two channels onto one
            row and leaves another empty, a negative entry indexes from
            the end, a long one is ignored, and only an entry past the
            last channel raises anything at all -- from numpy, naming
            nothing.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act & assert
        with pytest.raises(ValueError, match="permutation"):
            built.window(0, 0, 8, reverse_complement=True, perm=perm)

    def test_window_should_leave_the_channels_alone_for_an_unmirrored_draw(self, tracks):
        """Test what a permutation does without a mirror.

        Given:
            A cache over two channels and a permutation swapping them.
        When:
            An unmirrored draw is read with it.
        Then:
            The channels should be left alone, because the permutation
            is what a mirror applies and not a channel reordering of
            its own.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )
        swap = numpy.array([1, 0])

        # Act
        plain = built.window(0, 0, 8).numpy()
        permuted = built.window(0, 0, 8, perm=swap).numpy()

        # Assert
        assert numpy.array_equal(plain, permuted)

    def test_window_should_answer_zeros_for_a_window_holding_nothing(self, tracks):
        """Test a window no channel recorded a position in.

        Given:
            A cache whose window set includes one window with no rows
            in any channel.
        When:
            That window is read.
        Then:
            It should be the full shape and all zero, because a window
            near a contig edge is routine and a read of it must be a
            shape rather than a refusal.
        """

        # Arrange
        windows = frame([("chr1", 8, 16), ("chr1", 500, 520)])
        built = cache.WindowSignalCache.build(windows, tracks, residency="memory")

        # Act
        empty = built.window(1, 0, 20).numpy()

        # Assert
        assert tuple(empty.shape) == (2, 20)
        assert empty.sum() == 0

    def test_window_should_refuse_an_index_past_the_last_window(self, tracks):
        """Test a draw from a window the cache was not built for.

        Given:
            A cache built for one window.
        When:
            A second window is read.
        Then:
            It should raise, because the index is the sampler's row
            number and one past the set is a caller error rather than
            an empty draw.

        The positive side only. A negative index is outside the
        contract, neither refused nor supported, so nothing here pins
        what one answers.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act & assert
        with pytest.raises(IndexError):
            built.window(1, 0, 8)

    def test_window_should_answer_a_draw_the_caller_may_write_to(self, tracks):
        """Test whether a draw aliases the arrays behind it.

        Given:
            A mapped cache and a draw read from it.
        When:
            The draw is written to in place and the same draw read
            again.
        Then:
            The second read should be unchanged, because a worker
            normalising a batch in place must not corrupt a mapping
            every other worker shares.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="mapped"
        )
        first = built.window(0, 0, 8)
        expected = first.numpy().copy()

        # Act
        first += 99.0

        # Assert
        assert numpy.array_equal(built.window(0, 0, 8).numpy(), expected)

    @settings(FILE_BACKED, max_examples=25)
    @given(spans=window_sets, residency=st.sampled_from(["memory", "mapped"]))
    def test_window_should_answer_the_same_after_a_pickle_round_trip(
        self, tracks, spans, residency
    ):
        """Test the crossing a spawned worker makes, over the domain.

        Given:
            Any window set over one contig, cached in either residency,
            with every window already read so a mapped cache's arrays
            are open.
        When:
            The cache is pickled, restored, and every window read again.
        Then:
            Each draw and the counts should be identical, because under
            spawn a worker holds only what the pickle carried, whichever
            residency holds the arrays.
        """

        # Arrange
        windows = frame([("chr1", min(a, b), max(a, b)) for a, b in spans])
        built = cache.WindowSignalCache.build(windows, tracks, residency=residency)
        lengths = [
            max(1, int(end) - int(start))
            for start, end in zip(windows["sig_start"], windows["sig_end"])
        ]
        expected = [
            built.window(index, 0, length).numpy()
            for index, length in enumerate(lengths)
        ]

        # Act
        restored = pickle.loads(pickle.dumps(built))

        # Assert
        assert (restored.n_channels, restored.meta) == (built.n_channels, built.meta)
        for index, length in enumerate(lengths):
            read = restored.window(index, 0, length).numpy()
            assert numpy.array_equal(read, expected[index]), index

    def test_window_should_map_the_arrays_afresh_after_a_pickle_round_trip(
        self, tmp_path, tracks
    ):
        """Test that a restored mapped cache opens its files rather than carrying them.

        Given:
            A mapped cache that has already read a window, whose array
            files are then replaced on disk by another grouping's.
        When:
            It is pickled, restored, and the window read from the copy.
        Then:
            The copy should read the replacement files, because a
            worker's copy has to map the arrays in its own process
            rather than carry the parent's mapping across.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="mapped"
        )
        built.window(0, 0, 8)
        other = cache.group_track(frame([("chr1", 10, 18)]), tracks)
        expected = cache.WindowSignalCache.from_arrays(other, residency="memory")
        staged = str(tmp_path / "other")
        os.makedirs(staged)
        cache.write_cache_arrays(staged, *other[:4])
        for name in os.listdir(staged):
            os.replace(os.path.join(staged, name), os.path.join(built.mapped_dir, name))

        # Act
        restored = pickle.loads(pickle.dumps(built))

        # Assert
        assert numpy.array_equal(
            restored.window(0, 0, 8).numpy(), expected.window(0, 0, 8).numpy()
        )

    def test_window_should_map_the_arrays_afresh_when_the_pid_moves(
        self, mocker, tmp_path, tracks
    ):
        """Test what a mapped cache reads in a process other than the one that opened it.

        Given:
            A mapped cache that has already read a window, whose array
            files are then replaced on disk by another grouping's.
        When:
            The window is read again under another process id, as a
            forked worker reads its inherited copy.
        Then:
            It should read the replacement files, because a mapping
            opened by the parent is not this process's to use.
        """

        # Arrange
        built = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="mapped"
        )
        built.window(0, 0, 8)
        other = cache.group_track(frame([("chr1", 10, 18)]), tracks)
        expected = cache.WindowSignalCache.from_arrays(other, residency="memory")
        staged = str(tmp_path / "other")
        os.makedirs(staged)
        cache.write_cache_arrays(staged, *other[:4])
        for name in os.listdir(staged):
            os.replace(os.path.join(staged, name), os.path.join(built.mapped_dir, name))
        mocker.patch.object(os, "getpid", return_value=os.getpid() + 1)

        # Act
        read = built.window(0, 0, 8).numpy()

        # Assert
        assert numpy.array_equal(read, expected.window(0, 0, 8).numpy())

    @settings(FILE_BACKED, max_examples=25)
    @given(
        jitter=st.integers(min_value=-8, max_value=8),
        length=st.integers(min_value=1, max_value=16),
        mirrored=st.booleans(),
    )
    def test_window_should_mirror_any_draw_it_can_answer_plainly(
        self, tracks, jitter, length, mirrored
    ):
        """Test the draw arithmetic as one invariant over its three inputs.

        Given:
            Any jitter within the window, any length up to twice the span,
            and either mirror setting, over one cache held in memory.
        When:
            The draw is read.
        Then:
            A mirrored draw should equal the plain draw flipped, and no
            draw should carry more non-zero entries than the window
            holds rows, so the three inputs compose rather than
            interacting.

        The example-based tests cover each input alone and the jitter
        and mirror together; this covers the domain, which is where an
        off-by-one in the mirror at an odd length would live.
        """

        # Arrange
        held = cache.WindowSignalCache.build(
            frame([("chr1", 8, 16)]), tracks, residency="memory"
        )

        # Act
        drawn = held.window(0, jitter, length, reverse_complement=mirrored).numpy()
        plain = held.window(0, jitter, length).numpy()

        # Assert
        assert tuple(drawn.shape) == (2, length)
        if mirrored:
            assert numpy.array_equal(drawn, numpy.flip(plain, axis=1))
        assert numpy.count_nonzero(drawn) <= int(held.meta["n_sites"])

    @settings(FILE_BACKED, max_examples=25)
    @given(
        jitter=st.integers(min_value=-8, max_value=8),
        length=st.integers(min_value=1, max_value=16),
        mirrored=st.booleans(),
    )
    def test_window_should_answer_the_same_draw_in_either_residency(
        self, tracks, jitter, length, mirrored
    ):
        """Test that the residency holding the arrays changes no draw.

        Given:
            Any jitter within the window, any length up to twice the span,
            and either mirror setting, over one cache held in memory and
            one mapped from the same grouping.
        When:
            The draw is read from each.
        Then:
            The two should be identical, so which residency holds the
            arrays changes nothing a batch carries.
        """

        # Arrange
        windows = frame([("chr1", 8, 16)])
        held = cache.WindowSignalCache.build(windows, tracks, residency="memory")
        mapped = cache.WindowSignalCache.build(windows, tracks, residency="mapped")

        # Act
        from_memory = held.window(0, jitter, length, reverse_complement=mirrored)
        from_mapping = mapped.window(0, jitter, length, reverse_complement=mirrored)

        # Assert
        assert numpy.array_equal(from_mapping.numpy(), from_memory.numpy())


def test_default_cache_dir_should_follow_the_environment_on_every_call(monkeypatch):
    """Test that the cache directory is read per call, not once.

    Given:
        A cache directory named in the environment, then renamed.
    When:
        The default is asked for after each.
    Then:
        It should follow, so a process that sets the variable after
        importing this package is not left pointed at the real user
        cache.

    The suite's own guard rests on this: it assigns the variable before
    anything imports the package, and a reader that resolved once at
    import would have already answered.
    """

    # Arrange
    monkeypatch.setenv("QUADRAT_CACHE_DIR", "/tmp/first")

    # Act
    first = cache.default_cache_dir()
    monkeypatch.setenv("QUADRAT_CACHE_DIR", "/tmp/second")
    second = cache.default_cache_dir()

    # Assert
    assert first == os.path.join("/tmp/first", "windows")
    assert second == os.path.join("/tmp/second", "windows")


def test_default_cache_dir_should_fall_back_to_the_xdg_convention(monkeypatch):
    """Test where entries go with no variable of our own set.

    Given:
        No ``QUADRAT_CACHE_DIR`` and an ``XDG_CACHE_HOME``.
    When:
        The default is asked for.
    Then:
        It should sit under the XDG directory, named for this package,
        so caching works with no configuration and relocating the
        whole user cache relocates this too.
    """

    # Arrange
    monkeypatch.delenv("QUADRAT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")

    # Act & assert
    assert cache.default_cache_dir() == os.path.join("/tmp/xdg", "quadrat", "windows")


def test_default_cache_dir_should_fall_back_to_a_home_relative_default(monkeypatch):
    """Test where entries go with nothing in the environment at all.

    Given:
        Neither the package's variable nor the XDG one set, and a home
        directory of its own.
    When:
        The default is asked for.
    Then:
        It should sit under that home directory, because caching has to
        need no configuration -- this is the tier a first run lands in.
    """

    # Arrange
    monkeypatch.delenv("QUADRAT_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/someone")

    # Act
    resolved = cache.default_cache_dir()

    # Assert
    assert resolved == os.path.join("/home/someone", ".cache", "quadrat", "windows")


@pytest.mark.parametrize("variable", ["QUADRAT_CACHE_DIR", "XDG_CACHE_HOME"])
def test_default_cache_dir_should_expand_a_home_relative_variable(monkeypatch, variable):
    """Test the spelling a shell would hand either variable.

    Given:
        A tilde-relative path in one of the two variables.
    When:
        The default is asked for.
    Then:
        It should expand to the home directory, because a variable set
        by hand carries whatever a shell does not expand for it.
    """

    # Arrange
    monkeypatch.delenv("QUADRAT_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    monkeypatch.setenv(variable, "~/elsewhere")

    # Act
    resolved = cache.default_cache_dir()

    # Assert
    assert resolved.startswith("/home/someone/elsewhere")


def test_default_cache_dir_should_ignore_a_variable_set_to_nothing(monkeypatch):
    """Test a variable exported with an empty value.

    Given:
        The package's variable set to the empty string and an XDG
        directory set properly.
    When:
        The default is asked for.
    Then:
        It should fall through to the XDG directory, because an empty
        variable names no directory -- and an export with nothing after
        the equals sign is the ordinary way to get one.
    """

    # Arrange
    monkeypatch.setenv("QUADRAT_CACHE_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")

    # Act
    resolved = cache.default_cache_dir()

    # Assert
    assert resolved == os.path.join("/xdg", "quadrat", "windows")


def test_window_cache_key_should_answer_the_same_key_for_the_same_inputs(tmp_path):
    """Test that the key is a function of the inputs alone.

    Given:
        One input set.
    When:
        Its key is taken twice.
    Then:
        The two should agree, because a key that moved between calls
        would make every run a cold build.
    """

    # Arrange
    given = inputs(tmp_path)

    # Act & assert
    assert cache.window_cache_key(**given) == cache.window_cache_key(**given)


def test_window_cache_key_should_answer_the_same_key_for_every_path_shape(tmp_path):
    """Test that the reference's spelling does not move the key.

    Given:
        One reference named as a `str` and as `bytes`.
    When:
        A key is taken with each.
    Then:
        They should agree, so a caller holding either shape Python
        calls a path gets the same entry.

    An index name built with `str` would hash, for a `bytes`
    reference, the index of a path spelled from a repr -- which is
    absent, so two different genomes would collide on it.
    """

    # Arrange
    given = inputs(tmp_path)
    encoded = dict(given, sequences=os.fsencode(given["sequences"]))

    # Act & assert
    assert cache.window_cache_key(**given) == cache.window_cache_key(**encoded)


def test_window_cache_key_should_notice_a_reference_with_another_index(tmp_path):
    """Test the key's sensitivity to the reference, through its index.

    Given:
        One input set whose reference index is replaced with another
        genome's.
    When:
        Keys are taken before and after.
    Then:
        They should differ, because the index is what `chrom_lengths`
        reads and so what decides the window geometry a reference
        admits.

    Why the index stands for the reference is
    `quadrat.cache.window_cache_key`'s to say.
    """

    # Arrange
    given = inputs(tmp_path)
    before = cache.window_cache_key(**given)

    # Act
    other = builders.write_fasta(str(tmp_path / "other.fa"), {"chr1": 700}, index=True)
    shutil.copyfile(other + ".fai", given["sequences"] + ".fai")

    # Assert
    assert cache.window_cache_key(**given) != before


def test_window_cache_key_should_notice_a_reference_with_no_index(tmp_path):
    """Test two unindexed references at different paths.

    Given:
        Two references with no FASTA index beside either, and the same
        reference once an index is written for it.
    When:
        Keys are taken for each.
    Then:
        The two paths should key differently and indexing one should
        move its key, so an unindexed reference is still identified by
        where it is -- the index being what identifies it better.
    """

    # Arrange
    first = builders.write_fasta(str(tmp_path / "a.fa"), {"chr1": 600}, index=False)
    second = builders.write_fasta(str(tmp_path / "b.fa"), {"chr1": 600}, index=False)
    arguments = inputs(tmp_path)

    # Act
    one = cache.window_cache_key(**dict(arguments, sequences=first))
    other = cache.window_cache_key(**dict(arguments, sequences=second))
    builders.write_fasta(str(tmp_path / "a.fa"), {"chr1": 600}, index=True)
    indexed = cache.window_cache_key(**dict(arguments, sequences=first))

    # Assert
    assert one != other
    assert one != indexed


def test_window_cache_key_should_change_when_the_bound_columns_do(tmp_path):
    """Test that the window a cache is grouped over is part of its key.

    Given:
        One input set, keyed over the output window and then over the
        wider input window.
    When:
        The two keys are compared.
    Then:
        They should differ, because the two group different rows into
        every window and an entry built for one would otherwise be
        served to a caller asking for the other.

    The parameter decides what the arrays hold rather than how they
    are computed, which is the line the key is supposed to draw.
    """

    # Arrange
    arguments = inputs(tmp_path)

    # Act
    narrow = cache.window_cache_key(**arguments)
    wide = cache.window_cache_key(bound_columns=("in_start", "in_end"), **arguments)

    # Assert
    assert narrow != wide


def test_window_cache_key_should_answer_the_key_pinned_for_a_fixed_input_set():
    """Test the key against a digest recorded for one input set.

    Given:
        An input set that touches no file: no tracks, no locus files,
        and a reference at an absolute path where nothing exists, with
        every scalar argument set.
    When:
        Its key is taken.
    Then:
        It should equal the digest recorded here, so a change to the
        cache version, the domain tag or how any argument is hashed
        moves every key and has to be made here on purpose.

    The read side of a version change is the wrong-version miss below.
    """

    # Arrange
    arguments = unfiled_inputs(
        chroms=["chr1", "chr2"],
        in_window=2114,
        out_window=1000,
        max_jitter=500,
        summits=True,
        min_counts=10.0,
        max_counts=[100.0, 200.0],
        signal_groups=[2],
        control_groups=[1],
    )

    # Act
    key = cache.window_cache_key(**arguments)

    # Assert
    assert key == "3dde7a8265843321c17284c9ae513c3f7986a4c74c814b224ca8b9312ce4291f"


@pytest.mark.parametrize(
    "spelling",
    [
        lambda: 10,
        lambda: numpy.float32(10.0),
        lambda: numpy.int64(10),
        lambda: pytest.importorskip("torch").tensor(10.0),
    ],
    ids=["int", "float32", "int64", "tensor"],
)
def test_window_cache_key_should_key_a_threshold_by_value_not_spelling(
    tmp_path, spelling
):
    """Test that a threshold of any numeric type reaches the key.

    Given:
        One input set, keyed with a count threshold spelled as a plain
        float and again as another numeric type holding the same
        value, among them the ``float32`` `window_group_sums` yields.
    When:
        The two keys are compared.
    Then:
        They should be equal, because the key is sensitive to the
        threshold and not to the type it arrived in.
    """

    # Arrange
    arguments = inputs(tmp_path)

    # Act
    plain = cache.window_cache_key(min_counts=10.0, **arguments)
    spelled = cache.window_cache_key(min_counts=spelling(), **arguments)

    # Assert
    assert spelled == plain


@settings(max_examples=50)
@given(value=st.floats(width=32, allow_nan=False, allow_infinity=False))
def test_window_cache_key_should_key_any_threshold_by_value_not_spelling(value):
    """Test that a threshold's spelling never reaches the key.

    Given:
        Any finite value ``float32`` can hold, as a lower threshold
        spelled as a Python float, as a ``float32`` scalar and as a
        tensor.
    When:
        A key is taken with each spelling.
    Then:
        All three should be equal, because the key is sensitive to the
        threshold and not to the type `window_group_sums` or a caller
        hands it in.
    """

    # Arrange
    spellings = (
        float(numpy.float32(value)),
        numpy.float32(value),
        torch.tensor(value, dtype=torch.float32),
    )

    # Act
    keys = {
        cache.window_cache_key(**unfiled_inputs(min_counts=spelled))
        for spelled in spellings
    }

    # Assert
    assert len(keys) == 1


@settings(max_examples=50)
@given(
    values=st.lists(
        st.floats(width=32, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=4,
    )
)
def test_window_cache_key_should_key_a_per_group_threshold_by_value_not_spelling(
    values,
):
    """Test that a per-group threshold's container never reaches the key.

    Given:
        Any per-group upper threshold of ``float32``-representable
        values, as a list and as a ``float32`` array.
    When:
        A key is taken with each.
    Then:
        The two should be equal, so a threshold array taken from
        `window_group_sums` keys as the list of the same values does.
    """

    # Act
    listed = cache.window_cache_key(**unfiled_inputs(max_counts=values))
    arrayed = cache.window_cache_key(
        **unfiled_inputs(max_counts=numpy.asarray(values, dtype=numpy.float32))
    )

    # Assert
    assert listed == arrayed


def test_window_cache_key_should_take_a_threshold_per_channel_group(tmp_path):
    """Test that a per-group threshold array is hashed by its values.

    Given:
        One input set, keyed with an upper threshold given as a
        ``float32`` array of one value per channel group, and again
        with one of those values changed.
    When:
        The keys are taken.
    Then:
        Both should be produced rather than refused, and should differ.
    """

    # Arrange
    arguments = inputs(tmp_path)

    # Act
    one = cache.window_cache_key(
        max_counts=numpy.asarray([100.0, 200.0], dtype=numpy.float32), **arguments
    )
    other = cache.window_cache_key(
        max_counts=numpy.asarray([100.0, 201.0], dtype=numpy.float32), **arguments
    )

    # Assert
    assert one != other


def test_window_cache_key_should_refuse_a_reference_that_is_not_a_path(tmp_path):
    """Test the refusal of a reference the key cannot hash.

    Given:
        A reference that is no kind of path.
    When:
        A key is taken.
    Then:
        It should raise `TypeError` naming the parameter, rather than
        hashing a repr and answering a key that looks usable.
    """

    # Arrange
    given = dict(inputs(tmp_path), sequences=42)

    # Act & assert
    with pytest.raises(TypeError, match="window_cache_key hashes.*sequences must be"):
        cache.window_cache_key(**given)


@pytest.mark.parametrize(
    "override",
    [
        {"in_window": 2000},
        {"out_window": 900},
        {"max_jitter": 400},
        {"summits": True},
        {"chroms": ["chr1"]},
        {"signal_groups": [1]},
    ],
)
def test_window_cache_key_should_notice_anything_that_moves_the_windows(
    tmp_path, override
):
    """Test the key's sensitivity to the geometry.

    Given:
        One input set and one changed window argument.
    When:
        Keys are taken with and without it.
    Then:
        They should differ, because every one of these changes which
        positions an entry holds and a shared key would serve the
        wrong ones.
    """

    # Arrange
    given = inputs(tmp_path)

    # Act & assert
    assert cache.window_cache_key(**given) != cache.window_cache_key(
        **dict(given, **override)
    )


@pytest.mark.parametrize(
    "override",
    [
        {"exclusion_lists": None},
        {"controls": None},
        {"control_groups": None},
        {"min_counts": None},
        {"max_counts": None},
        {"signal_groups": None},
    ],
    ids=lambda value: next(iter(value)),
)
def test_window_cache_key_should_notice_every_argument_that_filters_a_window(
    tmp_path, override
):
    """Test the arguments that decide which windows survive.

    Given:
        One input set, keyed with each of these arguments unset and
        then set.
    When:
        The two keys are compared.
    Then:
        They should differ, because each of these changes which windows
        an entry holds or how its channels are pooled -- and a stale
        hit would train on the wrong data and say nothing.
    """

    # Arrange
    arguments = inputs(tmp_path)
    name = next(iter(override))
    set_to = {
        "exclusion_lists": [
            builders.write_bed(str(tmp_path / "x.bed"), [("chr1", 0, 4)])
        ],
        "controls": [str(tmp_path / "c0.track.parquet")],
        "control_groups": [1],
        "min_counts": 10,
        "max_counts": 1000,
        "signal_groups": [1],
    }[name]

    # Act
    unset = cache.window_cache_key(**arguments)
    given = cache.window_cache_key(**dict(arguments, **{name: set_to}))

    # Assert
    assert unset != given


@settings(max_examples=50)
@given(pair=st.lists(thresholds, min_size=2, max_size=2, unique=True))
def test_window_cache_key_should_notice_a_changed_lower_threshold(pair):
    """Test the key's sensitivity to the lower threshold's value.

    Given:
        Two different finite lower thresholds.
    When:
        A key is taken with each.
    Then:
        They should differ, because a threshold decides which windows
        an entry holds, and a key that noticed only that one is set
        would serve one threshold's entry to the other.
    """

    # Arrange
    first, second = pair

    # Act
    one = cache.window_cache_key(**unfiled_inputs(min_counts=first))
    other = cache.window_cache_key(**unfiled_inputs(min_counts=second))

    # Assert
    assert one != other


@settings(max_examples=50)
@given(
    pair=st.lists(
        st.lists(thresholds, min_size=1, max_size=4),
        min_size=2,
        max_size=2,
        unique_by=tuple,
    )
)
def test_window_cache_key_should_notice_a_changed_upper_threshold(pair):
    """Test the key's sensitivity to the upper threshold's values.

    Given:
        Two different per-group upper thresholds.
    When:
        A key is taken with each.
    Then:
        They should differ, for the reason a changed lower threshold
        does.
    """

    # Arrange
    first, second = pair

    # Act
    one = cache.window_cache_key(**unfiled_inputs(max_counts=first))
    other = cache.window_cache_key(**unfiled_inputs(max_counts=second))

    # Assert
    assert one != other


@settings(max_examples=50)
@given(pair=st.lists(contig_subsets, min_size=2, max_size=2, unique_by=tuple))
def test_window_cache_key_should_notice_a_changed_contig_subset(pair):
    """Test the key's sensitivity to which contigs are kept.

    Given:
        Two different contig subsets, either of which may be empty.
    When:
        A key is taken with each.
    Then:
        They should differ, because the subset decides which windows
        exist, and a key that noticed only that one is given would
        serve one subset's entry to the other.
    """

    # Arrange
    first, second = pair

    # Act
    one = cache.window_cache_key(**unfiled_inputs(chroms=first))
    other = cache.window_cache_key(**unfiled_inputs(chroms=second))

    # Assert
    assert one != other


@settings(max_examples=50)
@given(pair=st.lists(group_sizes, min_size=2, max_size=2, unique_by=tuple))
@pytest.mark.parametrize("argument", ["signal_groups", "control_groups"])
def test_window_cache_key_should_notice_a_changed_channel_grouping(argument, pair):
    """Test the key's sensitivity to how a channel set is grouped.

    Given:
        Two different non-empty groupings, for the signal channels or
        for the control channels.
    When:
        A key is taken with each.
    Then:
        They should differ, because the grouping decides how channels
        are pooled for a threshold, and a key that noticed only that a
        grouping is given would serve one grouping's entry to the
        other.
    """

    # Arrange
    first, second = pair

    # Act
    one = cache.window_cache_key(**unfiled_inputs(**{argument: first}))
    other = cache.window_cache_key(**unfiled_inputs(**{argument: second}))

    # Assert
    assert one != other


def test_window_cache_key_should_notice_a_rewritten_track(tmp_path):
    """Test the key's sensitivity to a track's contents.

    Given:
        An input set whose track is rewritten with different rows.
    When:
        Keys are taken before and after.
    Then:
        They should differ, so a re-piled corpus is not read through an
        entry grouped from the previous one.

    A track is hashed by its record and a stat of every file rather
    than by its bytes, which is why this is worth pinning: the record
    carries the row counts and the pileup arguments, so a regenerated
    track almost always moves it.
    """

    # Arrange
    given = inputs(tmp_path)
    before = cache.window_cache_key(**given)

    # Act
    builders.write_track(given["signals"][0], [("chr1", 99, 1.0)])

    # Assert
    assert cache.window_cache_key(**given) != before


def test_window_cache_key_should_notice_a_rewritten_locus_file(tmp_path):
    """Test the key's sensitivity to a locus file's contents.

    Given:
        An input set whose peak file is rewritten in place.
    When:
        Keys are taken before and after.
    Then:
        They should differ, because the loci decide which windows exist
        at all and the path alone would not have moved.
    """

    # Arrange
    given = inputs(tmp_path)
    before = cache.window_cache_key(**given)

    # Act
    builders.write_bed(given["peaks"], [("chr2", 20, 28)])

    # Assert
    assert cache.window_cache_key(**given) != before


@pytest.mark.parametrize("rewritten", ["negatives", "exclusion_lists", "controls"])
def test_window_cache_key_should_notice_another_input_rewritten_in_place(
    tmp_path, rewritten
):
    """Test the key's sensitivity to the contents of every other file input.

    Given:
        An input set carrying a background locus file, an exclusion
        list and a control track, one of which is rewritten in place
        with different contents.
    When:
        Keys are taken before and after.
    Then:
        They should differ, because each decides which windows survive
        or what a control channel holds, and the path alone would not
        have moved -- a key taken over paths would serve the entry
        built from the previous contents.

    The background and exclusion files are rewritten at the same byte
    length, so the key has to have read what is in them rather than
    how big they are.
    """

    # Arrange
    given = inputs(tmp_path)
    given["exclusion_lists"] = [
        builders.write_bed(str(tmp_path / "exclude.bed"), [("chr1", 0, 4)])
    ]
    given["controls"] = [
        builders.write_track(str(tmp_path / "c0.track.parquet"), [("chr1", 20, 1.0)])
    ]
    before = cache.window_cache_key(**given)

    # Act
    if rewritten == "negatives":
        builders.write_bed(given["negatives"], [("chr1", 200, 208)])
    elif rewritten == "exclusion_lists":
        builders.write_bed(given["exclusion_lists"][0], [("chr1", 5, 9)])
    else:
        builders.write_track(given["controls"][0], [("chr1", 21, 2.0), ("chr2", 3, 1.0)])

    # Assert
    assert cache.window_cache_key(**given) != before


def test_window_cache_key_should_notice_the_channel_order(tmp_path):
    """Test that reordering the channels changes the key.

    Given:
        Two tracks, in one order and then the other.
    When:
        Keys are taken for each.
    Then:
        They should differ, because the order is the channel axis and
        an entry grouped in one order describes a different signal read
        in the other.
    """

    # Arrange
    given = inputs(tmp_path)
    second = builders.write_track(str(tmp_path / "s1.track.parquet"), CHANNEL_1)
    forward = dict(given, signals=[given["signals"][0], second])
    reversed_order = dict(given, signals=[second, given["signals"][0]])

    # Act & assert
    assert cache.window_cache_key(**forward) != cache.window_cache_key(**reversed_order)


def test_window_cache_key_should_take_no_argument_that_cannot_change_an_entry():
    """Test what the key deliberately does not depend on.

    Given:
        The key's own parameter list.
    When:
        It is inspected.
    Then:
        It should name none of the batch size, worker count, shuffle,
        seed, negative ratio or reverse-complement flag, because each
        changes what a run draws and none changes what the cache holds.

    Runs differing only in those share one entry, which is the point of
    hashing the inputs rather than the run.
    """

    # Arrange
    taken = set(inspect.signature(cache.window_cache_key).parameters)

    # Act & assert
    assert (
        taken
        & {
            "batch_size",
            "num_workers",
            "shuffle",
            "random_state",
            "negative_ratio",
            "reverse_complement",
            "pin_memory",
        }
        == set()
    )


def test_write_window_cache_should_commit_an_entry_the_package_verifies(
    tmp_path, tracks
):
    """Test that an entry is verifiable by the package's own verifier.

    Given:
        A grouped window set written as an entry.
    When:
        The package is asked to verify the entry complete.
    Then:
        It should pass, and the record should carry no row total, list
        every file it wrote with no count, and carry neither of the two
        fields nothing reads, because the verifier accepts exactly
        that shape and a field with no reader drifts from whatever it
        is meant to mirror.

    An absent total is what excuses the absent counts. The record
    itself is read here rather than inferred from the verifier
    accepting it, because `require_complete` also accepts an empty file
    list and a total of zero beside counted files, so a writer emitting
    either would still verify.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)

    # Act
    root = cache.write_window_cache(str(tmp_path), "deadbeef", windows, windows, grouped)

    # Assert
    assert root is not None
    assert quadrat.require_complete(root) == root
    with open(root + ".manifest.json") as handle:
        record = json.load(handle)
    assert "rows" not in record
    assert set(record["files"]) == {
        "peaks.parquet",
        "negatives.parquet",
    } | {
        os.path.join("signal", name + ".npy")
        for name in ("offsets", "pos", "val", "chan")
    }
    assert set(record["files"].values()) == {None}
    # A top-level n_windows would be a second derivation of one count
    # in one record, and from the peak and negative rows -- two here --
    # it disagrees with the one window the arrays hold. The spelling
    # kept is the one the arrays answer for.
    assert "n_windows" not in record
    assert "created" not in record
    assert record["signal_meta"]["n_windows"] == 1


def test_write_window_cache_should_create_a_cache_directory_that_is_not_there(
    tmp_path, tracks
):
    """Test a cache directory several levels below anything that exists.

    Given:
        A directory path whose parents do not exist.
    When:
        An entry is committed there and read back.
    Then:
        It should commit and read, because the directory resolves from
        an environment variable a caller may point anywhere and a first
        run must not have to create it by hand.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    deep = str(tmp_path / "a" / "b" / "c")

    # Act
    root = cache.write_window_cache(deep, "deep", windows, windows, grouped)

    # Assert
    assert root is not None
    assert cache.read_window_cache(deep, "deep") is not None


def test_write_window_cache_should_replace_an_entry_already_at_that_key(
    tmp_path, tracks
):
    """Test committing over an entry that is already there.

    Given:
        An entry committed at a key, and a second entry over a
        different window set at the same key.
    When:
        The second is committed and read back.
    Then:
        The read should answer the second, with no staging or discard
        sibling left beside it, so rebuilding in place is one step for
        every reader.
    """

    # Arrange
    first = frame([("chr1", 8, 16)])
    second = frame([("chr1", 8, 16), ("chr2", 0, 8)])
    cache.write_window_cache(
        str(tmp_path), "same", first, first, cache.group_track(first, tracks)
    )

    # Act
    cache.write_window_cache(
        str(tmp_path), "same", second, second, cache.group_track(second, tracks)
    )

    # Assert
    peaks, _, signal, _ = cache.read_window_cache(str(tmp_path), "same")
    assert len(peaks) == 2
    assert signal.meta["n_windows"] == 2
    assert sorted(os.listdir(str(tmp_path))) == ["same", "same.manifest.json"]


def test_write_window_cache_should_sweep_a_staging_directory_a_killed_write_left(
    tmp_path, tracks
):
    """Test what a new write does about an interrupted one's staging copy.

    Given:
        A staging sibling of an entry's key, as a write killed before
        its rename leaves.
    When:
        An entry is committed at that key.
    Then:
        It should commit and the sibling should be gone, because an
        entry is the size of its arrays and a cache directory
        accumulates one per failed key -- and the write that leaves
        them swallows its own failure, so nothing reports them.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    stale = str(tmp_path / "swept.staging.deadbeef")
    os.makedirs(stale)

    # Act
    root = cache.write_window_cache(str(tmp_path), "swept", windows, windows, grouped)

    # Assert
    assert root is not None
    assert not os.path.exists(stale)


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root is denied nothing by a mode, so the write would succeed",
)
def test_write_window_cache_should_answer_none_when_it_cannot_write(tmp_path, tracks):
    """Test what an unwritable cache directory does to a run.

    Given:
        A cache directory that cannot be written to.
    When:
        An entry is committed.
    Then:
        It should answer ``None`` rather than raise, because a cache
        that cannot be written is a slower run and must not take a
        training job down.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(str(locked), 0o500)

    # Act
    try:
        answer = cache.write_window_cache(
            str(locked / "under"), "key4", windows, windows, grouped
        )
    finally:
        os.chmod(str(locked), 0o700)

    # Assert
    assert answer is None


def test_write_window_cache_should_leave_no_staging_behind_when_it_fails(
    tmp_path, tracks
):
    """Test that a failed commit leaves the directory clean.

    Given:
        A commit that fails part way through.
    When:
        It returns.
    Then:
        The cache directory should hold no staging sibling, so a run
        that fails repeatedly does not fill the disk with half-written
        entries.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    broken = (grouped[0], grouped[1], grouped[2], grouped[3], object())

    # Act
    answer = cache.write_window_cache(str(tmp_path), "key5", windows, windows, broken)

    # Assert
    assert answer is None
    assert [name for name in os.listdir(str(tmp_path)) if ".staging." in name] == []


@pytest.mark.parametrize("residency", ["memory", "mapped"])
def test_read_window_cache_should_answer_what_was_written(tmp_path, tracks, residency):
    """Test the round trip, which is the whole contract.

    Given:
        An entry written from a grouped window set.
    When:
        It is read back in each residency.
    Then:
        Both window frames should match what was committed, the signal
        cache should be held in the residency asked for, and every
        window's draw should equal the cold build's, so a hit is
        indistinguishable from the build it replaces.
    """

    # Arrange
    windows = frame([("chr1", 8, 16), ("chr2", 0, 8), ("chr1", 396, 404)])
    grouped = cache.group_track(windows, tracks)
    built = cache.WindowSignalCache.from_arrays(grouped, residency="memory")
    cache.write_window_cache(str(tmp_path), "key0", windows, windows, grouped)

    # Act
    peaks, negatives, signal, control = cache.read_window_cache(
        str(tmp_path), "key0", residency=residency
    )

    # Assert
    assert peaks.to_dicts() == windows.to_dicts()
    assert negatives.to_dicts() == windows.to_dicts()
    assert control is None
    assert (signal.mapped_dir is None) is (residency == "memory")
    assert (signal.n_channels, signal.meta) == (built.n_channels, built.meta)
    for index in range(len(windows)):
        for mirrored in (False, True):
            read = signal.window(index, 0, 8, reverse_complement=mirrored)
            cold = built.window(index, 0, 8, reverse_complement=mirrored)
            assert numpy.array_equal(read.numpy(), cold.numpy()), (index, mirrored)


@settings(FILE_BACKED, max_examples=25)
@given(spans=window_sets, residency=st.sampled_from(["memory", "mapped"]))
def test_read_window_cache_should_answer_what_was_written_for_any_window_set(
    tmp_path_factory, tracks, spans, residency
):
    """Test the persisted round trip over arbitrary window sets.

    Given:
        Any number of windows over one contig, at any spans including
        empty ones, written as an entry and read back in either
        residency.
    When:
        Every window is drawn from the entry read back.
    Then:
        Each draw, and the counts the cache carries, should equal the
        cache made from the same grouped arrays without the disk in
        between, so a hit answers what the cold build would have.
    """

    # Arrange
    root = str(tmp_path_factory.mktemp("round-trip"))
    windows = frame([("chr1", min(a, b), max(a, b)) for a, b in spans])
    grouped = cache.group_track(windows, tracks)
    built = cache.WindowSignalCache.from_arrays(grouped, residency="memory")
    cache.write_window_cache(root, "any", windows, windows, grouped)

    # Act
    _, _, signal, _ = cache.read_window_cache(root, "any", residency=residency)

    # Assert
    assert signal.meta == built.meta
    for index, (start, end) in enumerate(zip(windows["sig_start"], windows["sig_end"])):
        length = max(1, int(end) - int(start))
        read = signal.window(index, 0, length).numpy()
        assert numpy.array_equal(read, built.window(index, 0, length).numpy()), index


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_read_window_cache_should_hold_a_hit_where_a_cold_build_would(
    mocker, tmp_path, tracks, method
):
    """Test the residency a hit takes when none is asked for.

    Given:
        A committed entry, and a start method of ``fork``, ``spawn`` or
        ``forkserver``.
    When:
        The entry is read with no residency asked for, and a cache is
        built cold from the same windows.
    Then:
        Both should be held the same way, in memory or mapped, so a hit
        is indistinguishable downstream from the build it replaces.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    cache.write_window_cache(
        str(tmp_path), "default", windows, windows, cache.group_track(windows, tracks)
    )
    mocker.patch.object(multiprocessing, "get_start_method", return_value=method)

    # Act
    _, _, signal, _ = cache.read_window_cache(str(tmp_path), "default")
    cold = cache.WindowSignalCache.build(windows, tracks)

    # Assert
    assert (signal.mapped_dir is None) == (cold.mapped_dir is None)


def test_read_window_cache_should_answer_the_controls_when_an_entry_carries_them(
    tmp_path, tracks
):
    """Test an entry written with a control set.

    Given:
        An entry carrying both signal and control arrays.
    When:
        It is read back.
    Then:
        Both caches should come back, each reporting its own channel
        count, because a control is grouped over the wider input window
        and is a separate cache rather than more channels of the first.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    signal_arrays = cache.group_track(windows, tracks)
    control_arrays = cache.group_track(windows, tracks[:1])
    cache.write_window_cache(
        str(tmp_path), "key1", windows, windows, signal_arrays, control_arrays
    )

    # Act
    _, _, signal, control = cache.read_window_cache(str(tmp_path), "key1")

    # Assert
    assert (signal.n_channels, control.n_channels) == (2, 1)


def test_read_window_cache_should_answer_the_counts_the_entry_recorded(tmp_path, tracks):
    """Test what an adopted cache knows about itself.

    Given:
        A committed entry.
    When:
        It is read and the signal cache's counts are taken.
    Then:
        They should be the counts the cache that wrote the entry
        carried, because a reader sizing a draw from a hit must not
        have to open an array to do so.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    cache.write_window_cache(str(tmp_path), "counts", windows, windows, grouped)

    # Act
    _, _, signal, _ = cache.read_window_cache(str(tmp_path), "counts")

    # Assert
    assert signal.meta == cache.array_meta(grouped)
    assert signal.meta["n_windows"] == 1


def test_read_window_cache_should_miss_a_directory_that_holds_no_entry(tmp_path):
    """Test the read of a key that was never written.

    Given:
        An empty cache directory.
    When:
        Any key is read.
    Then:
        It should answer ``None``, so a cold cache is a miss rather
        than an error the caller has to catch.
    """

    # Act & assert
    assert cache.read_window_cache(str(tmp_path), "absent") is None


@pytest.mark.parametrize("residency", [None, "memory", "mapped"])
@pytest.mark.parametrize(
    "damage",
    [
        "remove_array",
        "remove_manifest",
        "wrong_cache_version",
        "wrong_key",
        "no_channel_count",
        "unreadable_channel_count",
        "undeclared_controls",
        "garbage_array",
        "halved_array",
        "array_short_by_two_bytes",
        "array_long_by_two_bytes",
        "header_overwritten",
        "no_array_counts",
    ],
)
def test_read_window_cache_should_miss_an_entry_it_cannot_trust(
    tmp_path, tracks, damage, residency
):
    """Test every way an entry is refused rather than read.

    Given:
        A committed entry, damaged one way: an array file removed, the
        record removed, the record claiming another cache version or
        another key, the record omitting its channel count or carrying
        one that is not a count, the record declaring controls whose
        arrays were never written, an array rewritten as garbage or
        truncated, or the record carrying no array counts to check a
        size against.
    When:
        It is read.
    Then:
        It should answer ``None`` in every case, because a stale hit
        would train on the wrong data and say nothing, which is worse
        than rebuilding every time.

    The last seven are the shapes `require_complete` cannot see: it
    verifies the files a record lists and knows nothing about the
    fields an entry adds, nor about what is inside a file it finds, so
    each reaches this function as a record that has passed
    verification, and the read has to turn each into a miss rather
    than an exception. The four array shapes are the latest to surface
    if nothing opens an array before the first draw, which is why the
    read rather than a draw is what is asked here.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    root = cache.write_window_cache(str(tmp_path), "key2", windows, windows, grouped)
    marker = root + ".manifest.json"
    array = os.path.join(root, "signal", "pos.npy")
    if damage == "remove_array":
        os.remove(array)
    elif damage == "garbage_array":
        with open(array, "wb") as handle:
            handle.write(b"not an npy file at all")
    elif damage in ("halved_array", "array_short_by_two_bytes"):
        with open(array, "rb") as handle:
            whole = handle.read()
        keep = len(whole) // 2 if damage == "halved_array" else len(whole) - 2
        with open(array, "wb") as handle:
            handle.write(whole[:keep])
    elif damage == "array_long_by_two_bytes":
        # Longer, not shorter: numpy reads an over-long array without
        # complaint under either mmap mode, so the recorded counts are
        # the only thing that can refuse this one.
        with open(array, "ab") as handle:
            handle.write(b"\x00\x00")
    elif damage == "header_overwritten":
        # The one shape that keeps the file's size, so it reaches the
        # open rather than the size check.
        with open(array, "r+b") as handle:
            handle.write(b"\x93NUMPY" + b"\x00" * 122)
    elif damage == "remove_manifest":
        os.remove(marker)
    else:
        with open(marker) as handle:
            record = json.load(handle)
        if damage == "no_channel_count":
            del record["signal_channels"]
        elif damage == "unreadable_channel_count":
            record["signal_channels"] = "two"
        elif damage == "undeclared_controls":
            record["control_channels"] = 1
        elif damage == "no_array_counts":
            del record["signal_meta"]
        else:
            record["cache_version" if damage == "wrong_cache_version" else "key"] = 99
        with open(marker, "w") as handle:
            json.dump(record, handle)

    # Act & assert
    assert cache.read_window_cache(str(tmp_path), "key2", residency=residency) is None


def test_read_window_cache_should_miss_an_entry_whose_window_table_is_truncated(
    tmp_path, tracks
):
    """Test an entry whose Parquet is present but unreadable.

    Given:
        A committed entry whose peak table has been truncated to
        nothing.
    When:
        It is read.
    Then:
        It should answer ``None``, because the file being listed and
        present is all the record can promise and the read is what
        finds out otherwise.
    """

    # Arrange
    windows = frame([("chr1", 8, 16)])
    grouped = cache.group_track(windows, tracks)
    root = cache.write_window_cache(str(tmp_path), "key3", windows, windows, grouped)
    with open(os.path.join(root, "peaks.parquet"), "wb") as handle:
        handle.write(b"not parquet")

    # Act & assert
    assert cache.read_window_cache(str(tmp_path), "key3") is None


def test_read_window_cache_should_miss_a_track_parked_where_an_entry_goes(
    tmp_path, corpus
):
    """Test reading a track as though it were an entry.

    Given:
        A track directory sitting at an entry's key.
    When:
        It is read as an entry.
    Then:
        It should answer nothing, because a track and an entry share a
        schema version and a record shape -- only the fields an entry
        adds tell them apart, and verification alone passes on both.
    """

    # Arrange
    builders.write_track(str(tmp_path / "akey"), [("chr1", 5, 2.0)])

    # Act & assert
    assert cache.read_window_cache(str(tmp_path), "akey") is None
