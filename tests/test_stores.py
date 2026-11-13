"""Tests for the per-window readers.

The contract is that a window read answers exactly what the files hold
at those coordinates, and keeps answering it after the crossing into a
loader worker. So the assertions here are of two kinds: a dense array
or a one-hot matrix built independently of the code under test, and a
pickle round-trip followed by another read.

A reader with a live file handle is the failure a worker provokes, and
it is silent under ``fork``, so the round-trip tests open a handle
first and pickle afterwards -- pickling a store that has never read
anything would pass whether or not the handles are dropped.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import os
import pathlib
import pickle
from collections import Counter

import numpy
import polars
import pyfaidx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat import stores, tables
from tests import FILE_BACKED, TWO_CONTIGS, builders, one_hot

# Two channels over two contigs, with chr2 absent from channel 1, which
# is the contig-missing-from-a-channel case every reader has to answer
# as zeros rather than as an error.
CHANNEL_0 = [("chr1", 10, 1.0), ("chr1", 12, 3.0), ("chr2", 5, 2.0)]


CHANNEL_1 = [("chr1", 11, 5.0)]


# Rows sitting exactly on a window's bounds. The span is half-open, so
# the row at the start is read and the row at the end is not, and no
# row in the corpus above touches an edge of any window the suite asks
# for -- which leaves both comparisons in `TrackStore.window` free to
# move a base in either direction unnoticed.
EDGE_ROWS = [("chr1", 8, 11.0), ("chr1", 12, 3.0), ("chr1", 16, 13.0)]


# One position recorded twice, which the format does not promise and
# nothing rejects. Every reader of a track has to answer the same total
# for it.
DUPLICATE_ROWS = [("chr1", 10, 1.0), ("chr1", 10, 4.0), ("chr1", 12, 3.0)]


# A soft-masked run and an N run, so the one-hot tests have a residue
# that must be upcased and one that must encode to nothing.
SOFT_MASK = (("chr1", 200, 240),)


N_RUNS = (("chr1", 300, 320),)


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    root = corpus_factory(
        "stores",
        TWO_CONTIGS,
        index_fasta=True,
        soft_mask=SOFT_MASK,
        n_runs=N_RUNS,
        tracks=[CHANNEL_0, CHANNEL_1],
        in_window=60,
        out_window=30,
        max_jitter=6,
    )
    return root


@pytest.fixture
def store(corpus):
    return stores.TrackStore(corpus.tracks)


@pytest.fixture(scope="module")
def edge_store(corpus_factory):
    corpus = corpus_factory("stores-edges", TWO_CONTIGS, tracks=[EDGE_ROWS])
    return stores.TrackStore(corpus.tracks)


@pytest.fixture(scope="module")
def duplicate_store(corpus_factory):
    corpus = corpus_factory("stores-duplicates", TWO_CONTIGS, tracks=[DUPLICATE_ROWS])
    return stores.TrackStore(corpus.tracks)


@pytest.fixture
def sequence(corpus):
    return stores.SequenceStore(corpus.fasta)


def dense(rows, start, end):
    """Return the dense ``chr1`` row one channel's ``(chrom, pos, value)`` rows imply."""

    out = numpy.zeros(end - start, dtype=numpy.float32)
    for chrom, pos, value in rows:
        if chrom == "chr1" and start <= pos < end:
            out[pos - start] = value
    return out


# Rows over one contig of the shared corpus, as a reader will meet them:
# positions inside the contig, values a track can hold, and duplicates
# allowed because the format does not forbid them.
track_rows = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=999),
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    min_size=0,
    max_size=25,
)


class TestSequenceStore:
    """The one-hot sequence reader."""

    @pytest.mark.parametrize("source", [None, 42, object()])
    def test___init___should_refuse_a_path_that_is_not_one(self, source):
        """Test the refusal of an argument that is no kind of path.

        Given:
            A reference that is not a path.
        When:
            A store is built from it.
        Then:
            It should raise `TypeError` naming the store and the
            parameter, rather than failing later from inside the FASTA
            provider with a message about neither.
        """

        # Act & assert
        with pytest.raises(TypeError, match="SequenceStore reads sequence"):
            stores.SequenceStore(source)

    @pytest.mark.parametrize("shape", [str, os.fsencode, pathlib.Path])
    def test___init___should_accept_every_path_shape(self, corpus, shape):
        """Test that the accepted path shapes all reach the same reader.

        Given:
            One reference named as a `str`, as `bytes` and as a
            `pathlib.Path`.
        When:
            A store is built from each and asked for one window.
        Then:
            All three should answer identically, so a caller holding
            any of the three shapes Python calls a path is not made to
            convert it first.
        """

        # Arrange
        expected = stores.SequenceStore(corpus.fasta).window("chr1", 100, 110)

        # Act
        window = stores.SequenceStore(shape(corpus.fasta)).window("chr1", 100, 110)

        # Assert
        assert window.tolist() == expected.tolist()

    def test_window_should_one_hot_the_residues_at_those_coordinates(
        self, corpus, sequence
    ):
        """Test the window against the file rather than against the encoder.

        Given:
            A reference and an interval inside one of its contigs.
        When:
            The window is read.
        Then:
            It should be the one-hot matrix of exactly those residues,
            in alphabet row order, with one column per base.
        """

        # Arrange
        residues = str(pyfaidx.Fasta(corpus.fasta)["chr1"][100:140])

        # Act
        window = sequence.window("chr1", 100, 140)

        # Assert
        assert tuple(window.shape) == (4, 40)
        assert numpy.array_equal(window.numpy(), one_hot(residues))

    def test_window_should_upcase_a_soft_masked_residue(self, corpus, sequence):
        """Test that lowercase sequence still encodes to a base.

        Given:
            A soft-masked run, which a reference writes in lowercase.
        When:
            That interval is read.
        Then:
            Every column should still carry exactly one set row, so
            repeat-masked sequence reads as the base it is rather than
            as no base.

        The masking convention is the reference's, not this format's. A
        store that passed it through would silently drop every
        repeat-masked locus to zero.
        """

        # Arrange
        residues = str(pyfaidx.Fasta(corpus.fasta)["chr1"][200:240])
        assert residues.islower()

        # Act
        window = sequence.window("chr1", 200, 240)

        # Assert
        assert window.numpy().sum(axis=0).tolist() == [1] * 40

    def test_window_should_order_the_rows_by_the_alphabet_it_was_given(self, corpus):
        """Test that the alphabet reaches the encoder.

        Given:
            Two stores over one reference, one with the default
            alphabet and one with the same residues in another order.
        When:
            Both read the same window.
        Then:
            The second's rows should be the first's permuted to that
            order, because the alphabet is the row axis.

        The encoder's own default is the same four residues in the same
        order as this store's, so dropping the argument entirely left
        every window test passing.
        """

        # Arrange
        default = stores.SequenceStore(corpus.fasta)
        reordered = stores.SequenceStore(corpus.fasta, alphabet=("T", "G", "C", "A"))

        # Act
        rows = default.window("chr1", 100, 140).numpy()
        moved = reordered.window("chr1", 100, 140).numpy()

        # Assert
        assert numpy.array_equal(moved, rows[[3, 2, 1, 0]])

    def test_window_should_encode_a_residue_the_caller_does_not_ignore(self, corpus):
        """Test that the ignore set reaches the encoder.

        Given:
            A store whose alphabet includes ``N`` and whose ignore set
            is empty, over a run of ``N``.
        When:
            The window is read.
        Then:
            The ``N`` row should be set, because the caller narrowed the
            ignore set and an ambiguity code is a residue like any
            other once it has a row.

        The encoder's own default ignores ``N``, exactly as this store
        does, so dropping the argument left the test for the default
        behaviour passing.
        """

        # Arrange
        store = stores.SequenceStore(
            corpus.fasta, alphabet=("A", "C", "G", "T", "N"), ignore=()
        )

        # Act
        window = store.window("chr1", 300, 320).numpy()

        # Assert
        assert tuple(window.shape) == (5, 20)
        assert window[4].tolist() == [1] * 20

    def test_window_should_zero_a_column_for_an_ignored_residue(self, sequence):
        """Test what an ambiguous base encodes to.

        Given:
            A run of ``N``, which is in the ignore set.
        When:
            That interval is read.
        Then:
            Every column should be all zero, so an ambiguous base reads
            as no base rather than as one of the four.
        """

        # Act
        window = sequence.window("chr1", 300, 320)

        # Assert
        assert window.numpy().sum() == 0

    def test_window_should_raise_when_a_residue_is_neither_encoded_nor_ignored(
        self, corpus
    ):
        """Test the refusal for a residue the store has no provision for.

        Given:
            A store with the default alphabet and nothing ignored, over
            a run of ``N``.
        When:
            That interval is read.
        Then:
            It should raise `ValueError` saying a residue is in neither
            the alphabet nor the ignored residues.
        """

        # Arrange
        store = stores.SequenceStore(corpus.fasta, ignore=())

        # Act & assert
        with pytest.raises(ValueError, match="not in `alphabet` or in `ignore`"):
            store.window("chr1", 300, 320)

    @pytest.mark.parametrize(
        "start, end",
        [(990, 1010), (1000, 1010), (-5, 10), (1200, 1220)],
        ids=["over_the_end", "at_the_end", "before_the_start", "past_the_contig"],
    )
    def test_window_should_pad_a_span_reaching_outside_the_contig(
        self, sequence, start, end
    ):
        """Test the shape of a window that does not fit inside the contig.

        Given:
            A span running past the contig's end, starting at its end,
            starting before zero, or lying wholly past it.
        When:
            The window is read.
        Then:
            It should be as wide as the span asked for in every case,
            with the positions outside the contig all zero, so the
            shape is a function of the span rather than of where the
            span sits.

            `TrackStore.window` answers the full span for these same
            coordinates, and two readers of one locus disagreeing on
            width is a wrong-shaped pair much later rather than an
            error here.
        """

        # Act
        window = sequence.window("chr1", start, end).numpy()

        # Assert
        assert tuple(window.shape) == (4, end - start)
        assert window[:, : max(0, -start)].sum() == 0
        assert window[:, min(end - start, max(0, 1000 - start)) :].sum() == 0

    def test_window_should_match_the_signal_readers_width_for_one_span(
        self, sequence, store
    ):
        """Test that the two readers agree on the width of a span.

        Given:
            A span running past the contig's end.
        When:
            Sequence and signal are both read over it.
        Then:
            They should be the same number of columns wide, because a
            training sample pairs one with the other and nothing
            downstream checks that they match.
        """

        # Act
        residues = sequence.window("chr1", 990, 1010)
        signal = store.window("chr1", 990, 1010)

        # Assert
        assert residues.shape[1] == signal.shape[1]

    def test_window_should_refuse_a_span_that_runs_backwards(self, sequence):
        """Test a window whose end is before its start.

        Given:
            A span given back to front.
        When:
            The window is read.
        Then:
            It should refuse, rather than answering the empty window
            the clipping would otherwise produce.
        """

        # Act & assert
        with pytest.raises(ValueError, match="SequenceStore.window"):
            sequence.window("chr1", 20, 10)

    def test_window_should_answer_the_same_after_a_pickle_round_trip(self, sequence):
        """Test that a store survives the crossing into a worker.

        Given:
            A store that has already read a window, so its reference is
            open.
        When:
            It is pickled, restored, and asked again.
        Then:
            It should answer identically, which requires the open
            handle to have been dropped on the way out: a
            ``pyfaidx.Fasta`` cannot be pickled at all.

        A store pickled before its first read would pass this whether
        or not the handles are process-local, which is why the read
        comes first.
        """

        # Arrange
        expected = sequence.window("chr1", 100, 140)

        # Act
        restored = pickle.loads(pickle.dumps(sequence))

        # Assert
        assert restored.window("chr1", 100, 140).tolist() == expected.tolist()

    def test_window_should_reopen_the_reference_after_the_pid_moves(
        self, mocker, sequence
    ):
        """Test that a forked child does not read the parent's handle.

        Given:
            A store that has opened its reference.
        When:
            The process id changes and another window is read.
        Then:
            It should open the reference again rather than reuse the
            descriptor it inherited, because two processes sharing one
            descriptor seek the same file out from under each other and
            neither read looks wrong afterwards.
        """

        # Arrange
        opened = mocker.spy(pyfaidx, "Fasta")
        sequence.window("chr1", 100, 110)
        mocker.patch.object(os, "getpid", return_value=os.getpid() + 1)

        # Act
        sequence.window("chr1", 100, 110)

        # Assert
        assert opened.call_count == 2

    def test_window_should_raise_os_error_when_no_index_can_be_written(
        self, unindexable_fasta
    ):
        """Test the refusal for a reference that cannot be indexed in place.

        Given:
            A reference with no index, in a directory that cannot be
            written to.
        When:
            A window is read.
        Then:
            It should raise `OSError` naming the reference and the
            command that indexes it.
        """

        # Arrange
        unindexed = stores.SequenceStore(unindexable_fasta)

        # Act & assert
        with pytest.raises(
            OSError, match=r"genome\.fa has no usable faidx index.*`samtools faidx \S+`"
        ):
            unindexed.window("chr1", 0, 8)

    def test_window_should_raise_as_pyfaidx_does_when_the_reference_is_missing(
        self, tmp_path
    ):
        """Test that a missing reference is not reported as a missing index.

        Given:
            A path at which there is no reference.
        When:
            A window is read.
        Then:
            It should raise `pyfaidx.FastaNotFoundError`, an `OSError`,
            naming the path and saying nothing of an index.
        """

        # Arrange
        missing = stores.SequenceStore(str(tmp_path / "missing.fa"))

        # Act & assert
        with pytest.raises(pyfaidx.FastaNotFoundError, match=r"missing\.fa") as caught:
            missing.window("chr1", 0, 8)
        assert isinstance(caught.value, OSError)
        assert "faidx index" not in str(caught.value)

    @settings(FILE_BACKED, max_examples=25)
    @given(bounds=st.tuples(st.integers(-200, 1200), st.integers(-200, 1200)))
    def test_window_should_be_as_wide_as_any_span_it_is_given(self, sequence, bounds):
        """Test the shape invariant over spans inside and outside the contig.

        Given:
            Any span, including ones reaching past either end of the
            contig.
        When:
            The window is read.
        Then:
            It should be as wide as the span and every column should sum to
            zero or one, so the shape is a function of the span alone and
            no column claims two residues.
        """

        # Arrange
        start, end = min(bounds), max(bounds)

        # Act
        window = sequence.window("chr1", start, end).numpy()

        # Assert
        assert tuple(window.shape) == (4, end - start)
        assert set(window.sum(axis=0).tolist()) <= {0, 1}

    def test_contig_lengths_should_answer_every_contig_in_the_references_order(
        self, sequence
    ):
        """Test the lengths a store reads off its reference's index.

        Given:
            A store over an indexed reference of a known geometry.
        When:
            Its contig lengths are asked for.
        Then:
            It should answer every contig at the length it was written
            at, in the reference's own order.
        """

        # Act
        lengths = sequence.contig_lengths()

        # Assert
        assert list(lengths.items()) == list(TWO_CONTIGS.items())

    def test_contig_lengths_should_build_the_index_a_reference_lacks(self, tmp_path):
        """Test that asking for lengths counts as a first read.

        Given:
            A reference with no index, in a directory that can be
            written to.
        When:
            Its contig lengths are asked for.
        Then:
            It should answer them and leave the index beside the
            reference.
        """

        # Arrange
        path = builders.write_fasta(str(tmp_path / "genome.fa"), TWO_CONTIGS)

        # Act
        lengths = stores.SequenceStore(path).contig_lengths()

        # Assert
        assert lengths == dict(TWO_CONTIGS)
        assert os.path.exists(path + ".fai")

    def test_contig_lengths_should_raise_os_error_when_no_index_can_be_written(
        self, unindexable_fasta
    ):
        """Test the refusal for a reference that cannot be indexed in place.

        Given:
            A reference with no index, in a directory that cannot be
            written to.
        When:
            Its contig lengths are asked for.
        Then:
            It should raise `OSError` naming the reference and the
            command that indexes it.
        """

        # Arrange
        unindexed = stores.SequenceStore(unindexable_fasta)

        # Act & assert
        with pytest.raises(
            OSError, match=r"genome\.fa has no usable faidx index.*`samtools faidx \S+`"
        ):
            unindexed.contig_lengths()


def test_chrom_plan_should_answer_none_for_a_contig_the_track_lacks(corpus):
    """Test what a track with no rows for a contig answers.

    Given:
        A channel that recorded nothing on a contig.
    When:
        Its plan for that contig is asked for.
    Then:
        It should be ``None``, which is what tells a caller to leave
        the window at zero rather than to read an empty frame.
    """

    # Act & assert
    assert stores.chrom_plan(corpus.tracks[1], "chr2") is None


def test_chrom_plan_should_read_a_table_written_as_a_single_file(tmp_path):
    """Test a plan over a single Parquet file holding every contig.

    Given:
        One Parquet file holding every contig's rows.
    When:
        A plan for one contig is asked for and collected.
    Then:
        It should carry that contig's rows and no other's, so a track
        in the single-file layout reads as a directory does.

    The suffix is write-side only and a reader is handed a path, so a
    caller holding a single-file table is not made to rewrite it.
    """

    # Arrange
    path = str(tmp_path / "flat.parquet")
    polars.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr1"],
            "pos": [10, 5, 12],
            "value": [1.0, 2.0, 3.0],
        },
        schema={"chrom": polars.Utf8, "pos": polars.Int64, "value": polars.Float64},
    ).write_parquet(path)

    # Act
    rows = stores.chrom_plan(path, "chr1").collect().sort("pos")

    # Assert
    assert rows.get_column("pos").to_list() == [10, 12]
    assert rows.get_column("value").to_list() == [1.0, 3.0]


def test_chrom_plan_should_project_away_the_contig_column(corpus):
    """Test the columns a plan carries.

    Given:
        A track directory, whose per-contig files each hold a contig
        column as well as the two a reader wants.
    When:
        A plan for one contig is asked for and its schema read.
    Then:
        It should carry the position and the value and nothing else,
        because the contig is already known from the file the plan
        opens and every caller indexes the result by name.
    """

    # Act
    plan = stores.chrom_plan(corpus.tracks[0], "chr1")

    # Assert
    assert plan.collect_schema().names() == ["pos", "value"]


def test_chrom_plan_should_read_a_contig_whose_name_needs_quoting(tmp_path):
    """Test a contig named with a character a filesystem reserves.

    Given:
        A track holding a contig whose name carries a slash, a space
        and a percent sign.
    When:
        Its plan is asked for and collected.
    Then:
        It should carry that contig's rows, because the per-contig
        filename is percent-quoted and this is the only reader of that
        quoting.
    """

    # Arrange
    name = "chr1/a b%c"
    path = str(tmp_path / "t.track.parquet")
    builders.write_track(path, [(name, 5, 2.0)])

    # Act
    rows = stores.chrom_plan(path, name).collect()

    # Assert
    assert rows["pos"].to_list() == [5]
    assert rows["value"].to_list() == [2.0]


def test_chrom_track_should_answer_positions_in_order(tmp_path):
    """Test the order a collected contig comes back in.

    Given:
        A single-file table whose rows for one contig descend by
        position, which is the layout this package does not write and
        still reads.
    When:
        The contig is collected.
    Then:
        The positions should ascend with each value still beside its
        own position, because both callers search the result -- the
        grouping with two searches and the group sums with a prefix
        sum -- and an unsorted array answers neither.
    """

    # Arrange
    path = str(tmp_path / "flat.parquet")
    polars.DataFrame(
        {
            "chrom": ["chr1"] * 3,
            "pos": [12, 8, 10],
            "value": [3.0, 1.0, 2.0],
        },
        schema={"chrom": polars.Utf8, "pos": polars.Int64, "value": polars.Float64},
    ).write_parquet(path)

    # Act
    positions, values = stores.chrom_track(path, "chr1")

    # Assert
    assert positions.tolist() == [8, 10, 12]
    assert values.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("layout", ["directory", "single_file"])
def test_chrom_track_should_answer_none_for_a_contig_with_no_rows(tmp_path, layout):
    """Test a contig neither layout has rows for.

    Given:
        A track as a directory and as a single file, neither holding
        the contig asked for.
    When:
        It is collected.
    Then:
        It should answer nothing in both cases, so a caller leaves that
        window at zero -- where a plan for a single-file table answers
        an empty plan rather than nothing, and the difference would
        otherwise reach the caller.
    """

    # Arrange
    if layout == "directory":
        source = str(tmp_path / "t.track.parquet")
        builders.write_track(source, [("chr1", 5, 2.0)])
    else:
        source = str(tmp_path / "flat.parquet")
        polars.DataFrame(
            {"chrom": ["chr1"], "pos": [5], "value": [2.0]},
            schema={
                "chrom": polars.Utf8,
                "pos": polars.Int64,
                "value": polars.Float64,
            },
        ).write_parquet(source)

    # Act & assert
    assert stores.chrom_track(source, "chr2") is None


@settings(FILE_BACKED, max_examples=25)
@given(rows=track_rows)
def test_chrom_track_should_answer_sorted_rows_for_any_input(tmp_path, rows):
    """Test the sort and the round trip over arbitrary rows.

    Given:
        Any rows over one contig, in any order, with any duplicates.
    When:
        The contig is collected.
    Then:
        The positions should ascend and the pairs should be the same
        multiset that was written, because both callers search the
        result and neither may have rows reordered out from under their
        values.
    """

    # Arrange
    path = str(tmp_path / "property.track.parquet")
    builders.write_track(path, [("chr1", pos, value) for pos, value in rows])

    # Act
    answer = stores.chrom_track(path, "chr1")

    # Assert
    if not rows:
        assert answer is None
        return
    positions, values = answer
    assert positions.tolist() == sorted(positions.tolist())
    assert Counter(zip(positions.tolist(), values.tolist())) == Counter(
        (pos, numpy.float64(value)) for pos, value in rows
    )


@pytest.mark.parametrize("shape", ["directory", "single_file", "eager", "lazy"])
def test_track_frame_should_read_every_shape_a_track_takes(tmp_path, shape):
    """Test the four shapes a whole-table read accepts.

    Given:
        One track's rows as a directory, as a single Parquet file, and
        as an eager and a lazy polars frame.
    When:
        Each is read as a whole frame and collected.
    Then:
        All four should carry the same rows over every contig at once,
        because a caller that filters on the contig itself should not
        have to care which shape it was handed -- and the in-memory
        shapes are what let a caller go from alignments to tensors with
        no track on disk.
    """

    # Arrange
    rows = [("chr1", 5, 2.0), ("chr2", 7, 3.0)]
    frame = builders.track_rows_frame(rows)
    if shape == "directory":
        source = str(tmp_path / "t.track.parquet")
        builders.write_track(source, rows)
    elif shape == "single_file":
        source = str(tmp_path / "flat.parquet")
        frame.write_parquet(source)
    else:
        source = frame if shape == "eager" else frame.lazy()

    # Act
    table = stores.track_frame(source).collect().sort("chrom")

    # Assert
    assert table["chrom"].to_list() == ["chr1", "chr2"]
    assert table["pos"].to_list() == [5, 7]
    assert table["value"].to_list() == [2.0, 3.0]


def test_track_frame_should_refuse_a_shape_that_is_no_track():
    """Test a value that is none of the shapes a track takes.

    Given:
        An integer.
    When:
        It is read as a whole track.
    Then:
        It should refuse, naming the suffix and the shapes it accepts,
        so a caller is told what to pass rather than what went wrong
        inside.
    """

    # Act & assert
    with pytest.raises(
        TypeError,
        match=(
            r"track_frame reads a track: a \.track\.parquet directory, a "
            r"single Parquet file, or an in-memory polars frame"
        ),
    ):
        stores.track_frame(7)


@pytest.mark.parametrize("name", ["mistyped.parquet", "mistyped.track.parquet"])
def test_track_frame_should_raise_when_nothing_is_at_the_path(tmp_path, name):
    """Test a whole-table read of a path at which nothing exists.

    Given:
        A path under a single-file or a directory track's name, with
        no file or directory at it.
    When:
        It is read as a whole frame, and the frame is not collected.
    Then:
        It should raise `FileNotFoundError` naming the path, from the
        read itself rather than from a later collect.
    """

    # Arrange
    path = str(tmp_path / name)

    # Act & assert
    with pytest.raises(
        FileNotFoundError,
        match=r"{} is not a track; there is no such file".format(
            name.replace(".", r"\.")
        ),
    ):
        stores.track_frame(path)


def test_track_frame_should_refuse_a_directory_that_does_not_verify(tmp_path):
    """Test a whole-table read of a track whose write did not finish.

    Given:
        A track directory whose record has been removed.
    When:
        It is read as a whole frame.
    Then:
        It should refuse, because this read globs every contig together
        and a contig whose file never landed would otherwise read as a
        contig with nothing in it.

        The per-contig readers do not verify, deliberately: the store
        verifies once at construction, and these are reached through
        it. This one is reached directly.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    builders.write_track(path, [("chr1", 5, 2.0)])
    os.remove(tables.manifest_path(path))

    # Act & assert
    with pytest.raises(ValueError):
        stores.track_frame(path)


class TestTrackStore:
    """The dense signal reader over one track per channel."""

    def test___init___should_refuse_a_single_path(self, corpus):
        """Test the refusal of one track where a channel set is wanted.

        Given:
            One track, passed as a bare path.
        When:
            A store is built from it.
        Then:
            It should raise `TypeError` saying so and showing the
            one-channel form, rather than iterating the path into its
            characters and verifying each as a track.
        """

        # Act & assert
        with pytest.raises(TypeError, match="not a single path"):
            stores.TrackStore(corpus.tracks[0])

    def test___init___should_refuse_a_source_that_is_not_a_path(self, corpus):
        """Test the refusal of a channel that is no kind of path.

        Given:
            A channel set holding one real track and one wrong type.
        When:
            A store is built from it.
        Then:
            It should raise `TypeError` naming the channel's index, so
            a caller with sixty-eight channels is told which one.
        """

        # Act & assert
        with pytest.raises(TypeError, match="source 1 must be a str"):
            stores.TrackStore([corpus.tracks[0], 42])

    def test___init___should_refuse_an_unverifiable_track(self, tmp_path, corpus):
        """Test that an incomplete channel is refused up front.

        Given:
            A channel set holding a track whose record is missing.
        When:
            A store is built from it.
        Then:
            It should raise rather than build, because a truncated
            channel reads as a contig with no signal and is otherwise
            indistinguishable from one that is genuinely empty.

        Verified once here rather than per read: the per-read path runs
        once per contig per channel, and the answer cannot change
        mid-epoch.
        """

        # Arrange
        root = str(tmp_path / "partial.track.parquet")
        builders.write_track(root, CHANNEL_0)
        os.remove(tables.manifest_path(root))

        # Act & assert
        with pytest.raises(ValueError, match="cannot be verified complete"):
            stores.TrackStore([corpus.tracks[0], root])

    def test_n_channels_should_count_the_channels(self, store):
        """Test the channel count a store reports.

        Given:
            A store over two tracks.
        When:
            Its channel count is read.
        Then:
            It should be the source count, which is the height of
            every window it answers.
        """

        # Act & assert
        assert store.n_channels == 2

    def test_window_should_place_every_channel_at_its_own_positions(self, store):
        """Test a window against the rows the tracks were built from.

        Given:
            Two channels whose recorded positions interleave inside one
            interval.
        When:
            The window is read.
        Then:
            It should be the dense array those rows imply, each
            channel's values at its own offsets and zero everywhere
            else.
        """

        # Act
        window = store.window("chr1", 8, 16).numpy()

        # Assert
        assert tuple(window.shape) == (2, 8)
        assert numpy.array_equal(window[0], dense(CHANNEL_0, 8, 16))
        assert numpy.array_equal(window[1], dense(CHANNEL_1, 8, 16))

    def test_window_should_answer_zeros_for_a_contig_a_channel_lacks(self, store):
        """Test the read of a contig one channel never recorded.

        Given:
            A contig present in channel 0 and absent from channel 1.
        When:
            A window over it is read.
        Then:
            Channel 0 should carry its values and channel 1 should be
            all zero, rather than the read failing.

        A track holds a file only for the contigs it recorded, so a
        channel with nothing on a small unplaced contig is routine and
        is not a truncated channel; that case is `require_complete`'s.
        """

        # Act
        window = store.window("chr2", 0, 8).numpy()

        # Assert
        assert numpy.array_equal(window[0], numpy.array([0, 0, 0, 0, 0, 2, 0, 0]))
        assert window[1].sum() == 0

    def test_window_should_answer_zeros_outside_every_recorded_position(self, store):
        """Test an interval no channel recorded anything in.

        Given:
            An interval well away from every recorded position.
        When:
            The window is read.
        Then:
            It should be the right shape and all zero, so an empty
            window is a shape rather than an absence.
        """

        # Act
        window = store.window("chr1", 500, 520).numpy()

        # Assert
        assert tuple(window.shape) == (2, 20)
        assert window.sum() == 0

    def test_window_should_read_a_row_sitting_on_the_windows_start(self, edge_store):
        """Test the lower bound of the span a window covers.

        Given:
            A track whose rows include one at exactly the window's
            start.
        When:
            The window is read.
        Then:
            That row should be read, at column zero, because the span
            is closed at its start.
        """

        # Act
        window = edge_store.window("chr1", 8, 16).numpy()

        # Assert
        assert window[0][0] == 11.0

    def test_window_should_omit_a_row_sitting_on_the_windows_end(self, edge_store):
        """Test the upper bound of the span a window covers.

        Given:
            A track whose rows include one at exactly the window's end.
        When:
            The window is read.
        Then:
            That row should be absent and the window should hold only
            the two rows inside it, because the span is open at its
            end -- so the row at the end belongs to the next window and
            is not counted twice.
        """

        # Act
        window = edge_store.window("chr1", 8, 16).numpy()

        # Assert
        assert numpy.array_equal(
            window[0], numpy.array([11, 0, 0, 0, 3, 0, 0, 0], dtype=numpy.float32)
        )

    def test_window_should_total_a_position_recorded_twice(self, duplicate_store):
        """Test what a duplicated position reads as.

        Given:
            A track recording one position twice, with different
            values.
        When:
            A window covering it is read.
        Then:
            That column should hold the sum of the two, because a track
            promises one value per position and a reader that keeps one
            of them drops signal without saying so.

            `window_group_sums` totals both rows too, so a reader that
            kept one would leave three readers of one track answering
            two different numbers.
        """

        # Act
        window = duplicate_store.window("chr1", 8, 16).numpy()

        # Assert
        assert window[0][2] == 5.0
        assert window[0][4] == 3.0

    def test_window_should_refuse_a_span_that_runs_backwards(self, store):
        """Test a window whose end is before its start.

        Given:
            A span given back to front.
        When:
            The window is read.
        Then:
            It should refuse, naming this reader and the contig,
            because numpy refuses the negative width on its own and
            names neither -- in a module where every other refusal
            names both.
        """

        # Act & assert
        with pytest.raises(ValueError, match=r"TrackStore\.window.*on 'chr1'"):
            store.window("chr1", 20, 10)

    def test_window_should_answer_float32_whatever_the_file_holds(self, store):
        """Test the dtype a window carries.

        Given:
            Tracks whose values are stored as Float64, which is what
            the format writes.
        When:
            A window is read.
        Then:
            It should be ``float32``, because narrowing is free in
            memory and the format deliberately does not narrow on disk.
        """

        # Act & assert
        assert store.window("chr1", 8, 16).dtype is torch.float32

    def test_window_should_answer_the_same_after_a_pickle_round_trip(self, store):
        """Test that a store survives the crossing into a worker.

        Given:
            A store that has already read a window, so its scan plans
            are built.
        When:
            It is pickled, restored, and asked again.
        Then:
            It should answer identically, so a dataset handed to a
            worker reads what the dataset in the parent would have.

        Current polars pickles a scan plan, so this passes whether or
        not the plans are dropped. That they are dropped is the next
        test's claim, and the two are separate because only one of them
        would survive polars making a plan unpicklable again.
        """

        # Arrange
        expected = store.window("chr1", 8, 16).tolist()

        # Act
        restored = pickle.loads(pickle.dumps(store))

        # Assert
        assert restored.window("chr1", 8, 16).tolist() == expected

    def test_window_should_rebuild_its_plans_after_a_pickle_round_trip(
        self, mocker, store
    ):
        """Test that a restored store carries no plan from the original.

        Given:
            A store whose plan for a contig is already built.
        When:
            It is pickled, restored, and asked for that same contig.
        Then:
            It should build the plan again -- one new Parquet scan per
            channel -- so a parent's cache is not shipped into every
            worker for plans the worker may never ask for.

        This holds the pruning and the restore jointly, not either one:
        each is sufficient by itself, so removing one leaves this
        passing and removing both fails it. See
        ``ProcessLocalHandles``.
        """

        # Arrange
        store.window("chr1", 8, 16)
        restored = pickle.loads(pickle.dumps(store))
        scanned = mocker.spy(polars, "scan_parquet")

        # Act
        restored.window("chr1", 8, 16)

        # Assert
        assert scanned.call_count == store.n_channels

    @settings(FILE_BACKED, max_examples=25)
    @given(rows=track_rows, bounds=st.tuples(st.integers(0, 999), st.integers(0, 999)))
    def test_window_should_equal_the_dense_array_any_rows_imply(
        self, tmp_path, rows, bounds
    ):
        """Test the window read against an independently built array.

        Given:
            Any rows over one contig, including duplicate positions, and
            any span inside that contig.
        When:
            A window is read over that span.
        Then:
            It should equal the dense row those rows imply, built here
            rather than read, with duplicates summed.

        One property rather than a case per shape: the reader's whole
        contract is that it answers what the file holds at coordinates, so
        the arithmetic worth testing over a wide domain is which rows fall
        inside the span and where they land.
        """

        # Arrange
        start, end = min(bounds), max(bounds)
        path = str(tmp_path / "property.track.parquet")
        builders.write_track(path, [("chr1", pos, value) for pos, value in rows])
        store = stores.TrackStore([path])
        expected = numpy.zeros(end - start, dtype=numpy.float64)
        for pos, value in rows:
            if start <= pos < end:
                expected[pos - start] += value
        expected = expected.astype(numpy.float32)

        # Act
        window = store.window("chr1", start, end).numpy()

        # Assert
        assert tuple(window.shape) == (1, end - start)
        assert numpy.allclose(window[0], expected, rtol=1e-5, atol=1e-5)
