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

import numpy
import pytest
import torch

from quadrat import stores, tables
from tests import TWO_CONTIGS, builders

# Two channels over two contigs, with chr2 absent from channel 1, which
# is the contig-missing-from-a-channel case every reader has to answer
# as zeros rather than as an error.
CHANNEL_0 = [("chr1", 10, 1.0), ("chr1", 12, 3.0), ("chr2", 5, 2.0)]


CHANNEL_1 = [("chr1", 11, 5.0)]


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
    pytest.importorskip("polars")
    return stores.TrackStore(corpus.tracks)


@pytest.fixture
def sequence(corpus):
    pytest.importorskip("pyfaidx")
    return stores.SequenceStore(corpus.fasta)


def dense(rows, start, end):
    """The dense row one channel's ``(chrom, pos, value)`` rows imply."""

    out = numpy.zeros(end - start, dtype=numpy.float32)
    for chrom, pos, value in rows:
        if chrom == "chr1" and start <= pos < end:
            out[pos - start] = value
    return out


def one_hot(residues):
    """The one-hot matrix a residue string implies, built here rather than read."""

    out = numpy.zeros((4, len(residues)), dtype=numpy.int8)
    for column, base in enumerate(residues.upper()):
        if base in "ACGT":
            out["ACGT".index(base), column] = 1
    return out


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
        pytest.importorskip("pyfaidx")
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
        pyfaidx = pytest.importorskip("pyfaidx")
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
        pyfaidx = pytest.importorskip("pyfaidx")
        residues = str(pyfaidx.Fasta(corpus.fasta)["chr1"][200:240])
        assert residues.islower()

        # Act
        window = sequence.window("chr1", 200, 240)

        # Assert
        assert window.numpy().sum(axis=0).tolist() == [1] * 40

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
        pyfaidx = pytest.importorskip("pyfaidx")
        opened = mocker.spy(pyfaidx, "Fasta")
        sequence.window("chr1", 100, 110)
        mocker.patch.object(os, "getpid", return_value=os.getpid() + 1)

        # Act
        sequence.window("chr1", 100, 110)

        # Assert
        assert opened.call_count == 2


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
        pytest.importorskip("polars")
        root = str(tmp_path / "partial.track.parquet")
        builders.write_track(root, CHANNEL_0)
        os.remove(tables.manifest_path(root))

        # Act & assert
        with pytest.raises(ValueError, match="cannot be verified complete"):
            stores.TrackStore([corpus.tracks[0], root])

    def test___len___should_count_the_channels(self, store):
        """Test the length of a store.

        Given:
            A store over two tracks.
        When:
            Its length is taken.
        Then:
            It should be the channel count, so a store stands in for
            its channel axis wherever a length is wanted.
        """

        # Act & assert
        assert len(store) == 2

    def test_n_channels_should_count_the_channels(self, store):
        """Test the channel count a store reports.

        Given:
            A store over two tracks.
        When:
            Its channel count is read.
        Then:
            It should agree with its length, which is the height of
            every window it answers.
        """

        # Act & assert
        assert store.n_channels == len(store) == 2

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
            It should build the plan again, so a parent's cache is not
            shipped into every worker for plans the worker may never
            ask for.

        This holds the pruning and the restore jointly, not either one:
        each is sufficient by itself, so removing one leaves this
        passing and removing both fails it. See
        ``_ProcessLocalHandles``.
        """

        # Arrange
        store.window("chr1", 8, 16)
        restored = pickle.loads(pickle.dumps(store))
        planned = mocker.spy(stores, "chrom_plan")

        # Act
        restored.window("chr1", 8, 16)

        # Assert
        assert planned.call_count == store.n_channels


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

    # Arrange
    pytest.importorskip("polars")

    # Act & assert
    assert stores.chrom_plan(corpus.tracks[1], "chr2") is None


def test_chrom_plan_should_read_a_table_written_as_a_single_file(tmp_path):
    """Test the layout written before the format became a directory.

    Given:
        One Parquet file holding every contig's rows.
    When:
        A plan for one contig is asked for and collected.
    Then:
        It should carry that contig's rows and no other's, so a table
        written under the older layout still reads.

    The suffix is write-side only and a reader is handed a path, so a
    caller holding a pre-directory table is not made to rewrite it.
    """

    # Arrange
    polars = pytest.importorskip("polars")
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
