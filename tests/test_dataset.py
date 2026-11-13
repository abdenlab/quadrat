"""Tests for the dataset, which turns one draw into one sample's tensors.

The contract is that a draw reads the bases and the signal its
coordinates name. So the central assertions are against oracles that
share nothing with the code under test: the reference parsed from its
own lines and the tracks' rows densified by hand, over arbitrary
draws; and, once, alignments counted with pysam against the tracks a
pileup of them wrote.

Around that sit the properties a reader of the draws relies on. Jitter
shifts by exactly what it says, a mirrored draw permutes within signal
groups and never across them, signal and control groupings do not
consult each other, and a copy of the dataset in another process reads
what the original reads.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import pickle
from types import MappingProxyType

import numpy
import pandas
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat import Aggregate, cache, dataset, groups, loci, pileup, sampling, stores
from tests import (
    TWO_CONTIGS,
    WORKER_TIMEOUT,
    builders,
    dense,
    drain,
    one_hot,
    oracle_pileup,
    parse_fasta,
    same_batches,
)

CHROM_SIZES = MappingProxyType({"chr1": 4000, "chr2": 3000})


IN_WINDOW, OUT_WINDOW, MAX_JITTER = 100, 50, 8


# What the piled corpus is piled up under: ATAC's Tn5 shift, unstranded.
ATAC = Aggregate(start_shift=4, end_shift=-4, unstranded=True)


# Twenty-bp loci, every one far enough inside its contig that its
# widest window is in bounds, so a locus's row in its file is its row
# in the window frame.
PEAKS = builders.intervals("chr1", 600, 2600, 110, 20) + builders.intervals(
    "chr2", 600, 1700, 130, 20
)


NEGATIVES = builders.intervals("chr1", 2700, 3500, 90, 20) + builders.intervals(
    "chr2", 1800, 2600, 95, 20
)


# Four signal channels and four control channels, so any group layout
# of up to four channels is a prefix of each.
SIGNAL_ROWS = [builders.scattered_rows(CHROM_SIZES, seed) for seed in (10, 11, 12, 13)]


CONTROL_ROWS = [builders.scattered_rows(CHROM_SIZES, seed) for seed in (20, 21, 22, 23)]


# A soft-masked run and a run of N under the first peaks of each
# contig, so a read has to upper-case the one and zero the other.
SOFT_MASK = (("chr1", 600, 660), ("chr2", 600, 660))


N_RUNS = (("chr1", 700, 710), ("chr2", 730, 740))


# Any draw the module's corpus can answer: a peak at any jitter inside
# its margin, or a negative at none, in either orientation.
corpus_draws = st.one_of(
    st.builds(
        sampling.LocusDraw,
        st.just(True),
        st.integers(min_value=0, max_value=len(PEAKS) - 1),
        st.integers(min_value=0, max_value=2 * MAX_JITTER - 1),
        st.booleans(),
    ),
    st.builds(
        sampling.LocusDraw,
        st.just(False),
        st.integers(min_value=0, max_value=len(NEGATIVES) - 1),
        st.just(0),
        st.booleans(),
    ),
)


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Build the reference, locus files and eight tracks the module reads."""

    pytest.importorskip("polars_bio")
    return corpus_factory(
        "dataset",
        CHROM_SIZES,
        soft_mask=SOFT_MASK,
        n_runs=N_RUNS,
        peaks=PEAKS,
        negatives=NEGATIVES,
        tracks=SIGNAL_ROWS + CONTROL_ROWS,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        max_jitter=MAX_JITTER,
    )


@pytest.fixture(scope="module")
def plain(corpus):
    """Return a dataset over three signal channels grouped one and two."""

    return build_dataset(corpus, signal_groups=[1, 2])


@pytest.fixture(scope="module")
def controlled(corpus):
    """Return a dataset with signals grouped one and two and one control pair."""

    return build_dataset(corpus, signal_groups=[1, 2], control_groups=[2])


@pytest.fixture(scope="module")
def piled(corpus_factory, pysam):
    """Build two alignments, the track a pileup of each writes, and loci over them."""

    pytest.importorskip("polars_bio")
    built = corpus_factory(
        "dataset-piled",
        TWO_CONTIGS,
        n_bams=2,
        n_reads=400,
        index_fasta=True,
        peaks=builders.intervals("chr1", 100, 900, 90, 20),
        negatives=builders.intervals("chr2", 100, 400, 70, 20),
        in_window=60,
        out_window=30,
        max_jitter=5,
    )
    for index, bam in enumerate(built.bams):
        built.tracks += pileup.pileup_alignments(
            [bam],
            built.fasta,
            built.path("sample{}".format(index)),
            aggregate=ATAC,
        )
    return built


def kept_windows(corpus):
    """Return the in-bounds peak and negative window frames of a corpus."""

    frames = []
    for source, jitter in ((corpus.peaks, corpus.max_jitter), (corpus.negatives, 0)):
        windows = loci.load_windows(
            source, corpus.fasta, corpus.in_window, corpus.out_window, max_jitter=jitter
        )
        frames.append(windows.filter(windows["in_bounds"]))
    return frames


def build_dataset(
    corpus,
    signal_groups,
    control_groups=None,
    signal_tracks=None,
    control_tracks=None,
    sampler=None,
):
    """Return a dataset over a corpus's loci, with caches built for its windows.

    The signal tracks default to the first ``sum(signal_groups)`` of
    the module corpus's signal channels and the control tracks to the
    first ``sum(control_groups)`` of its control channels.
    """

    peak_windows, negative_windows = kept_windows(corpus)
    joint = cache.joint_windows(peak_windows, negative_windows)
    if signal_tracks is None:
        signal_tracks = corpus.tracks[: sum(signal_groups)]
    if control_tracks is None and control_groups is not None:
        control_tracks = corpus.tracks[4 : 4 + sum(control_groups)]
    return dataset.LocusDataset(
        peak_windows,
        negative_windows,
        stores.SequenceStore(corpus.fasta),
        cache.WindowSignalCache.build(joint, signal_tracks, residency="memory"),
        corpus.in_window,
        corpus.out_window,
        control_cache=(
            None
            if control_tracks is None
            else cache.WindowSignalCache.build(
                joint,
                control_tracks,
                bound_columns=("in_start", "in_end"),
                residency="memory",
            )
        ),
        signal_groups=signal_groups,
        control_groups=control_groups,
        sampler=sampler,
    )


def placed(draw, in_window, out_window, max_jitter, peaks=PEAKS, negatives=NEGATIVES):
    """Return the contig and the input and output window starts a draw names.

    Derived from the locus rows themselves: a window is centred on its
    locus's midpoint, a peak's begins ``max_jitter`` early, and the
    draw reads ``jitter`` bases into it.
    """

    chrom, start, end = (peaks if draw.is_peak else negatives)[draw.index]
    middle = start + (end - start) // 2
    shift = draw.jitter - (max_jitter if draw.is_peak else 0)
    return chrom, middle - in_window // 2 + shift, middle - out_window // 2 + shift


def expected_signal(dense, chrom, start, width, mirrored, perm):
    """Return the signal tensor dense channels imply over one window."""

    signal = numpy.stack([channel[chrom][start : start + width] for channel in dense])
    if mirrored:
        signal = signal[perm][:, ::-1]
    return torch.from_numpy(signal.astype(numpy.float32))


def expected_sequence(residues, mirrored):
    """Return the one-hot tensor a residue string implies, mirrored or not."""

    encoded = one_hot(residues)
    if mirrored:
        encoded = encoded[::-1, ::-1]
    return torch.from_numpy(encoded.copy())


def sharded_loader(reader, batch_size, rank=0, world_size=1, num_workers=0):
    """Return a loader over one rank's share of a dataset, and its shard."""

    shard = sampling.ShardedEpochSampler(
        len(reader), batch_size, rank=rank, world_size=world_size
    )
    loader = torch.utils.data.DataLoader(
        reader,
        sampler=shard,
        batch_size=shard.local_batch_size,
        num_workers=num_workers,
        timeout=WORKER_TIMEOUT if num_workers else 0,
    )
    return loader, shard


class TestLocusDataset:
    """One draw in, one sample's tensors out."""

    def test___init___should_raise_when_windows_are_not_a_polars_frame(self, corpus):
        """Test the refusal of a window frame of another library.

        Given:
            Peak windows as a pandas frame.
        When:
            A dataset is constructed over them.
        Then:
            It should raise `TypeError` naming the dataset and the
            frame it takes.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        as_pandas = pandas.DataFrame(peak_windows.to_dict(as_series=False))

        # Act & assert
        with pytest.raises(TypeError, match="LocusDataset takes a polars DataFrame"):
            dataset.LocusDataset(
                as_pandas, negative_windows, None, None, IN_WINDOW, OUT_WINDOW
            )

    def test___init___should_raise_when_there_are_no_peaks(self, corpus):
        """Test the refusal of an empty peak set.

        Given:
            A peak window frame with no rows.
        When:
            A dataset is constructed over it.
        Then:
            It should raise `ValueError` saying there is nothing to
            draw from.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)

        # Act & assert
        with pytest.raises(ValueError, match="peak_windows is empty"):
            dataset.LocusDataset(
                peak_windows.head(0), negative_windows, None, None, IN_WINDOW, OUT_WINDOW
            )

    def test___init___should_raise_when_control_groups_have_no_control_cache(
        self, corpus
    ):
        """Test the refusal of a control grouping with nothing to permute.

        Given:
            A control grouping and no control cache.
        When:
            A dataset is constructed.
        Then:
            It should raise `ValueError` naming the grouping, rather
            than silently ignore it.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)

        # Act & assert
        with pytest.raises(ValueError, match="control_groups"):
            dataset.LocusDataset(
                peak_windows,
                negative_windows,
                None,
                None,
                IN_WINDOW,
                OUT_WINDOW,
                control_groups=[2],
            )

    def test___init___should_raise_when_the_groups_disagree_with_the_cache(self, corpus):
        """Test the refusal of a grouping of another channel count.

        Given:
            A signal cache of three channels and a grouping of two.
        When:
            A dataset is constructed.
        Then:
            It should raise `ValueError` naming the grouping and both
            counts, rather than fail on the first mirrored draw.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        joint = cache.joint_windows(peak_windows, negative_windows)
        signal = cache.WindowSignalCache.build(
            joint, corpus.tracks[:3], residency="memory"
        )

        # Act & assert
        with pytest.raises(ValueError, match="signal_groups .* 2 channels .* holds 3"):
            dataset.LocusDataset(
                peak_windows,
                negative_windows,
                stores.SequenceStore(corpus.fasta),
                signal,
                IN_WINDOW,
                OUT_WINDOW,
                signal_groups=[2],
            )

    @pytest.mark.parametrize(
        "sampler",
        [
            object(),
            [sampling.LocusDraw(True, 0, 0, False)],
            type("PlanOnly", (), {"plan": lambda self, epoch: None})(),
        ],
    )
    def test___init___should_raise_when_the_sampler_is_not_a_mix_sampler(
        self, corpus, sampler
    ):
        """Test the refusal of a sampler that is not the one a pair resolves through.

        Given:
            An object with no ``plan``, a list of draws, which a loader
            takes as a sampler and a dataset cannot, and an object with
            a ``plan`` and nothing else.
        When:
            A dataset is constructed with it.
        Then:
            It should raise `TypeError` naming the sampler type, rather
            than fail in a loader worker on the first pair.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)

        # Act & assert
        with pytest.raises(TypeError, match="through a PeakNegativeMixSampler"):
            dataset.LocusDataset(
                peak_windows,
                negative_windows,
                None,
                None,
                IN_WINDOW,
                OUT_WINDOW,
                sampler=sampler,
            )

    @pytest.mark.parametrize("extra_peaks,extra_negatives", [(2, 0), (0, 1), (-1, 0)])
    def test___init___should_raise_when_the_sampler_is_sized_for_other_loci(
        self, corpus, extra_peaks, extra_negatives
    ):
        """Test the refusal of a sampler that draws from other counts.

        Given:
            A mix sampler over more peaks, more negatives, or fewer
            peaks than the frames hold.
        When:
            A dataset is constructed with it.
        Then:
            It should raise `ValueError` naming the dataset's counts and
            the sampler's, rather than read a negative as a peak and
            label it so.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        n_peaks = len(peak_windows) + extra_peaks
        n_negatives = len(negative_windows) + extra_negatives
        sampler = sampling.PeakNegativeMixSampler(n_peaks, n_negatives)
        match = "holds {} peaks and {} negatives.*draws from {} peaks and {} negatives".format(
            len(peak_windows), len(negative_windows), n_peaks, n_negatives
        )

        # Act & assert
        with pytest.raises(ValueError, match=match):
            dataset.LocusDataset(
                peak_windows,
                negative_windows,
                None,
                None,
                IN_WINDOW,
                OUT_WINDOW,
                sampler=sampler,
            )

    def test___init___should_keep_the_groupings_it_was_given(self, controlled):
        """Test that a dataset reports the groupings it mirrors by.

        Given:
            A dataset built with signals grouped one and two and one
            control pair.
        When:
            Its groupings and permutations are read.
        Then:
            It should report the two groupings as given, and the
            permutation each implies.
        """

        # Act
        reported = (controlled.signal_groups, controlled.control_groups)

        # Assert
        assert reported == ((1, 2), (2,))
        assert controlled.signal_perm.tolist() == [0, 2, 1]
        assert controlled.control_perm.tolist() == [1, 0]

    def test___init___should_accept_an_empty_negative_set(self, corpus):
        """Test a peak-only dataset.

        Given:
            Peak windows and a negative frame with no rows.
        When:
            A dataset is constructed and a peak read.
        Then:
            It should count no negatives and read the peak as a
            dataset with negatives reads it.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        signal = cache.WindowSignalCache.build(
            peak_windows, corpus.tracks[:1], residency="memory"
        )
        draw = sampling.LocusDraw(True, 2, 3, False)

        # Act
        reader = dataset.LocusDataset(
            peak_windows,
            negative_windows.head(0),
            stores.SequenceStore(corpus.fasta),
            signal,
            IN_WINDOW,
            OUT_WINDOW,
        )

        # Assert
        assert (reader.n_peaks, reader.n_negatives) == (len(PEAKS), 0)
        assert len(reader) == len(PEAKS)
        chrom, in_start, _ = placed(draw, IN_WINDOW, OUT_WINDOW, MAX_JITTER)
        residues = parse_fasta(corpus.fasta)[chrom][in_start : in_start + IN_WINDOW]
        assert torch.equal(reader[draw][0], expected_sequence(residues, False))

    @pytest.mark.parametrize(
        "sampler",
        [
            object(),
            [sampling.LocusDraw(True, 0, 0, False)],
            type("PlanOnly", (), {"plan": lambda self, epoch: None})(),
        ],
    )
    def test_sampler_should_raise_when_assigned_a_sampler_that_is_not_a_mix_sampler(
        self, corpus, sampler
    ):
        """Test the refusal of a sampler assigned after construction.

        Given:
            A dataset built without a sampler, and an object with no
            ``plan``, a list of draws, or an object with a ``plan`` and
            nothing else.
        When:
            The object is assigned as the dataset's sampler.
        Then:
            It should raise `TypeError` naming the sampler type, rather
            than fail in a loader worker on the first pair.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        reader = dataset.LocusDataset(
            peak_windows, negative_windows, None, None, IN_WINDOW, OUT_WINDOW
        )

        # Act & assert
        with pytest.raises(TypeError, match="through a PeakNegativeMixSampler"):
            reader.sampler = sampler

    @pytest.mark.parametrize("extra_peaks,extra_negatives", [(2, 0), (0, 1), (-1, 0)])
    def test_sampler_should_raise_when_assigned_a_sampler_sized_for_other_loci(
        self, corpus, extra_peaks, extra_negatives
    ):
        """Test the refusal of a mis-sized sampler assigned after construction.

        Given:
            A dataset built without a sampler, and a mix sampler over
            more peaks, more negatives, or fewer peaks than its frames
            hold.
        When:
            The sampler is assigned as the dataset's.
        Then:
            It should raise `ValueError` naming both pairs of counts,
            as the constructor does.
        """

        # Arrange
        peak_windows, negative_windows = kept_windows(corpus)
        reader = dataset.LocusDataset(
            peak_windows, negative_windows, None, None, IN_WINDOW, OUT_WINDOW
        )
        n_peaks = len(peak_windows) + extra_peaks
        n_negatives = len(negative_windows) + extra_negatives
        sampler = sampling.PeakNegativeMixSampler(n_peaks, n_negatives)
        match = "holds {} peaks and {} negatives.*draws from {} peaks and {} negatives".format(
            len(peak_windows), len(negative_windows), n_peaks, n_negatives
        )

        # Act & assert
        with pytest.raises(ValueError, match=match):
            reader.sampler = sampler

    def test___len___should_count_draws_with_a_sampler_and_loci_without(self, corpus):
        """Test what the length means with and without a sampler.

        Given:
            A dataset, and a sampler over its loci that draws negatives.
        When:
            Its length is taken before and after the sampler is
            attached.
        Then:
            It should count the addressable loci without one and the
            draws of an epoch with one, which is more than the peaks.
        """

        # Arrange
        reader = build_dataset(corpus, signal_groups=[1])
        sampler = sampling.PeakNegativeMixSampler(
            reader.n_peaks, reader.n_negatives, negative_ratio=0.5
        )

        # Act
        without = len(reader)
        reader.sampler = sampler
        attached = len(reader)

        # Assert
        assert (reader.n_peaks, reader.n_negatives) == (len(PEAKS), len(NEGATIVES))
        assert without == len(PEAKS) + len(NEGATIVES)
        assert attached == len(sampler)
        assert attached > reader.n_peaks

    @pytest.mark.parametrize("index", [0, numpy.int64(3)])
    def test___getitem___should_raise_when_indexed_by_an_integer(self, plain, index):
        """Test the refusal of a bare position.

        Given:
            A dataset and an integer index, Python's or numpy's.
        When:
            The dataset is indexed by it.
        Then:
            It should raise `TypeError` naming what it is indexed by,
            because an integer says neither which set nor which epoch.
        """

        # Act & assert
        with pytest.raises(TypeError, match="LocusDraw"):
            plain[index]

    def test___getitem___should_raise_when_a_pair_has_no_sampler(self, plain):
        """Test the refusal of a pair that cannot be resolved.

        Given:
            A dataset with no sampler attached.
        When:
            It is indexed by an ``(epoch, index)`` pair.
        Then:
            It should raise `TypeError` saying what is missing.
        """

        # Act & assert
        with pytest.raises(TypeError, match="no sampler to resolve it"):
            plain[(0, 0)]

    def test___getitem___should_answer_int8_sequence_and_float32_signal(self, plain):
        """Test the shapes and dtypes of one read.

        Given:
            A dataset over three signal channels.
        When:
            A peak draw is read.
        Then:
            The sequence should be ``int8`` of shape ``(4, in_window)``,
            the signal ``float32`` of shape ``(3, out_window)``, and the
            label a plain ``1``.
        """

        # Act
        sequence, signal, label = plain[sampling.LocusDraw(True, 0, 3, False)]

        # Assert
        assert sequence.shape == (4, IN_WINDOW)
        assert sequence.dtype == torch.int8
        assert signal.shape == (3, OUT_WINDOW)
        assert signal.dtype == torch.float32
        assert label == 1
        assert type(label) is int

    @settings(deadline=None, max_examples=60)
    @given(draw=corpus_draws)
    def test___getitem___should_read_what_the_reference_and_the_tracks_hold(
        self, corpus, plain, draw
    ):
        """Test a read against oracles built from the files themselves.

        Given:
            Any peak draw at any jitter inside its margin, or any
            negative draw, in either orientation.
        When:
            It is read.
        Then:
            The sequence should be the one-hot of the reference's own
            residues over the draw's input window, upper-cased and with
            ``N`` as zeros, and the signal the tracks' rows over its
            output window, both turned around and the stranded pair
            swapped when the draw is mirrored; the label should say
            which set it came from.
        """

        # Arrange
        chrom, in_start, out_start = placed(draw, IN_WINDOW, OUT_WINDOW, MAX_JITTER)
        residues = parse_fasta(corpus.fasta)[chrom][in_start : in_start + IN_WINDOW]
        channels = [dense(rows, CHROM_SIZES, numpy.float64) for rows in SIGNAL_ROWS[:3]]

        # Act
        sequence, signal, label = plain[draw]

        # Assert
        assert torch.equal(
            sequence, expected_sequence(residues, draw.reverse_complement)
        )
        assert torch.equal(
            signal,
            expected_signal(
                channels,
                chrom,
                out_start,
                OUT_WINDOW,
                draw.reverse_complement,
                [0, 2, 1],
            ),
        )
        assert label == int(draw.is_peak)

    def test___getitem___should_match_a_pysam_oracle(self, piled, pysam):
        """Test a read against alignments counted with pysam.

        Given:
            Two alignments, the track a pileup of each wrote, and a
            dataset reading the two as one stranded pair.
        When:
            Peak draws at several jitters, a mirrored one and a
            negative draw are read.
        Then:
            Every sequence should be ``int8`` of shape ``(4,
            in_window)`` and equal the one-hot of what pysam fetches
            from the reference, and every signal ``float32`` of shape
            ``(2, out_window)`` and equal the shifted 5' ends pysam
            counts in each alignment, with the pair swapped and both
            turned around for the mirrored draw.
        """

        # Arrange
        reader = build_dataset(piled, signal_groups=[2], signal_tracks=piled.tracks)
        dense = [
            oracle_pileup([bam], piled.chrom_sizes, pysam, ATAC) for bam in piled.bams
        ]
        peaks = builders.intervals("chr1", 100, 900, 90, 20)
        negatives = builders.intervals("chr2", 100, 400, 70, 20)
        requested = [
            sampling.LocusDraw(True, 0, 0, False),
            sampling.LocusDraw(True, 3, 9, False),
            sampling.LocusDraw(True, 7, 4, True),
            sampling.LocusDraw(False, 2, 0, False),
            sampling.LocusDraw(False, 4, 0, True),
        ]

        # Act
        samples = [reader[draw] for draw in requested]

        # Assert
        assert sum(float(signal.sum()) for _, signal, _ in samples) > 0
        with pysam.FastaFile(piled.fasta) as reference:
            for draw, (sequence, signal, label) in zip(requested, samples):
                chrom, in_start, out_start = placed(draw, 60, 30, 5, peaks, negatives)
                residues = reference.fetch(chrom, in_start, in_start + 60)
                assert sequence.shape == (4, 60)
                assert sequence.dtype == torch.int8
                assert torch.equal(
                    sequence, expected_sequence(residues, draw.reverse_complement)
                )
                assert signal.shape == (2, 30)
                assert signal.dtype == torch.float32
                assert torch.equal(
                    signal,
                    expected_signal(
                        dense, chrom, out_start, 30, draw.reverse_complement, [1, 0]
                    ),
                )
                assert label == int(draw.is_peak)

    def test___getitem___should_shift_by_one_base_per_unit_of_jitter(self, plain):
        """Test what one unit of jitter moves.

        Given:
            Two draws on one peak differing by one in jitter.
        When:
            Both are read.
        Then:
            The second should be the first shifted by one base, in the
            sequence and in the signal.
        """

        # Act
        first = plain[sampling.LocusDraw(True, 2, 0, False)]
        second = plain[sampling.LocusDraw(True, 2, 1, False)]

        # Assert
        assert torch.equal(first[0][:, 1:], second[0][:, :-1])
        assert torch.equal(first[1][:, 1:], second[1][:, :-1])
        assert not torch.equal(first[0], second[0])

    def test___getitem___should_flip_the_sequence_and_permute_the_signal_when_mirrored(
        self, plain
    ):
        """Test a mirrored draw against the unmirrored one.

        Given:
            One draw read with and without reverse-complement, over
            signals grouped one and two.
        When:
            Both are read.
        Then:
            The sequence should be flipped on both axes, and the signal
            flipped along its length with only the stranded pair
            swapped.
        """

        # Act
        forward = plain[sampling.LocusDraw(True, 3, 2, False)]
        mirrored = plain[sampling.LocusDraw(True, 3, 2, True)]

        # Assert
        assert torch.equal(mirrored[0], torch.flip(forward[0], [0, 1]))
        assert torch.equal(mirrored[1], forward[1][[0, 2, 1]].flip(-1))
        assert not torch.equal(forward[1][1], forward[1][2])

    def test___getitem___should_read_the_negative_set_when_the_draw_is_a_negative(
        self, corpus, plain
    ):
        """Test that a negative draw reads a negative locus.

        Given:
            A peak draw and a negative draw of the same row.
        When:
            Both are read.
        Then:
            The labels should be one and zero, and the negative's
            sequence should be the reference over the negative locus's
            window rather than the peak's.
        """

        # Arrange
        draw = sampling.LocusDraw(False, 0, 0, False)
        chrom, in_start, _ = placed(draw, IN_WINDOW, OUT_WINDOW, MAX_JITTER)
        residues = parse_fasta(corpus.fasta)[chrom][in_start : in_start + IN_WINDOW]

        # Act
        peak = plain[sampling.LocusDraw(True, 0, 0, False)]
        negative = plain[draw]

        # Assert
        assert (peak[2], negative[2]) == (1, 0)
        assert torch.equal(negative[0], expected_sequence(residues, False))
        assert not torch.equal(peak[0], negative[0])

    def test___getitem___should_read_controls_over_the_input_window(
        self, corpus, controlled
    ):
        """Test the four-element read a control cache adds.

        Given:
            A dataset with a two-channel control cache.
        When:
            A draw is read with and without reverse-complement.
        Then:
            It should answer sequence, control, signal and label, the
            control being the control tracks' rows over the input
            window at ``float32``, and the mirrored control the pair
            swapped and turned around.
        """

        # Arrange
        draw = sampling.LocusDraw(True, 1, 5, False)
        chrom, in_start, _ = placed(draw, IN_WINDOW, OUT_WINDOW, MAX_JITTER)
        channels = [dense(rows, CHROM_SIZES, numpy.float64) for rows in CONTROL_ROWS[:2]]

        # Act
        sequence, control, signal, label = controlled[draw]
        mirrored = controlled[draw._replace(reverse_complement=True)]

        # Assert
        assert sequence.shape == (4, IN_WINDOW)
        assert signal.shape == (3, OUT_WINDOW)
        assert label == 1
        assert control.dtype == torch.float32
        assert torch.equal(
            control, expected_signal(channels, chrom, in_start, IN_WINDOW, False, None)
        )
        assert torch.equal(mirrored[1], control[[1, 0]].flip(-1))

    @pytest.mark.parametrize(
        "signal_groups,control_groups",
        [
            ([2], [1]),
            ([2], [2]),
            ([1, 2], [2]),
            ([1, 2], [1]),
            ([1, 2], [1, 1, 2]),
            ([2, 2], [2]),
        ],
    )
    def test___getitem___should_permute_signal_and_control_by_their_own_groups(
        self, corpus, signal_groups, control_groups
    ):
        """Test that the two groupings never consult each other.

        Given:
            Any pairing of a signal group layout with a control group
            layout, including one nothing like the other.
        When:
            A draw is read with and without reverse-complement.
        Then:
            Each tensor should be permuted by its own grouping and
            then flipped, and the sequence flipped on both axes.
        """

        # Arrange
        reader = build_dataset(
            corpus, signal_groups=signal_groups, control_groups=control_groups
        )
        signal_perm = groups.channel_permutation_from_groups(signal_groups)
        control_perm = groups.channel_permutation_from_groups(control_groups)

        # Act
        forward = reader[sampling.LocusDraw(True, 2, 1, False)]
        mirrored = reader[sampling.LocusDraw(True, 2, 1, True)]

        # Assert
        assert torch.equal(mirrored[2], forward[2][signal_perm].flip(-1))
        assert torch.equal(mirrored[1], forward[1][control_perm].flip(-1))
        assert torch.equal(mirrored[0], torch.flip(forward[0], [0, 1]))
        assert torch.equal(reader.signal_perm, signal_perm)
        assert torch.equal(reader.control_perm, control_perm)

    def test___getitem___should_resolve_a_pair_to_the_draw_it_names(self, corpus):
        """Test the ``(epoch, index)`` protocol against the draw protocol.

        Given:
            A dataset that holds its sampler, whose own epoch has been
            set to one that is not read.
        When:
            It is indexed by an ``(epoch, index)`` pair and by the draw
            the sampler plans for that slot.
        Then:
            Both should return the same tensors and label, at the
            first slot, a middle one and the last, in two epochs.
        """

        # Arrange
        reader = build_dataset(corpus, signal_groups=[1, 2])
        sampler = sampling.PeakNegativeMixSampler(
            reader.n_peaks,
            reader.n_negatives,
            negative_ratio=0.5,
            max_jitter=MAX_JITTER,
            reverse_complement=True,
            random_state=0,
        )
        reader.sampler = sampler
        sampler.set_epoch(7)

        # Act & assert
        for epoch in (0, 3):
            plan = sampler.plan(epoch)
            for index in (0, 1, len(plan) - 1):
                by_pair = reader[(epoch, index)]
                by_draw = reader[plan[index]]
                assert torch.equal(by_pair[0], by_draw[0])
                assert torch.equal(by_pair[1], by_draw[1])
                assert by_pair[2] == by_draw[2]

    def test_pickling_should_restore_a_dataset_that_reads_the_same(self, controlled):
        """Test a pickle round trip of a dataset whose stores are open.

        Given:
            A dataset that has already read a draw, so its reference
            handle is open.
        When:
            It is pickled and restored.
        Then:
            The copy should read the same draw to the same tensors.
        """

        # Arrange
        draw = sampling.LocusDraw(True, 4, 5, True)
        before = controlled[draw]

        # Act
        restored = pickle.loads(pickle.dumps(controlled))
        after = restored[draw]

        # Assert
        assert len(after) == 4
        for expected, actual in zip(before[:3], after[:3]):
            assert torch.equal(expected, actual)
        assert before[3] == after[3]

    @pytest.mark.workers
    def test_pickling_should_carry_the_dataset_into_a_spawned_worker(self, controlled):
        """Test that a dataset survives pickling into a worker process.

        Given:
            A dataset that has already read a draw, and a loader that
            starts its worker by ``spawn``, so the worker's copy can
            only have arrived by pickle.
        When:
            Draws are read through the worker and in this process.
        Then:
            The worker's batches should equal this process's.
        """

        # Arrange
        requested = [
            sampling.LocusDraw(True, 1, 3, False),
            sampling.LocusDraw(True, 6, 0, True),
            sampling.LocusDraw(False, 2, 0, False),
            sampling.LocusDraw(False, 5, 0, True),
        ]
        controlled[requested[0]]
        local = torch.utils.data.DataLoader(controlled, sampler=requested, batch_size=2)
        worker = torch.utils.data.DataLoader(
            controlled,
            sampler=requested,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
            timeout=WORKER_TIMEOUT,
        )

        # Act
        expected = drain(local, n_epochs=1)
        through_worker = drain(worker, n_epochs=1)

        # Assert
        assert len(expected) == 2
        assert same_batches(through_worker, expected)

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test___getitem___should_let_the_ranks_together_read_one_ranks_batches(
        self, corpus, world_size
    ):
        """Test sharded reads across ranks, tensor for tensor.

        Given:
            A dataset holding its sampler, and one sharded loader per
            rank of a world of one, two or four.
        When:
            Two epochs are read on every rank and the ranks' batches
            are concatenated step by step.
        Then:
            They should equal the batches a single rank reads, which
            are in turn the sampler's own plan read slot by slot.
        """

        # Arrange
        reader = build_dataset(corpus, signal_groups=[1, 2])
        sampler = sampling.PeakNegativeMixSampler(
            reader.n_peaks,
            reader.n_negatives,
            negative_ratio=0.5,
            max_jitter=MAX_JITTER,
            reverse_complement=True,
            random_state=4,
        )
        reader.sampler = sampler
        batch_size = 8
        single, single_shard = sharded_loader(reader, batch_size)
        ranks = [
            sharded_loader(reader, batch_size, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]

        # Act & assert
        for epoch in (0, 1):
            single_shard.set_epoch(epoch)
            expected = drain(single, n_epochs=1)
            per_rank = []
            for loader, shard in ranks:
                shard.set_epoch(epoch)
                per_rank.append(drain(loader, n_epochs=1))
            combined = [
                tuple(
                    torch.cat(parts) for parts in zip(*(rows[step] for rows in per_rank))
                )
                for step in range(len(expected))
            ]
            assert len(expected) == len(sampler) // batch_size
            assert same_batches(combined, expected)
            plan = sampler.plan(epoch)
            first = torch.stack([reader[plan[slot]][0] for slot in range(batch_size)])
            assert torch.equal(expected[0][0], first)

    @pytest.mark.workers
    def test___getitem___should_answer_the_same_batches_from_one_worker_and_many(
        self, corpus
    ):
        """Test that the batch sequence does not depend on the worker count.

        Given:
            A dataset holding its sampler, read through a sharded
            sampler in this process, by one worker and by two.
        When:
            Two epochs are drained from each.
        Then:
            All three streams should be equal batch for batch, and the
            two epochs should differ, so equality is not three replays
            of one epoch.
        """

        # Arrange
        reader = build_dataset(corpus, signal_groups=[1, 2])
        reader.sampler = sampling.PeakNegativeMixSampler(
            reader.n_peaks,
            reader.n_negatives,
            negative_ratio=0.5,
            max_jitter=MAX_JITTER,
            reverse_complement=True,
            random_state=9,
        )

        # Act
        streams = []
        for num_workers in (0, 1, 2):
            loader, shard = sharded_loader(reader, 8, num_workers=num_workers)
            stream = []
            for epoch in (0, 1):
                shard.set_epoch(epoch)
                stream.append(drain(loader, n_epochs=1))
            streams.append(stream)

        # Assert
        in_process, one_worker, two_workers = streams
        assert not same_batches(in_process[0], in_process[1])
        for epoch in (0, 1):
            assert same_batches(one_worker[epoch], in_process[epoch])
            assert same_batches(two_workers[epoch], one_worker[epoch])
