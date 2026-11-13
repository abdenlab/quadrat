"""Tests for the training loader and for fetching in bulk.

`build_training_loader` wires every other component together, so what
is pinned here is what only the wiring can get wrong: the filters that
decide how many loci there are, which size every draw of every epoch;
the refusals of inputs that would otherwise train on nothing; that a
persisted entry answers the batches a cold build answers; and that the
emitted stream does not depend on how many workers draw it or on
whether they survive between epochs.

The golden batch digests are literal values carried over with the plan
digests in ``tests/test_sampling.py``. They hash the tensors a loader
emits over a corpus built from index arithmetic alone, and are never
to be regenerated to make a test pass.

`RechunkedLoader`, `auto_fetch_size` and `sample_nbytes` are pinned on
the two contracts that make fetching in bulk safe: only the number of
transfers changes, and a fetch stays under the byte ceiling however
large the epoch, wherever one batch is within it, as `auto_fetch_size`
states. Sharding is pinned through the entry point, against
the unsharded stream.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import copy
import hashlib
import math
import os
import sys
import weakref
from types import MappingProxyType

import numpy
import polars
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from quadrat import cache, dataset, groups, loader, loci, sampling, stores
from tests import (
    CONFIG_IDS,
    GOLDEN_CONFIGS,
    WORKER_TIMEOUT,
    builders,
    dense,
    drain,
    one_hot,
    parse_fasta,
    same_batches,
)

CHROM_SIZES = MappingProxyType({"chr1": 6000, "chr2": 4000})


IN_WINDOW, OUT_WINDOW, MAX_JITTER = 80, 40, 6


BATCH_SIZE = 4


PEAKS = builders.intervals("chr1", 400, 3400, 75, 20) + builders.intervals(
    "chr2", 400, 2400, 80, 20
)


NEGATIVES = builders.intervals("chr1", 3600, 5600, 85, 20) + builders.intervals(
    "chr2", 2600, 3800, 90, 20
)


# A region covering every negative and no peak, on the 100-base grid
# an exclusion is tested on.
NEGATIVE_REGIONS = [("chr1", 3500, 6000), ("chr2", 2500, 4000)]


# One quiet channel with a short region fifty times the rest, and a
# loud pair at two hundred times the ordinary level without one. The
# region lies inside the output window of one peak and of no other
# locus, so grouped one and two, that peak is extreme in its own group
# and unremarkable in the total over all three.
SIGNAL_ROWS = [
    [
        (chrom, pos, value * 50 if chrom == "chr1" and 990 < pos < 1030 else value)
        for chrom, pos, value in builders.scattered_rows(CHROM_SIZES, 30)
    ],
    *(
        [
            (chrom, pos, value * 200)
            for chrom, pos, value in builders.scattered_rows(CHROM_SIZES, seed)
        ]
        for seed in (31, 32)
    ),
]


CONTROL_ROWS = [builders.scattered_rows(CHROM_SIZES, seed) for seed in (40, 41)]


# The batch digests, recorded over `GOLDEN_CONFIGS`, two epochs each.
# Carried as literals: a value that stops matching means a batch moved,
# and the fix is in whatever moved it.
GOLDEN_BATCHES = MappingProxyType(
    {
        "baseline": "a257008ecdcd2ae7b95bc2fce12fe83da71dca2000e0c77a30276a5fc9fa8a70",
        "grouped_ctl": "08b7b8e3eac510a736f4eaae507c15bf1f8a84ff291a7f457830ca82af57cdb6",
        "no_jitter": "d4e0507f0bdeba5ffadc3116c11b3ce5db55cef5562945d2dd6b6b6b1420046a",
        "no_negatives": "72fd5e315abda15d61715e5fce10cd4077d07b43239ec46329e4981d6c229d53",
        "no_rc": "eaeabed7f774046b7a351918fe94ec6de2b9d9b801cbd155982e327be7ecfe6b",
        "sequential": "33341e2fe8fff775740b5ca82125b15e0caf05d6a0544af2fdb600959832323e",
    }
)


GOLDEN_SIZES = MappingProxyType({"chr1": 3000, "chr2": 2000})


GOLDEN_IN_WINDOW, GOLDEN_OUT_WINDOW = 60, 30


# Per-group totals of up to forty loci in up to four groups, the
# quantile a ceiling is taken from, and the scale applied to it.
outlier_arguments = st.fixed_dictionaries(
    {
        "group_counts": hnp.arrays(
            numpy.float32,
            st.tuples(
                st.integers(min_value=0, max_value=40),
                st.integers(min_value=1, max_value=4),
            ),
            elements=st.floats(min_value=0, max_value=1e6, width=32),
        ),
        "quantile": st.floats(min_value=0, max_value=1),
        "scale": st.floats(min_value=0.25, max_value=4),
    }
)


# Shapes of run a fetch size is chosen for.
fetch_arguments = st.fixed_dictionaries(
    {
        "n_draws": st.integers(min_value=0, max_value=500_000),
        "batch_size": st.integers(min_value=1, max_value=512),
        "num_workers": st.integers(min_value=0, max_value=32),
        "bytes_per_sample": st.integers(min_value=1, max_value=10**8),
        "prefetch_factor": st.integers(min_value=1, max_value=8),
    }
)


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Build the reference, locus files, three signal tracks and two controls."""

    pytest.importorskip("polars_bio")
    return corpus_factory(
        "loader",
        CHROM_SIZES,
        peaks=PEAKS,
        negatives=NEGATIVES,
        exclusion=NEGATIVE_REGIONS,
        tracks=SIGNAL_ROWS + CONTROL_ROWS,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        max_jitter=MAX_JITTER,
    )


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    """Build the corpus the golden batch digests were recorded over.

    Every byte comes from index arithmetic: no random generator is in
    the chain, so no release of anything can move a digest for a
    reason unrelated to the code under test.
    """

    pytest.importorskip("polars_bio")
    root = tmp_path_factory.mktemp("loader-golden")
    fasta = str(root / "genome.fa")
    with open(fasta, "w") as handle:
        for chrom, size in GOLDEN_SIZES.items():
            residues = "".join(
                "ACGT"[(index * 7 + len(chrom) * 3 + 1) % 4] for index in range(size)
            )
            handle.write(">{}\n".format(chrom))
            for offset in range(0, size, 60):
                handle.write(residues[offset : offset + 60] + "\n")

    tracks = [
        builders.write_track(
            str(root / "ch{}.track.parquet".format(channel)),
            [
                (chrom, pos, float(pos % 7 + 1))
                for chrom, size in GOLDEN_SIZES.items()
                for pos in range(channel + 1, size, 3 + channel)
            ],
        )
        for channel in range(2)
    ]
    peaks = builders.write_bed(
        str(root / "peaks.bed"),
        builders.intervals("chr1", 300, 1500, 60, 20)
        + builders.intervals("chr2", 300, 900, 70, 20),
    )
    negatives = builders.write_bed(
        str(root / "negatives.bed"),
        builders.intervals("chr1", 1700, 2600, 75, 20)
        + builders.intervals("chr2", 1100, 1700, 80, 20),
    )
    return {
        "peaks": peaks,
        "negatives": negatives,
        "sequences": fasta,
        "signals": tracks,
        "in_window": GOLDEN_IN_WINDOW,
        "out_window": GOLDEN_OUT_WINDOW,
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": False,
    }


def arguments(corpus, **overrides):
    """Return the loader arguments most tests share, with ``overrides`` applied.

    Three signal channels grouped one and two, no controls, no
    workers, and a seed.
    """

    return dict(
        {
            "peaks": corpus.peaks,
            "negatives": corpus.negatives,
            "sequences": corpus.fasta,
            "signals": [corpus.tracks[0], [corpus.tracks[1], corpus.tracks[2]]],
            "in_window": IN_WINDOW,
            "out_window": OUT_WINDOW,
            "max_jitter": MAX_JITTER,
            "negative_ratio": 0.5,
            "reverse_complement": True,
            "shuffle": True,
            "random_state": 0,
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "pin_memory": False,
        },
        **overrides,
    )


def golden_arguments(golden, config_id):
    """Return the loader arguments of one golden configuration."""

    config = GOLDEN_CONFIGS[config_id]
    return dict(
        golden,
        max_jitter=config["max_jitter"],
        negative_ratio=config["negative_ratio"],
        reverse_complement=config["reverse_complement"],
        shuffle=config["shuffle"],
        random_state=config["random_state"],
    )


def batch_digest(batches):
    """Return the sha256 of a three-tensor batch stream: shapes, dtypes and bytes."""

    digest = hashlib.sha256()
    for batch in batches:
        for name, tensor in zip((b"sequence", b"signal", b"label"), batch):
            digest.update(name)
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def with_workers(built, num_workers, persistent):
    """Return a loader over a built loader's dataset and sampler, under workers."""

    return torch.utils.data.DataLoader(
        built.dataset,
        sampler=built.sampler,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        persistent_workers=persistent,
        timeout=WORKER_TIMEOUT,
    )


def one_peak_arguments(tmp_path, residues):
    """Return loader arguments over one peak whose input window holds ``residues``.

    The reference is ``ACGT`` repeated, with ``residues`` written from
    position 300, inside the input window of the one peak; the loader
    neither shuffles, jitters nor mirrors, and persists nothing.
    """

    sequence = list("ACGT" * 150)
    sequence[300 : 300 + len(residues)] = residues
    fasta = str(tmp_path / "genome.fa")
    with open(fasta, "w") as handle:
        handle.write(">chr1\n{}\n".format("".join(sequence)))
    return {
        "peaks": builders.write_bed(str(tmp_path / "peaks.bed"), [("chr1", 290, 310)]),
        "negatives": None,
        "sequences": fasta,
        "signals": [
            builders.write_track(str(tmp_path / "t.track.parquet"), [("chr1", 300, 1.0)])
        ],
        "in_window": IN_WINDOW,
        "out_window": OUT_WINDOW,
        "max_jitter": 0,
        "negative_ratio": 0.0,
        "reverse_complement": False,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
        "cache_dir": False,
    }


def window_totals(rows_per_channel, loci, jitter):
    """Return each locus's total per channel over its output window.

    Derived from the locus rows and the track rows directly: the
    output window is centred on the locus's midpoint and ``jitter``
    wider on both sides.
    """

    totals = numpy.zeros((len(loci), len(rows_per_channel)))
    for channel, rows in enumerate(rows_per_channel):
        arrays = dense(rows, CHROM_SIZES, numpy.float64)
        for index, (chrom, start, end) in enumerate(loci):
            middle = start + (end - start) // 2
            lo = middle - OUT_WINDOW // 2 - jitter
            hi = middle + OUT_WINDOW // 2 + jitter
            totals[index, channel] = arrays[chrom][lo:hi].sum()
    return totals


def direct_dataset(corpus, signals, controls=None):
    """Return a dataset over a corpus's readable loci, built without the loader.

    One track per channel in each of ``signals`` and ``controls``, each
    channel a group of its own, and no filter but the one
    `quadrat.loci.readable_windows` makes.
    """

    frames = []
    for loci_file, jitter in ((corpus.peaks, MAX_JITTER), (corpus.negatives, 0)):
        windows, keep = loci.readable_windows(
            loci_file, corpus.fasta, IN_WINDOW, OUT_WINDOW, max_jitter=jitter
        )
        frames.append(windows.filter(keep))
    joint = cache.joint_windows(*frames)
    return dataset.LocusDataset(
        *frames,
        stores.SequenceStore(corpus.fasta),
        cache.WindowSignalCache.build(joint, signals, residency="memory"),
        IN_WINDOW,
        OUT_WINDOW,
        control_cache=(
            None
            if controls is None
            else cache.WindowSignalCache.build(
                joint,
                controls,
                bound_columns=("in_start", "in_end"),
                residency="memory",
            )
        ),
    )


def recording_loader(n_samples, fetch_size, pin=False):
    """Return a loader over two tensors of ``n_samples``, and the fetches it makes.

    The second value is a list the loader appends every collated fetch
    to, so a test can compare a sub-batch with the fetch it was cut
    from. ``pin`` pins every tensor of every fetch.
    """

    data = torch.utils.data.TensorDataset(
        torch.arange(n_samples * 3, dtype=torch.float32).reshape(n_samples, 3),
        torch.arange(n_samples, dtype=torch.int64),
    )
    fetches = []

    def collate(samples):
        fetched = torch.utils.data.default_collate(samples)
        if pin:
            fetched = [tensor.pin_memory() for tensor in fetched]
        fetches.append(fetched)
        return fetched

    inner = torch.utils.data.DataLoader(data, batch_size=fetch_size, collate_fn=collate)
    return inner, fetches


def test_outlier_mask_should_flag_a_locus_extreme_in_one_group_only():
    """Test that groups are judged apart and the flags OR-ed.

    Given:
        A hundred loci equal in two groups, one of them extreme in the
        second group alone, where the second group's counts are a
        thousand times the first's.
    When:
        The mask is taken.
    Then:
        It should flag that locus and no other.
    """

    # Arrange
    counts = torch.ones(100, 2)
    counts[:, 1] *= 1000.0
    counts[7, 1] = 10_000_000.0

    # Act
    mask = loader.outlier_mask(counts)

    # Assert
    assert mask.dtype == torch.bool
    assert mask.shape == (100,)
    assert mask.nonzero().flatten().tolist() == [7]


def test_outlier_mask_should_flag_a_locus_a_pooled_ceiling_would_miss():
    """Test that a quiet group keeps its own ceiling.

    Given:
        A locus extreme in a quiet group whose whole range sits below
        the ordinary level of a loud group beside it.
    When:
        The mask is taken.
    Then:
        It should flag the locus, which one ceiling over the two
        groups' total would not.
    """

    # Arrange
    counts = torch.ones(200, 2)
    counts[:, 1] = 100_000.0
    counts[11, 0] = 50.0

    # Act
    mask = loader.outlier_mask(counts)

    # Assert
    assert mask.nonzero().flatten().tolist() == [11]
    assert not loader.outlier_mask(counts.sum(dim=1, keepdim=True)).any()


def test_outlier_mask_should_keep_a_locus_exactly_at_the_ceiling():
    """Test that the comparison is strict.

    Given:
        Counts whose largest value is exactly the ceiling, the
        quantile being the maximum and the scale one.
    When:
        The mask is taken.
    Then:
        It should flag nothing, and flag the largest once the scale
        drops below one.
    """

    # Arrange
    counts = torch.tensor([[1.0], [2.0], [4.0]])

    # Act
    at_ceiling = loader.outlier_mask(counts, quantile=1.0, scale=1.0)
    below = loader.outlier_mask(counts, quantile=1.0, scale=0.75)

    # Assert
    assert at_ceiling.tolist() == [False, False, False]
    assert below.tolist() == [False, False, True]


def test_outlier_mask_should_return_an_empty_mask_when_there_are_no_loci():
    """Test the mask of an empty locus set.

    Given:
        Counts with no rows.
    When:
        The mask is taken.
    Then:
        It should be an empty boolean tensor rather than an error from
        the quantile.
    """

    # Act
    mask = loader.outlier_mask(torch.zeros(0, 2))

    # Assert
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


@settings(max_examples=100)
@given(arguments=outlier_arguments)
def test_outlier_mask_should_be_the_or_of_each_group_judged_alone(arguments):
    """Test the per-group ceilings over arbitrary counts.

    Given:
        Any per-group totals of up to forty loci in up to four groups,
        any quantile and any scale.
    When:
        The mask is taken over every group, and over each group alone.
    Then:
        It should be a boolean mask of one entry per locus, true
        exactly where some group alone flags the locus.
    """

    # Arrange
    counts = torch.from_numpy(arguments.pop("group_counts"))

    # Act
    mask = loader.outlier_mask(counts, **arguments)

    # Assert
    alone = [
        loader.outlier_mask(counts[:, [group]], **arguments)
        for group in range(counts.shape[1])
    ]
    assert mask.dtype == torch.bool
    assert mask.shape == (counts.shape[0],)
    assert torch.equal(mask, torch.stack(alone).any(dim=0))


@pytest.mark.parametrize("with_controls", [False, True])
def test_sample_nbytes_should_count_the_bytes_of_a_collated_sample(
    corpus, with_controls
):
    """Test the size taken from shapes against a sample actually read.

    Given:
        A dataset over three signal channels, without controls and with
        a control pair.
    When:
        The size of one sample is taken from the dataset, and one
        sample is read and collated alone.
    Then:
        The two should agree to the byte.
    """

    # Arrange
    reader = direct_dataset(
        corpus,
        corpus.tracks[:3],
        controls=corpus.tracks[3:5] if with_controls else None,
    )

    # Act
    counted = loader.sample_nbytes(reader)
    first = torch.utils.data.default_collate(
        [reader[sampling.LocusDraw(True, 0, 0, False)]]
    )

    # Assert
    assert counted == sum(tensor.nbytes for tensor in first)


@pytest.mark.parametrize(
    "n_draws,num_workers,bytes_per_sample",
    [(104748, 0, 1), (0, 8, 1), (500, 8, 1), (104748, 8, 10**7)],
    ids=["no_workers", "no_draws", "too_few_batches", "one_batch_at_the_ceiling"],
)
def test_auto_fetch_size_should_return_the_batch_when_bulk_cannot_help(
    n_draws, num_workers, bytes_per_sample
):
    """Test the fetch size of a run that fetching in bulk cannot help.

    Given:
        A run of batches of sixty-four with no workers, with nothing to
        draw, with too few batches to divide among the prefetch slots,
        or with samples so large that one batch already reaches the
        byte ceiling.
    When:
        A fetch size is chosen.
    Then:
        It should be the batch size.
    """

    # Act
    chosen = loader.auto_fetch_size(n_draws, 64, num_workers, bytes_per_sample)

    # Assert
    assert chosen == 64


@pytest.mark.parametrize("batch_size,expected", [(64, 640), (8, 648)])
def test_auto_fetch_size_should_return_a_larger_multiple_when_batches_are_plenty(
    batch_size, expected
):
    """Test the fetch size of a run with batches to spare.

    Given:
        An epoch of 104748 one-byte samples read by eight workers, in
        batches of sixty-four or of eight.
    When:
        A fetch size is chosen.
    Then:
        It should be the largest whole multiple of the batch that
        leaves every prefetch slot its share of fetches.
    """

    # Act
    chosen = loader.auto_fetch_size(104748, batch_size, 8, 1)

    # Assert
    assert chosen == expected


def test_auto_fetch_size_should_cap_a_fetch_at_the_byte_ceiling():
    """Test the byte ceiling where it binds between one batch and the budget.

    Given:
        A run whose batch budget allows ten batches per fetch, and
        samples sized so the ceiling allows four.
    When:
        A fetch size is chosen under the default ceiling and under a
        named one.
    Then:
        It should be four batches under the default, and track the
        named ceiling.
    """

    # Arrange
    per_sample = loader.MAX_FETCH_BYTES // (64 * 4)

    # Act
    default = loader.auto_fetch_size(104748, 64, 8, bytes_per_sample=per_sample)
    named = loader.auto_fetch_size(
        104748, 64, 8, bytes_per_sample=per_sample, max_fetch_bytes=per_sample * 64 * 2
    )

    # Assert
    assert default == 64 * 4
    assert named == 64 * 2


@settings(max_examples=100)
@given(shape=fetch_arguments)
def test_auto_fetch_size_should_always_return_a_whole_multiple_of_the_batch(shape):
    """Test the fetch size's invariants over arbitrary runs.

    Given:
        Any draw count, batch size, worker count, sample size and
        prefetch factor.
    When:
        A fetch size is chosen.
    Then:
        It should be a positive whole multiple of the batch size,
        leave every prefetch slot its share of fetches whenever it
        exceeds one batch, and exceed the byte ceiling only as a
        single batch.
    """

    # Act
    chosen = loader.auto_fetch_size(**shape)

    # Assert
    batch_size = shape["batch_size"]
    assert chosen >= batch_size
    assert chosen % batch_size == 0
    if chosen > batch_size:
        slots = shape["num_workers"] * shape["prefetch_factor"]
        assert shape["n_draws"] // chosen >= slots * loader.FETCHES_PER_PREFETCH_SLOT
        assert chosen * shape["bytes_per_sample"] <= loader.MAX_FETCH_BYTES


def test_auto_fetch_size_should_keep_a_large_epoch_under_the_ceiling(corpus):
    """Test the default ceiling at an epoch large enough to bind it.

    Given:
        The size of one sample of a dataset over sixty-eight signal
        channels, and an epoch of a million draws read by eight
        workers.
    When:
        A fetch size is chosen for it.
    Then:
        It should be more than one batch, and one fetch of it should
        stay within `MAX_FETCH_BYTES`, where the epoch alone would
        choose a fetch past it.
    """

    # Arrange
    per_sample = loader.sample_nbytes(direct_dataset(corpus, [corpus.tracks[0]] * 68))

    # Act
    chosen = loader.auto_fetch_size(10**6, 64, 8, per_sample)

    # Assert
    assert chosen > 64
    assert chosen * per_sample <= loader.MAX_FETCH_BYTES
    assert loader.auto_fetch_size(10**6, 64, 8, 1) * per_sample > (
        loader.MAX_FETCH_BYTES
    )


class TestRechunkedLoader:
    """Fetches handed out one training batch at a time."""

    def test___init___should_raise_when_the_fetch_is_not_a_whole_multiple(self):
        """Test the refusal of a ragged fetch.

        Given:
            A loader fetching five samples and a batch size of four.
        When:
            The loader is wrapped.
        Then:
            It should raise `ValueError`, rather than leave short
            batches in the middle of an epoch.
        """

        # Arrange
        inner, _ = recording_loader(20, 5)

        # Act & assert
        with pytest.raises(ValueError, match="whole multiple"):
            loader.RechunkedLoader(inner, 4)

    @pytest.mark.parametrize("batch_size", [0, -4])
    def test___init___should_raise_when_the_batch_is_below_one(self, batch_size):
        """Test the refusal of a consumer batch that holds no sample.

        Given:
            A loader fetching eight samples, and a batch size of zero
            or below.
        When:
            The loader is wrapped.
        Then:
            It should raise `ValueError` naming ``batch_size``, rather
            than divide by zero or slice backwards.
        """

        # Arrange
        inner, _ = recording_loader(16, 8)

        # Act & assert
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            loader.RechunkedLoader(inner, batch_size)

    def test___init___should_raise_when_the_loader_has_no_batch_size(self):
        """Test the refusal of a loader batching through a batch sampler.

        Given:
            A loader whose batches come from a ``batch_sampler``, so it
            has no ``batch_size``.
        When:
            The loader is wrapped.
        Then:
            It should raise `ValueError` naming ``batch_sampler``.
        """

        # Arrange
        data = torch.utils.data.TensorDataset(torch.arange(16))
        inner = torch.utils.data.DataLoader(
            data,
            batch_sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(data), 8, drop_last=False
            ),
        )

        # Act & assert
        with pytest.raises(ValueError, match="batch_sampler"):
            loader.RechunkedLoader(inner, 4)

    def test___init___should_raise_when_the_loader_drops_its_last_fetch(self):
        """Test the refusal of a loader that loses its partial fetch.

        Given:
            A loader fetching eight of fourteen samples with
            ``drop_last``, which yields one fetch where a loader of
            four would yield three batches.
        When:
            The loader is wrapped at a batch of four.
        Then:
            It should raise `ValueError` naming ``drop_last``.
        """

        # Arrange
        data = torch.utils.data.TensorDataset(torch.arange(14))
        inner = torch.utils.data.DataLoader(data, batch_size=8, drop_last=True)

        # Act & assert
        with pytest.raises(ValueError, match="drop_last"):
            loader.RechunkedLoader(inner, 4)

    def test___init___should_record_the_two_sizes(self):
        """Test the sizes a wrapper reports.

        Given:
            A loader fetching twelve samples, wrapped at a batch of
            four.
        When:
            The wrapper's sizes are read.
        Then:
            It should report four as the batch size, twelve as the
            fetch size, and the loader it wraps.
        """

        # Arrange
        inner, _ = recording_loader(24, 12)

        # Act
        wrapped = loader.RechunkedLoader(inner, 4)

        # Assert
        assert (wrapped.batch_size, wrapped.fetch_size) == (4, 12)
        assert wrapped.loader is inner

    def test___getattr___should_reach_the_wrapped_loaders_attributes(self):
        """Test that the wrapper stays transparent.

        Given:
            A wrapped loader.
        When:
            Its dataset, its sampler and another loader attribute are
            reached for.
        Then:
            They should be the wrapped loader's own, and a name
            neither has should raise `AttributeError`.
        """

        # Arrange
        inner, _ = recording_loader(24, 12)

        # Act
        wrapped = loader.RechunkedLoader(inner, 4)

        # Assert
        assert wrapped.dataset is inner.dataset
        assert wrapped.sampler is inner.sampler
        assert wrapped.num_workers == 0
        with pytest.raises(AttributeError):
            wrapped.no_such_attribute

    def test___getattr___should_let_the_wrapper_be_copied(self):
        """Test that forwarding does not recurse on a half-built wrapper.

        Given:
            A wrapped loader.
        When:
            The wrapper is shallow-copied, which builds an instance
            with no attributes and then asks it for hooks it lacks.
        Then:
            The copy should wrap the same loader at the same sizes.
        """

        # Arrange
        inner, _ = recording_loader(24, 12)
        wrapped = loader.RechunkedLoader(inner, 4)

        # Act
        duplicate = copy.copy(wrapped)

        # Assert
        assert duplicate.loader is inner
        assert (duplicate.batch_size, duplicate.fetch_size) == (4, 12)

    @pytest.mark.parametrize(
        "n_samples,expected", [(24, 6), (26, 7), (12, 3), (3, 1), (0, 0)]
    )
    def test___len___should_count_the_batches_a_pass_yields(self, n_samples, expected):
        """Test the wrapper's length against what it yields.

        Given:
            A loader fetching twelve samples over a number of samples
            that is, or is not, a whole number of fetches and batches.
        When:
            It is wrapped at a batch of four, measured and iterated.
        Then:
            The length should be the batch count the consumer sees,
            which the fetch size cannot change.
        """

        # Arrange
        inner, _ = recording_loader(n_samples, 12)

        # Act
        wrapped = loader.RechunkedLoader(inner, 4)
        yielded = list(wrapped)

        # Assert
        assert len(wrapped) == expected
        assert len(yielded) == expected

    def test___iter___should_keep_tuple_order_and_cut_batches_in_sequence(self):
        """Test what a pass yields, tensor by tensor.

        Given:
            Twenty-six samples of a float tensor and an integer
            tensor, fetched twelve at a time.
        When:
            The wrapped loader is iterated at a batch of four.
        Then:
            Every batch should hold the two tensors in the fetch's
            order, the batches should be the samples in sequence four
            at a time, and the last should be the short remainder.
        """

        # Arrange
        inner, _ = recording_loader(26, 12)

        # Act
        batches = list(loader.RechunkedLoader(inner, 4))

        # Assert
        assert all(len(batch) == 2 for batch in batches)
        assert all(batch[0].dtype == torch.float32 for batch in batches)
        assert all(batch[1].dtype == torch.int64 for batch in batches)
        assert [batch[1].tolist() for batch in batches] == [
            list(range(start, min(start + 4, 26))) for start in range(0, 26, 4)
        ]
        assert torch.equal(
            torch.cat([batch[0] for batch in batches]),
            torch.arange(78, dtype=torch.float32).reshape(26, 3),
        )

    @settings(max_examples=50)
    @given(
        n_samples=st.integers(min_value=0, max_value=60),
        batch_size=st.integers(min_value=1, max_value=8),
        multiple=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    def test___iter___should_yield_what_a_loader_of_the_batch_size_yields(
        self, n_samples, batch_size, multiple, seed
    ):
        """Test the wrapper against a loader of the consumer's batch, tensor for tensor.

        Given:
            Any number of samples of a float tensor and an integer
            tensor, drawn in any order, any batch size, and any whole
            multiple of it as the fetch size.
        When:
            The wrapped loader is iterated and measured, and a loader of
            the batch size over the same order is iterated.
        Then:
            Both should yield the same batches, tensor for tensor, and
            the wrapper's length should be how many it yields.
        """

        # Arrange
        order = numpy.random.RandomState(seed).permutation(n_samples).tolist()
        data = torch.utils.data.TensorDataset(
            torch.arange(n_samples * 3, dtype=torch.float32).reshape(n_samples, 3),
            torch.arange(n_samples, dtype=torch.int64),
        )
        inner = torch.utils.data.DataLoader(
            data, sampler=order, batch_size=batch_size * multiple
        )
        expected = list(
            torch.utils.data.DataLoader(data, sampler=order, batch_size=batch_size)
        )

        # Act
        wrapped = loader.RechunkedLoader(inner, batch_size)
        batches = list(wrapped)

        # Assert
        assert len(wrapped) == len(batches)
        assert same_batches(batches, expected)

    @pytest.mark.parametrize("fetch_size", [12, 4])
    def test___iter___should_yield_views_of_each_fetch(self, fetch_size):
        """Test that a sub-batch copies nothing.

        Given:
            A loader that records every fetch it collates, fetching
            three batches at a time, or one, an identity rechunk.
        When:
            The wrapped loader is iterated.
        Then:
            Every tensor of every batch should share its fetch's
            memory at the offset of its slice, which is what lets a
            pinned fetch hand out pinned batches. On a platform that
            pins no memory, as the Linux CI jobs are, the pinning test
            below skips and this is what holds that.
        """

        # Arrange
        inner, fetches = recording_loader(24, fetch_size)
        per_fetch = fetch_size // 4

        # Act
        batches = list(loader.RechunkedLoader(inner, 4))

        # Assert
        assert len(fetches) == 24 // fetch_size
        for index, batch in enumerate(batches):
            fetched = fetches[index // per_fetch]
            start = (index % per_fetch) * 4
            for tensor, whole in zip(batch, fetched):
                assert tensor.data_ptr() == whole[start:].data_ptr()
                assert tensor.untyped_storage().data_ptr() == (
                    whole.untyped_storage().data_ptr()
                )

    @pytest.mark.parametrize("fetch_size", [12, 4])
    def test___iter___should_hand_out_pinned_batches_from_a_pinned_fetch(
        self, fetch_size
    ):
        """Test that pinning survives the rechunking.

        Given:
            A loader that pins its fetches, three batches at a time or
            one, on a platform where pinning is available.
        When:
            The wrapped loader is iterated.
        Then:
            Every tensor of every batch should be pinned.
        """

        # Arrange
        try:
            pinnable = torch.zeros(1).pin_memory().is_pinned()
        except RuntimeError:
            pinnable = False
        if not pinnable:
            pytest.skip("this platform pins no memory")
        inner, fetches = recording_loader(24, fetch_size, pin=True)

        # Act
        batches = list(loader.RechunkedLoader(inner, 4))

        # Assert
        assert len(batches) == 6
        assert all(tensor.is_pinned() for fetched in fetches for tensor in fetched)
        assert all(tensor.is_pinned() for batch in batches for tensor in batch)

    def test___iter___should_let_go_of_a_fetch_before_pulling_the_next(self):
        """Test that a pass holds one fetch at a time.

        Given:
            A loader fetching eight samples, which notes at each fetch
            whether the tensors of the one before are still alive.
        When:
            The wrapper is iterated at a batch of four, each batch
            dropped as soon as it is yielded.
        Then:
            No earlier fetch should be alive when a later one is
            pulled.
        """

        # Arrange
        data = torch.utils.data.TensorDataset(torch.arange(32, dtype=torch.float32))
        earlier = []
        alive_at_pull = []

        def collate(samples):
            alive_at_pull.append(any(ref() is not None for ref in earlier))
            fetched = torch.utils.data.default_collate(samples)
            earlier[:] = [weakref.ref(tensor) for tensor in fetched]
            return fetched

        inner = torch.utils.data.DataLoader(data, batch_size=8, collate_fn=collate)

        # Act
        batches = iter(loader.RechunkedLoader(inner, 4))
        while next(batches, None) is not None:
            pass

        # Assert
        assert alive_at_pull == [False, False, False, False]


def test_build_training_loader_should_raise_when_no_signal_is_given(corpus):
    """Test the refusal of a loader with nothing to predict.

    Given:
        Loader arguments whose signals are ``None``, and then empty.
    When:
        The loader is built from each.
    Then:
        It should raise `ValueError` saying signals are required.
    """

    # Act & assert
    for signals in (None, []):
        with pytest.raises(ValueError, match="signals is required"):
            loader.build_training_loader(**arguments(corpus, signals=signals))


@pytest.mark.parametrize("label", ["signals", "controls"])
def test_build_training_loader_should_raise_when_given_a_single_track(corpus, label):
    """Test the refusal of a bare track where a list of groups belongs.

    Given:
        Loader arguments whose signals, or whose controls, are one
        path.
    When:
        The loader is built.
    Then:
        It should raise `TypeError`, rather than read the path as one
        group per character.
    """

    # Arrange
    given = arguments(corpus, **{label: corpus.tracks[0]})

    # Act & assert
    with pytest.raises(TypeError, match="one entry per group"):
        loader.build_training_loader(**given)


def test_build_training_loader_should_raise_when_chroms_names_no_locus(corpus):
    """Test that a contig filter matching nothing reaches the caller.

    Given:
        A ``chroms`` list in a naming the loci never use.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` naming ``chroms``, rather than
        build a loader with nothing in it.
    """

    # Act & assert
    with pytest.raises(ValueError, match="is on a contig `chroms` names"):
        loader.build_training_loader(**arguments(corpus, chroms=["1", "2"]))


def test_build_training_loader_should_raise_when_no_signal_shares_a_contig(
    corpus, tmp_path
):
    """Test that tracks naming other contigs than the windows are refused.

    Given:
        A signal track holding positions on one contig only, and a
        contig filter keeping the other.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` naming the signals, rather than
        train on targets that are all zero.
    """

    # Arrange
    track = builders.write_track(
        str(tmp_path / "chr2-only.track.parquet"),
        [("chr2", pos, 1.0) for pos in (500, 600, 700, 800)],
    )

    # Act & assert
    with pytest.raises(ValueError, match="no `signals` track holds a position"):
        loader.build_training_loader(
            **arguments(corpus, signals=[track], chroms=["chr1"])
        )


def test_build_training_loader_should_raise_when_no_control_shares_a_contig(
    corpus, tmp_path
):
    """Test that the contig check covers the controls as well.

    Given:
        Signals that share contigs with the windows, and a control
        track holding positions only on a contig the filter drops.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` naming the controls.
    """

    # Arrange
    track = builders.write_track(
        str(tmp_path / "chr2-only.track.parquet"), [("chr2", 500, 1.0)]
    )

    # Act & assert
    with pytest.raises(ValueError, match="no `controls` track holds a position"):
        loader.build_training_loader(
            **arguments(corpus, controls=[track], chroms=["chr1"])
        )


def test_build_training_loader_should_raise_when_no_peak_is_kept(corpus, tmp_path):
    """Test the refusal of a peak set the filters emptied.

    Given:
        A count floor no locus reaches, at a negative ratio of zero,
        and an empty cache directory.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` saying no peak is kept, and the
        directory should hold no entry, so a later run builds again
        rather than reading an entry that refuses again.
    """

    # Arrange
    cache_dir = tmp_path / "entries"

    # Act
    with pytest.raises(ValueError, match="no peak is kept"):
        loader.build_training_loader(
            **arguments(
                corpus, min_counts=10**9, negative_ratio=0.0, cache_dir=str(cache_dir)
            )
        )

    # Assert
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_build_training_loader_should_raise_when_no_negative_is_kept(corpus, tmp_path):
    """Test the refusal of a positive ratio over a negative set the filters emptied.

    Given:
        An exclusion list covering every negative, a positive negative
        ratio, and an empty cache directory.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` saying there are no negatives to
        draw from, and the directory should hold no entry.
    """

    # Arrange
    cache_dir = tmp_path / "entries"

    # Act
    with pytest.raises(ValueError, match="no negatives to draw from"):
        loader.build_training_loader(
            **arguments(
                corpus,
                exclusion_lists=corpus.exclusion,
                negative_ratio=0.5,
                cache_dir=str(cache_dir),
            )
        )

    # Assert
    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_build_training_loader_should_raise_when_a_frame_is_given_with_a_cache(corpus):
    """Test that an in-memory input cannot be keyed.

    Given:
        Peaks as a frame, and then a signal track as a frame, with the
        persisted cache left on.
    When:
        The loader is built from each.
    Then:
        It should raise `TypeError` naming the argument and saying an
        input held in memory has no key, and naming ``cache_dir=False``
        as the way to read one.
    """

    # Arrange
    peaks = polars.DataFrame(
        {
            "chrom": [row[0] for row in PEAKS],
            "start": [row[1] for row in PEAKS],
            "end": [row[2] for row in PEAKS],
        }
    )
    track = builders.track_rows_frame(SIGNAL_ROWS[0])

    # Act & assert
    with pytest.raises(TypeError, match="`peaks` must be a str.*cache_dir=False"):
        loader.build_training_loader(**arguments(corpus, peaks=peaks))
    with pytest.raises(TypeError, match="`signals` must be a str.*cache_dir=False"):
        loader.build_training_loader(**arguments(corpus, signals=[track]))


def test_build_training_loader_should_report_batches_per_epoch(corpus):
    """Test the loader's length.

    Given:
        A loader.
    When:
        Its length is taken.
    Then:
        It should be the sampler's draws in batches, the last one
        partial.
    """

    # Act
    built = loader.build_training_loader(**arguments(corpus))

    # Assert
    assert len(built) == -(-len(built.sampler) // BATCH_SIZE)
    assert len(built.dataset) == len(built.sampler)
    assert built.dataset.sampler is built.sampler


def test_build_training_loader_should_carry_the_signal_grouping_to_the_dataset(corpus):
    """Test that the nesting of the signals decides the permutation.

    Given:
        Signals grouped one and two.
    When:
        The loader is built.
    Then:
        Its dataset should carry that grouping and its permutation,
        and neither for the controls.
    """

    # Act
    built = loader.build_training_loader(**arguments(corpus))

    # Assert
    assert built.dataset.signal_groups == (1, 2)
    assert torch.equal(
        built.dataset.signal_perm, groups.channel_permutation_from_groups([1, 2])
    )
    assert built.dataset.control_groups is None
    assert built.dataset.control_perm is None


def test_build_training_loader_should_carry_the_control_grouping_to_the_dataset(
    corpus,
):
    """Test that the nesting of the controls is kept beside the signals'.

    Given:
        Signals grouped one and two, and controls given as one pair.
    When:
        The loader is built.
    Then:
        Its dataset should report both groupings as they were nested.
    """

    # Act
    built = loader.build_training_loader(
        **arguments(corpus, controls=[[corpus.tracks[3], corpus.tracks[4]]])
    )

    # Assert
    assert (built.dataset.signal_groups, built.dataset.control_groups) == (
        (1, 2),
        (2,),
    )


def test_build_training_loader_should_pass_the_loader_settings_through(corpus):
    """Test the worker and pinning settings of the loader returned.

    Given:
        Loader arguments asking for two workers with a timeout, and for
        none.
    When:
        Each loader is built, and not iterated.
    Then:
        The first should hold two persistent workers and the second
        none, both should carry the batch size and the pinning asked
        for, and each the timeout asked for or none.
    """

    # Act
    workers = loader.build_training_loader(
        **arguments(corpus, num_workers=2, timeout=WORKER_TIMEOUT)
    )
    inline = loader.build_training_loader(**arguments(corpus, pin_memory=True))

    # Assert
    assert (workers.num_workers, workers.persistent_workers) == (2, True)
    assert (inline.num_workers, inline.persistent_workers) == (0, False)
    assert (workers.pin_memory, inline.pin_memory) == (False, True)
    assert (workers.timeout, inline.timeout) == (WORKER_TIMEOUT, 0)
    assert workers.batch_size == BATCH_SIZE


@pytest.mark.parametrize("rank,world_size", [(0, None), (None, 2)])
def test_build_training_loader_should_raise_when_only_one_of_rank_and_world_size_is_given(
    corpus, rank, world_size
):
    """Test the refusal of half a shard.

    Given:
        A rank without a world size, and a world size without a rank.
    When:
        The loader is built from each.
    Then:
        It should raise `ValueError` naming both arguments.
    """

    # Act & assert
    with pytest.raises(ValueError, match="rank and world_size are given together"):
        loader.build_training_loader(
            **arguments(corpus, rank=rank, world_size=world_size)
        )


def test_build_training_loader_should_raise_when_ranks_share_no_seed(corpus):
    """Test the refusal of ranks that would each draw their own epochs.

    Given:
        A world of two ranks and no ``random_state``.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` asking for a shared
        ``random_state``.
    """

    # Act & assert
    with pytest.raises(ValueError, match="random_state every rank shares"):
        loader.build_training_loader(
            **arguments(corpus, random_state=None, rank=0, world_size=2)
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"batch_size": 0}, "batch_size and world_size must be at least 1"),
        ({"batch_size": 8, "rank": 2, "world_size": 2}, "rank must be in range"),
        ({"batch_size": 6, "rank": 0, "world_size": 4}, "must be divisible"),
        ({"negative_ratio": -0.5}, "negative_ratio must be non-negative"),
        ({"max_jitter": -1}, "max_jitter must be non-negative"),
        ({"negatives": None, "negative_ratio": 0.5}, "no negatives to draw from"),
        ({"fetch_size": BATCH_SIZE + 2}, "fetch_size must be a positive"),
        (
            {"batch_size": 8, "rank": 0, "world_size": 2, "fetch_size": 6},
            "fetch_size must be a positive whole multiple of the 4 samples",
        ),
        ({"fetch_size": float(BATCH_SIZE)}, "fetch_size must be a positive"),
        ({"signals": [[]]}, "a signal group is empty"),
        ({"cache_residency": "bogus"}, "residency must be one of"),
        ({"num_workers": -1}, "num_workers option should be non-negative"),
        ({"timeout": -1}, "timeout option should be non-negative"),
    ],
)
def test_build_training_loader_should_refuse_an_argument_before_building(
    tmp_path, overrides, match
):
    """Test that a refusal needing no data is made before anything is read.

    Given:
        An argument whose value alone is refused, among locus files and
        a track that do not exist, a reference written without an
        index, and an empty cache directory.
    When:
        The loader is built.
    Then:
        It should raise `ValueError` naming the argument rather than
        the missing files, and leave the reference unindexed, which
        taking the key would not, and the cache directory empty.
    """

    # Arrange
    fasta = builders.write_fasta(str(tmp_path / "genome.fa"), CHROM_SIZES, index=False)
    cache_dir = tmp_path / "entries"
    cache_dir.mkdir()
    given = dict(
        {
            "peaks": str(tmp_path / "absent-peaks.bed"),
            "negatives": str(tmp_path / "absent-negatives.bed"),
            "sequences": fasta,
            "signals": [str(tmp_path / "absent.track.parquet")],
            "cache_dir": str(cache_dir),
            "random_state": 0,
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "pin_memory": False,
        },
        **overrides,
    )

    # Act
    with pytest.raises(ValueError, match=match):
        loader.build_training_loader(**given)

    # Assert
    assert not os.path.exists(fasta + ".fai")
    assert not any(cache_dir.iterdir())


@pytest.mark.parametrize("fetch_size", [0, BATCH_SIZE + 2, BATCH_SIZE // 2])
def test_build_training_loader_should_raise_when_the_fetch_is_not_a_whole_multiple(
    corpus, fetch_size
):
    """Test the refusal of a fetch that cannot be cut into batches.

    Given:
        A fetch size of zero, one that is not a multiple of the batch,
        and one smaller than the batch.
    When:
        The loader is built with each.
    Then:
        It should raise `ValueError` naming ``fetch_size``.
    """

    # Act & assert
    with pytest.raises(ValueError, match="fetch_size must be a positive whole multiple"):
        loader.build_training_loader(**arguments(corpus, fetch_size=fetch_size))


@pytest.mark.parametrize("multiple", [1, 2, 4, 8, numpy.int64(2)])
def test_build_training_loader_should_yield_one_stream_at_any_fetch_size(
    corpus, multiple
):
    """Test that fetching in bulk does not change the batch stream.

    Given:
        A loader fetching several batches per transfer, or one, the
        fetch size given as a Python or a numpy integer, and a loader
        built with the default fetch size.
    When:
        Two epochs are drained from each.
    Then:
        Each should expose the dataset and sampler it draws through,
        report the same number of batches and yield the same tensors,
        batch for batch.
    """

    # Arrange
    bulk = loader.build_training_loader(
        **arguments(corpus, fetch_size=BATCH_SIZE * multiple)
    )
    reference = loader.build_training_loader(**arguments(corpus))

    # Act
    rechunked = drain(bulk)
    expected = drain(reference)

    # Assert
    assert isinstance(bulk, loader.RechunkedLoader)
    assert isinstance(reference, loader.RechunkedLoader)
    assert bulk.dataset.sampler is bulk.sampler
    assert len(bulk) == len(reference)
    assert same_batches(rechunked, expected)


def test_build_training_loader_should_fetch_less_under_a_lower_byte_ceiling(corpus):
    """Test that ``max_fetch_bytes`` reaches the fetch size chosen by default.

    Given:
        A loader of one-sample batches over a worker, built with the
        default ceiling and with one of two samples' bytes.
    When:
        The fetch size each chose is read, without iterating.
    Then:
        The default should fetch more than two samples at a time and
        the lower ceiling exactly two.
    """

    # Arrange
    given = arguments(corpus, batch_size=1, num_workers=1)
    default = loader.build_training_loader(**given)
    ceiling = 2 * loader.sample_nbytes(default.dataset)

    # Act
    lowered = loader.build_training_loader(**dict(given, max_fetch_bytes=ceiling))

    # Assert
    assert default.fetch_size > 2
    assert lowered.fetch_size == 2


def test_build_training_loader_should_advance_the_epoch_once_per_pass_in_bulk(corpus):
    """Test that a pass advances the sampler's epoch exactly once.

    Given:
        A loader fetching two batches per transfer.
    When:
        Two passes are made.
    Then:
        The sampler should be at epoch two.
    """

    # Arrange
    built = loader.build_training_loader(**arguments(corpus, fetch_size=BATCH_SIZE * 2))

    # Act
    drain(built, n_epochs=2)

    # Assert
    assert built.sampler.epoch == 2


@pytest.mark.parametrize("world_size,fetch_multiple", [(1, 1), (2, 2), (4, 1)])
def test_build_training_loader_should_let_the_ranks_together_read_one_ranks_batches(
    corpus, world_size, fetch_multiple
):
    """Test sharded loaders against the unsharded one, epoch by epoch.

    Given:
        One loader per rank of a world of one, two or four, sharing a
        global batch of eight and a seed, some fetching in bulk, and
        the unsharded loader of the same batch.
    When:
        Each rank's ``sampler`` is set to an epoch and the ranks'
        batches are concatenated step by step, for two epochs.
    Then:
        Every step should be the unsharded loader's batch of that
        epoch, the trailing partial batch being dropped.
    """

    # Arrange
    batch_size = 8
    local = batch_size // world_size
    ranks = [
        loader.build_training_loader(
            **arguments(
                corpus,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                fetch_size=local * fetch_multiple,
            )
        )
        for rank in range(world_size)
    ]
    unsharded = loader.build_training_loader(**arguments(corpus, batch_size=batch_size))

    # Act & assert
    for epoch in (0, 1):
        expected = drain(unsharded, n_epochs=1)
        per_rank = []
        for built in ranks:
            built.sampler.set_epoch(epoch)
            per_rank.append(drain(built, n_epochs=1))
        combined = [
            tuple(torch.cat(parts) for parts in zip(*steps)) for steps in zip(*per_rank)
        ]
        full = len(unsharded.sampler) // batch_size
        assert len(ranks[0]) == full
        assert same_batches(combined, expected[:full])


def test_build_training_loader_should_drop_the_outlier_peaks(corpus):
    """Test the outlier filter against totals taken from the rows.

    Given:
        Peaks one of which sits over a region fifty times the rest in
        a quiet channel, beside a loud pair of channels in a group of
        their own.
    When:
        The loader is built.
    Then:
        It should keep exactly the peaks the mask leaves over per-group
        totals derived from the track rows, which is all but that one
        and is not what one ceiling over the pooled total would leave,
        and every negative, the filter being over peaks only.
    """

    # Arrange
    totals = window_totals(SIGNAL_ROWS, PEAKS, MAX_JITTER)
    by_group = torch.from_numpy(
        numpy.stack([totals[:, 0], totals[:, 1] + totals[:, 2]], axis=1).astype(
            numpy.float32
        )
    )
    expected = int((~loader.outlier_mask(by_group)).sum())

    # Act
    built = loader.build_training_loader(**arguments(corpus))

    # Assert
    assert expected == len(PEAKS) - 1
    assert not loader.outlier_mask(by_group.sum(dim=1, keepdim=True)).any()
    assert built.dataset.n_peaks == expected
    assert built.dataset.n_negatives == len(NEGATIVES)


def test_build_training_loader_should_drop_loci_outside_the_count_band(corpus):
    """Test the count filter against totals taken from the rows.

    Given:
        A count floor and ceiling that are themselves the totals of
        two negatives on the first signal channel.
    When:
        The loader is built with and without them.
    Then:
        It should keep exactly the negatives whose total lies in the
        band, both ends included, and fewer peaks than without.
    """

    # Arrange
    totals = numpy.sort(window_totals(SIGNAL_ROWS[:1], NEGATIVES, 0)[:, 0])
    floor, ceiling = float(totals[5]), float(totals[-6])
    within = int(((totals >= floor) & (totals <= ceiling)).sum())
    unfiltered = loader.build_training_loader(**arguments(corpus))

    # Act
    filtered = loader.build_training_loader(
        **arguments(corpus, min_counts=floor, max_counts=ceiling)
    )

    # Assert
    assert 0 < within < len(NEGATIVES)
    assert filtered.dataset.n_negatives == within
    assert 0 < filtered.dataset.n_peaks < unfiltered.dataset.n_peaks


@pytest.mark.parametrize(
    "min_counts,max_counts,expected",
    [(None, None, 6), (2.0, None, 5), (None, 4.0, 5), (2.0, 4.0, 4)],
)
def test_build_training_loader_should_keep_the_loci_extract_loci_keeps(
    tmp_path, min_counts, max_counts, expected
):
    """Test the count band through both entry points that apply it.

    Given:
        Six peaks whose counts are one to five and, for the last, not a
        number, and a count floor, a ceiling, both or neither.
    When:
        The loci are extracted, and a loader is built over them.
    Then:
        Both should keep the same number of peaks, the count that is
        not a number among them.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    centres = [105 + 100 * index for index in range(6)]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, math.nan]
    given = {
        "sequences": builders.write_fasta(
            str(tmp_path / "genome.fa"), {"chr1": 1000}, index=True
        ),
        "signals": [
            builders.write_track(
                str(tmp_path / "t.track.parquet"),
                [("chr1", centre, value) for centre, value in zip(centres, values)],
            )
        ],
        "min_counts": min_counts,
        "max_counts": max_counts,
    }
    peaks = builders.write_bed(
        str(tmp_path / "peaks.bed"),
        [("chr1", centre - 5, centre + 5) for centre in centres],
    )

    # Act
    extracted = loci.extract_loci(peaks, in_window=20, out_window=10, **given)
    built = loader.build_training_loader(
        peaks,
        None,
        in_window=20,
        out_window=10,
        max_jitter=0,
        negative_ratio=0.0,
        random_state=0,
        num_workers=0,
        pin_memory=False,
        cache_dir=False,
        **given,
    )

    # Assert
    assert int(extracted.kept.sum()) == expected
    assert built.dataset.n_peaks == expected


def test_build_training_loader_should_apply_the_exclusion_lists(corpus):
    """Test that excluded regions remove the loci they cover.

    Given:
        An exclusion list covering every negative and no peak, given
        as one path and as a list of it.
    When:
        Loaders are built with it, at a negative ratio of zero.
    Then:
        Both should keep no negative and every peak the unexcluded
        loader keeps.
    """

    # Arrange
    unexcluded = loader.build_training_loader(**arguments(corpus))

    # Act
    by_path = loader.build_training_loader(
        **arguments(corpus, exclusion_lists=corpus.exclusion, negative_ratio=0.0)
    )
    by_list = loader.build_training_loader(
        **arguments(corpus, exclusion_lists=[corpus.exclusion], negative_ratio=0.0)
    )

    # Assert
    assert unexcluded.dataset.n_negatives == len(NEGATIVES)
    for built in (by_path, by_list):
        assert built.dataset.n_negatives == 0
        assert built.dataset.n_peaks == unexcluded.dataset.n_peaks


def test_build_training_loader_should_draw_peaks_only_when_negatives_are_none(corpus):
    """Test a loader with no negative set.

    Given:
        Loader arguments with no negatives, at a ratio of zero.
    When:
        The loader is built and an epoch drained.
    Then:
        It should label every sample of the epoch a peak.
    """

    # Act
    built = loader.build_training_loader(
        **arguments(corpus, negatives=None, negative_ratio=0.0)
    )
    labels = torch.cat([batch[2] for batch in drain(built, n_epochs=1)])

    # Assert
    assert built.dataset.n_negatives == 0
    assert len(labels) == built.dataset.n_peaks
    assert labels.all()


def test_build_training_loader_should_centre_peaks_and_not_negatives_on_summits(
    corpus, tmp_path
):
    """Test which loci the summit column moves.

    Given:
        Ten-column peaks whose summits sit off their midpoints, a
        three-column negative file, and a loader that neither shuffles,
        jitters nor mirrors.
    When:
        The loader is built with ``summits`` and its first batches
        read.
    Then:
        Each peak's sequence should be the reference centred on its
        summit, and each negative's the reference centred on its
        midpoint, the negative file having no summit column to read.
    """

    # Arrange
    summits = [("chr1", 500 + 200 * index, 560 + 200 * index, 7) for index in range(4)]
    peaks = builders.write_bed(
        str(tmp_path / "peaks.narrowPeak"),
        [
            (chrom, start, end, ".", 0, ".", 0, 0, 0, offset)
            for chrom, start, end, offset in summits
        ],
    )
    reference = parse_fasta(corpus.fasta)
    half = IN_WINDOW // 2

    # Act
    built = loader.build_training_loader(
        **arguments(
            corpus,
            peaks=peaks,
            summits=True,
            max_jitter=0,
            shuffle=False,
            reverse_complement=False,
            negative_ratio=1.0,
            batch_size=8,
        )
    )
    (sequences, _, labels) = drain(built, n_epochs=1)[0]

    # Assert
    assert built.dataset.n_peaks == 4
    peak_rows = sequences[labels == 1]
    for row, (chrom, start, _, offset) in zip(peak_rows, summits):
        centre = start + offset
        expected = torch.from_numpy(
            one_hot(reference[chrom][centre - half : centre + half])
        )
        assert torch.equal(row, expected)
    centres = {
        chrom: [
            start + (end - start) // 2 for on, start, end in NEGATIVES if on == chrom
        ]
        for chrom in CHROM_SIZES
    }
    for row in sequences[labels == 0]:
        assert any(
            torch.equal(
                row,
                torch.from_numpy(
                    one_hot(reference[chrom][centre - half : centre + half])
                ),
            )
            for chrom, mids in centres.items()
            for centre in mids
        )


def test_build_training_loader_should_read_an_ambiguity_code_as_no_base(tmp_path):
    """Test what a letter that is none of the four bases encodes to.

    Given:
        A reference carrying IUPAC ambiguity codes other than ``N``, in
        upper and lower case, inside a peak's input window.
    When:
        A loader is built over it and the peak read, unshuffled,
        unjittered and unmirrored.
    Then:
        Each ambiguous position should be an all-zero column and every
        other position should hold exactly one base, rather than the
        read being refused.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    given = one_peak_arguments(tmp_path, "RYkmSw")

    # Act
    built = loader.build_training_loader(**given)
    (sequences, _, _) = next(iter(built))

    # Assert
    per_position = sequences[0].sum(dim=0).tolist()
    first = 300 - (300 - IN_WINDOW // 2)
    assert per_position[first : first + 6] == [0] * 6
    assert per_position[:first] == [1] * first
    assert per_position[first + 6 :] == [1] * (IN_WINDOW - first - 6)


def test_build_training_loader_should_refuse_a_window_holding_a_character_that_is_no_letter(
    tmp_path,
):
    """Test what a character that is no letter does to a read.

    Given:
        A reference carrying a gap character inside a peak's input
        window.
    When:
        A loader is built over it and the peak read.
    Then:
        The build should succeed, and the read should raise
        `ValueError` saying the character is in neither the alphabet
        nor the ignored residues.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    built = loader.build_training_loader(**one_peak_arguments(tmp_path, "AC-GTA"))

    # Act & assert
    with pytest.raises(ValueError, match="not in `alphabet` or in `ignore`"):
        next(iter(built))


def test_build_training_loader_should_differ_between_epochs(corpus):
    """Test that a second pass is a second epoch.

    Given:
        A shuffled loader.
    When:
        Two epochs are drained.
    Then:
        They should be the same length and not the same stream, so
        that agreement between two loaders elsewhere is not two
        replays of one epoch.
    """

    # Arrange
    built = loader.build_training_loader(**arguments(corpus))

    # Act
    both = drain(built, n_epochs=2)

    # Assert
    half = len(both) // 2
    assert half == len(built)
    assert not same_batches(both[:half], both[half:])


@pytest.mark.parametrize("config_id", CONFIG_IDS)
def test_build_training_loader_should_match_the_golden_digest(golden, config_id):
    """Test that the emitted batches are the recorded ones.

    Given:
        The arithmetic corpus and one golden sampler configuration.
    When:
        The loader is built in this process and two epochs are hashed.
    Then:
        The digest should equal the one recorded for that
        configuration.
    """

    # Arrange
    built = loader.build_training_loader(**golden_arguments(golden, config_id))

    # Act
    digest = batch_digest(drain(built))

    # Assert
    assert digest == GOLDEN_BATCHES[config_id]


@pytest.mark.workers
@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("config_id", CONFIG_IDS)
def test_build_training_loader_should_return_a_dataset_and_sampler_workers_read_as_recorded(
    golden, config_id, persistent
):
    """Test the built dataset and sampler under a loader of their own with workers.

    Given:
        The dataset and sampler built for one golden sampler
        configuration over the arithmetic corpus, wrapped in a plain
        loader of two workers that persist between epochs or are
        started again for each.
    When:
        Two epochs are hashed.
    Then:
        The digest should equal the one recorded for that
        configuration in a single process, the second epoch included,
        which a design keeping the epoch on a worker's copy of the
        dataset replays as the first once workers are started again.
    """

    # Arrange
    built = loader.build_training_loader(**golden_arguments(golden, config_id))
    workers = with_workers(built, 2, persistent)

    # Act
    digest = batch_digest(drain(workers))

    # Assert
    assert digest == GOLDEN_BATCHES[config_id]


@pytest.mark.workers
@pytest.mark.parametrize("num_workers", [1, 2])
def test_build_training_loader_should_yield_one_stream_under_any_worker_count(
    corpus, num_workers
):
    """Test the loader returned against the stream of a single process.

    Given:
        A loader built for one worker or two, with a timeout, and one
        built for none.
    When:
        Two epochs are drained from each.
    Then:
        The two streams should be equal, the second epoch included.
    """

    # Arrange
    expected = drain(loader.build_training_loader(**arguments(corpus)))
    built = loader.build_training_loader(
        **arguments(corpus, num_workers=num_workers, timeout=WORKER_TIMEOUT)
    )

    # Act
    stream = drain(built)

    # Assert
    assert same_batches(stream, expected)


@pytest.mark.parametrize("residency", ["memory", "mapped"])
def test_build_training_loader_should_yield_the_same_batches_on_a_hit_and_a_cold_build(
    corpus, tmp_path, capsys, residency
):
    """Test a persisted entry against the build that wrote it.

    Given:
        An empty cache directory, and loader arguments with controls,
        under either residency.
    When:
        The loader is built cold, built again over the entry the first
        build wrote, and built with the cache off.
    Then:
        The second build should report reusing the entry, all three
        should yield the same batches, and the directory should hold
        exactly the one entry.
    """

    # Arrange
    cache_dir = str(tmp_path / "entries")
    given = arguments(
        corpus,
        controls=[[corpus.tracks[3], corpus.tracks[4]]],
        cache_dir=cache_dir,
        cache_residency=residency,
        verbose=True,
    )

    # Act
    cold = drain(loader.build_training_loader(**given))
    cold_report = capsys.readouterr().out
    hit = drain(loader.build_training_loader(**given))
    hit_report = capsys.readouterr().out
    uncached = drain(loader.build_training_loader(**dict(given, cache_dir=False)))

    # Assert
    assert "Reusing the window cache" not in cold_report
    assert "Reusing the window cache at {}".format(cache_dir) in hit_report
    assert all(len(batch) == 4 for batch in cold)
    assert same_batches(hit, cold)
    assert same_batches(uncached, cold)
    entries = [name for name in os.listdir(cache_dir) if "." not in name]
    assert len(entries) == 1


def test_build_training_loader_should_reuse_its_entry_over_an_unindexed_reference(
    corpus, tmp_path, capsys
):
    """Test the entry of a build over a reference written without an index.

    Given:
        A reference of the corpus's contigs with no FASTA index beside
        it, and an empty cache directory.
    When:
        The loader is built twice from the same arguments.
    Then:
        The second build should report reusing the entry, and the
        directory should hold exactly the one entry, the index the
        first build wrote being in its key from the start.
    """

    # Arrange
    fasta = builders.write_fasta(str(tmp_path / "genome.fa"), CHROM_SIZES, index=False)
    cache_dir = str(tmp_path / "entries")
    given = arguments(corpus, sequences=fasta, cache_dir=cache_dir, verbose=True)

    # Act
    loader.build_training_loader(**given)
    capsys.readouterr()
    loader.build_training_loader(**given)
    second_report = capsys.readouterr().out

    # Assert
    assert "Reusing the window cache at {}".format(cache_dir) in second_report
    entries = [name for name in os.listdir(cache_dir) if "." not in name]
    assert len(entries) == 1


def test_build_training_loader_should_read_an_entry_without_the_pileup_extra(
    corpus, tmp_path, mocker
):
    """Test what a hit needs installed.

    Given:
        An entry a build has written, and then an interpreter state in
        which neither module of the ``pileup`` extra will import.
    When:
        The loader is built again from the same arguments.
    Then:
        It should yield the batches the build yielded, a hit joining
        nothing.
    """

    # Arrange
    given = arguments(corpus, cache_dir=str(tmp_path / "entries"))
    cold = drain(loader.build_training_loader(**given))
    mocker.patch.dict(sys.modules, {"polars_bio": None, "giql": None})

    # Act
    hit = drain(loader.build_training_loader(**given))

    # Assert
    assert same_batches(hit, cold)


def test_build_training_loader_should_write_no_entry_when_the_cache_is_off(
    corpus, tmp_path, monkeypatch
):
    """Test that ``cache_dir=False`` touches no cache directory.

    Given:
        A default cache directory that does not exist yet.
    When:
        A loader is built with the cache off and a batch is read.
    Then:
        The directory should still not exist.
    """

    # Arrange
    root = tmp_path / "untouched"
    monkeypatch.setenv("QUADRAT_CACHE_DIR", str(root))

    # Act
    built = loader.build_training_loader(**arguments(corpus, cache_dir=False))
    first = next(iter(built))

    # Assert
    assert len(first) == 3
    assert not root.exists()


def test_build_training_loader_should_make_its_scratch_copy_in_the_scratch_dir(
    corpus, tmp_path
):
    """Test where a build that persists nothing keeps its mapped caches.

    Given:
        The cache off, the mapped residency, and an empty scratch
        directory named.
    When:
        The loader is built and a batch read.
    Then:
        The scratch directory should hold the caches' scratch copy.
    """

    # Arrange
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    # Act
    built = loader.build_training_loader(
        **arguments(
            corpus,
            cache_dir=False,
            cache_residency="mapped",
            scratch_dir=str(scratch),
        )
    )
    next(iter(built))

    # Assert
    assert any(name.startswith("quadrat-cache-") for name in os.listdir(scratch))


def test_build_training_loader_should_write_its_entry_to_the_default_directory(
    corpus, tmp_path, monkeypatch
):
    """Test where an entry goes when no directory is named.

    Given:
        A default cache directory named through the environment.
    When:
        A loader is built naming none.
    Then:
        The entry should be under the default directory.
    """

    # Arrange
    monkeypatch.setenv("QUADRAT_CACHE_DIR", str(tmp_path / "default"))

    # Act
    loader.build_training_loader(**arguments(corpus))

    # Assert
    entries = [name for name in os.listdir(cache.default_cache_dir()) if "." not in name]
    assert len(entries) == 1


def test_build_training_loader_should_accept_in_memory_inputs_when_the_cache_is_off(
    corpus,
):
    """Test frames for loci and tracks against the files they hold.

    Given:
        The peaks as a frame and every signal track as a frame, with
        the cache off.
    When:
        The loader is built from them and from the files.
    Then:
        Both should yield the same batches.
    """

    # Arrange
    peaks = polars.DataFrame(
        {
            "chrom": [row[0] for row in PEAKS],
            "start": [row[1] for row in PEAKS],
            "end": [row[2] for row in PEAKS],
        }
    )
    frames = [builders.track_rows_frame(rows) for rows in SIGNAL_ROWS]
    from_files = loader.build_training_loader(**arguments(corpus, cache_dir=False))

    # Act
    from_frames = loader.build_training_loader(
        **arguments(
            corpus, peaks=peaks, signals=[frames[0], frames[1:]], cache_dir=False
        )
    )

    # Assert
    assert same_batches(drain(from_frames, n_epochs=1), drain(from_files, n_epochs=1))


def test_build_training_loader_should_take_chroms_as_a_one_shot_iterable(corpus):
    """Test a contig filter that can be walked only once.

    Given:
        A contig filter as a generator, and the same filter as a list.
    When:
        A loader is built from each.
    Then:
        Both should keep the same loci, all of them on the one contig
        named.
    """

    # Arrange
    listed = loader.build_training_loader(**arguments(corpus, chroms=["chr2"]))

    # Act
    generated = loader.build_training_loader(
        **arguments(corpus, chroms=(name for name in ["chr2"]))
    )

    # Assert
    on_chr2 = len([row for row in NEGATIVES if row[0] == "chr2"])
    assert generated.dataset.n_negatives == listed.dataset.n_negatives == on_chr2
    assert generated.dataset.n_peaks == listed.dataset.n_peaks


def test_build_training_loader_should_report_what_it_kept_when_verbose(corpus, capsys):
    """Test the report of a cold build.

    Given:
        Loader arguments with ``verbose`` on.
    When:
        The loader is built cold.
    Then:
        It should print the peaks kept, the peaks dropped as outliers
        and the negatives kept.
    """

    # Act
    built = loader.build_training_loader(
        **arguments(corpus, verbose=True, cache_dir=False)
    )
    reported = capsys.readouterr().out

    # Assert
    dropped = len(PEAKS) - built.dataset.n_peaks
    assert (
        "Training Peaks: {} ({} dropped as outliers)".format(
            built.dataset.n_peaks, dropped
        )
        in reported
    )
    assert "Training Negatives: {}".format(len(NEGATIVES)) in reported


def test_build_training_loader_should_print_nothing_when_not_verbose(corpus, capsys):
    """Test the silence of a cold build.

    Given:
        Loader arguments with ``verbose`` off.
    When:
        The loader is built cold.
    Then:
        It should print nothing.
    """

    # Act
    loader.build_training_loader(**arguments(corpus, cache_dir=False))

    # Assert
    assert capsys.readouterr().out == ""
