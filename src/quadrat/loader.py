"""The training loader: locus files and tracks in, batches out.

Provides `build_training_loader`, which assembles a window cache, a
sampler and a dataset behind a `torch.utils.data.DataLoader`;
`outlier_mask`, the peak filter it applies; and `auto_fetch_size`,
`sample_nbytes` and `RechunkedLoader`, which move samples between
processes in larger units than a training step consumes.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`. Importing this module
imports ``torch``, which `import quadrat` does not.
"""

import numbers
import os

import torch

from quadrat.cache import (
    default_cache_dir,
    resolve_residency,
    window_cache,
    window_cache_key,
    window_group_sums,
)
from quadrat.dataset import LocusDataset
from quadrat.groups import normalize_signal_groups
from quadrat.loci import readable_windows, require_track_contigs, within_counts
from quadrat.sampling import PeakNegativeMixSampler, ShardedEpochSampler
from quadrat.stores import SequenceStore

#: How many fetches an epoch leaves each prefetch slot, as
#: `auto_fetch_size` states. Fewer, larger fetches amortise the
#: per-transfer cost further, until there are too few of them to keep
#: every slot filled; this many leaves the slots fed.
FETCHES_PER_PREFETCH_SLOT = 10


#: The default ceiling on one fetch, in bytes. Up to ``num_workers *
#: prefetch_factor`` fetches wait between the workers and the
#: consumer, and the consumer holds the fetch its current batch was cut
#: from for as long as it keeps any batch cut from it, a batch being
#: a view; what a `RechunkedLoader` itself holds, its docstring
#: states. So host memory in flight is a small multiple of this, set
#: by the worker count and the prefetch factor rather than by the
#: epoch, wherever one batch is itself within it. Pinning copies each
#: fetch once more on its way to the consumer.
MAX_FETCH_BYTES = 64 * 1024**2


#: The residues a training read encodes as all-zero columns, as
#: `build_training_loader` states under ``sequences``.
_IGNORED_RESIDUES = tuple("QWERYUIOPSDFHJKLZXVBNM")


class RechunkedLoader(object):
    """Hand out a loader's fetches one training batch at a time.

    Wraps a loader whose own batch size, the fetch size, is a whole
    multiple of the batch a consumer wants, and slices each fetch back
    down in order. The batches are exactly those the same sampler
    yields through a loader of the consumer's batch size, tensor for
    tensor, so only the number of transfers changes.

    Each batch is a `list` holding, in the fetch's own order, one
    slice per tensor of the fetch. Each slice shares its fetch's
    storage at the slice's offset, so nothing is copied, and a slice of
    a pinned fetch lies in pinned memory.

    A pass holds one fetch at a time: it lets go of each before it
    pulls the next, so a fetch outlives that pull only through a batch
    the consumer still holds.

    Iterating starts the wrapped loader exactly once per pass. The
    length is the number of batches a pass yields, given that the
    wrapped loader keeps its last partial fetch. Any other attribute is
    the wrapped loader's, ``dataset`` and ``sampler`` included.

    :param loader: The loader to wrap, as a
        `torch.utils.data.DataLoader` whose ``batch_size`` is the fetch
        size, which keeps its last partial fetch, and whose batches are
        sequences of tensors.
    :param batch_size: The batch to hand the consumer. Kept as the
        attribute of the same name, beside ``fetch_size`` and
        ``loader``.
    :raises ValueError: if ``batch_size`` is below ``1``; if
        ``loader`` has no ``batch_size``, as one batching through a
        ``batch_sampler`` has not; if ``loader`` drops its last partial
        fetch; or if ``batch_size`` does not divide the fetch size.

    .. rubric:: Implementation notes

    A transfer from a worker costs a collate, a pickle and a pipe write
    whatever it carries, so moving several batches at once pays that
    once rather than once per batch, while the consumer still steps on
    the batch size it was tuned for.

    A fetch that is not a whole multiple is refused because it would
    leave short batches in the middle of an epoch, and a training loop
    that skips odd-sized batches would then drop data without saying
    so.

    A loader that drops its last partial fetch is refused because it
    drops up to a fetch of samples where a loader of the consumer's
    batch size drops less than a batch, so the stream would differ and
    the length would overstate it.

    Once per pass matters because a sampler may advance its epoch each
    time it is iterated, as `quadrat.sampling.PeakNegativeMixSampler`
    does, so a wrapper that restarted the loader would step the epoch
    out of time with the consumer.
    """

    def __init__(self, loader, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1, got {}".format(batch_size))
        fetch_size = getattr(loader, "batch_size", None)
        if fetch_size is None:
            raise ValueError(
                "RechunkedLoader slices the fetches of a loader with a "
                "batch_size, and this one has none; it batches through a "
                "batch_sampler, whose fetches need not be one size"
            )
        # Refused for the reason in the notes.
        if getattr(loader, "drop_last", False):
            raise ValueError(
                "RechunkedLoader needs a loader that keeps its last partial "
                "fetch; one with drop_last=True loses up to a fetch of "
                "samples a loader of the batch size would yield"
            )
        if fetch_size % batch_size != 0:
            raise ValueError(
                "fetch size ({}) must be a whole multiple of batch size "
                "({}); anything else leaves short batches mid-epoch".format(
                    fetch_size, batch_size
                )
            )

        self.loader = loader
        self.batch_size = int(batch_size)
        self.fetch_size = int(fetch_size)

    def __getattr__(self, name):
        # Reached only for a name this object does not hold. `loader`
        # is one only while the object is half-built, as under `copy`
        # or `pickle`, and forwarding it would recurse without end.
        if name == "loader":
            raise AttributeError(name)
        return getattr(self.loader, name)

    def __len__(self):
        return -(-len(self.loader.sampler) // self.batch_size)

    def __iter__(self):
        step = self.batch_size
        for fetched in self.loader:
            for start in range(0, len(fetched[0]), step):
                yield [tensor[start : start + step] for tensor in fetched]
            # Let go as the docstring states.
            del fetched


def build_training_loader(
    peaks,
    negatives,
    sequences,
    signals,
    controls=None,
    chroms=None,
    in_window=2114,
    out_window=1000,
    max_jitter=500,
    negative_ratio=0.25,
    reverse_complement=True,
    shuffle=True,
    min_counts=None,
    max_counts=None,
    summits=False,
    exclusion_lists=None,
    random_state=None,
    pin_memory=True,
    num_workers=1,
    batch_size=64,
    verbose=False,
    cache_residency=None,
    cache_dir=None,
    fetch_size=None,
    rank=None,
    world_size=None,
    max_fetch_bytes=MAX_FETCH_BYTES,
    scratch_dir=None,
    timeout=0,
):
    """Build the loader that turns locus files and tracks into training batches.

    Nothing is read densely. The loci become windows, the positions
    the tracks hold inside those windows are collected into a
    `quadrat.cache.WindowSignalCache`, and each sample's tensors are
    assembled when its batch is drawn.

    **Which loci are drawn.** A peak is kept when all of these hold,
    tested in this order, and a negative when the first three do:

    1. its windows lie on its contig, as `quadrat.loci.load_windows`
       marks ``in_bounds``, a peak's being ``max_jitter`` wider on both
       sides and a negative's not;
    2. `quadrat.loci.exclusion_mask` does not flag it;
    3. its count is within ``min_counts`` and ``max_counts`` as
       `quadrat.loci.within_counts` decides, the count being the total
       of the first signal track over its output window, a peak's
       jitter margin included;
    4. `outlier_mask` does not flag it among the peaks still standing,
       on their totals per signal group.

    The first two are the test `quadrat.loci.readable_windows` makes.

    Between the second test and the third,
    `quadrat.loci.require_track_contigs` is applied to the signals and
    to the controls over the contigs of the loci still standing.

    **The persisted entry.** The kept windows and both caches are
    read from an entry under ``cache_dir`` where one matches these
    arguments, as `quadrat.cache.window_cache` reads it, and built and
    written there otherwise. A hit runs none of the tests above and
    answers the batches the build it was written by answers.
    `quadrat.cache.window_cache_key` states which arguments decide the
    entry, and `quadrat.cache.default_cache_dir` that nothing here
    reclaims one.

    **The batches.** Each is the default collation of what
    `quadrat.dataset.LocusDataset` reads, in the order
    `quadrat.sampling.PeakNegativeMixSampler` plans. Given a
    ``random_state``, the sequence of batches is a function of the
    arguments and the epoch alone: it is the same for any
    ``num_workers`` and any ``fetch_size``, and the same again from a
    loader over the returned ``dataset`` and ``sampler`` whose workers
    do not persist. Unsharded, each pass over the loader is the
    next epoch.

    **Fetching in bulk.** Workers hand samples over ``fetch_size`` at
    a time, and a `RechunkedLoader` cuts each fetch back into batches,
    which changes how many transfers there are and nothing about the
    batches. By default the fetch size is what `auto_fetch_size`
    chooses for this loader's draws, its workers, the size
    `sample_nbytes` gives a sample and ``max_fetch_bytes``, which one
    fetch exceeds only where one batch does.

    **Sharding.** Given ``rank`` and ``world_size``, the loader draws
    through a `quadrat.sampling.ShardedEpochSampler` over global batches
    of ``batch_size``, which states what each rank reads and that a
    pass reads the epoch its ``set_epoch`` last named: call it on the
    loader's ``sampler`` before each pass. Without them the loader's
    ``sampler`` is the `quadrat.sampling.PeakNegativeMixSampler`, whose
    ``set_epoch`` resumes or pins an epoch.

    A build needs the ``pileup`` extra and is unsupported concurrently
    in one process, as `quadrat.cache.group_track` states. Reading an
    entry needs neither.

    :param peaks: The peak loci, as `quadrat.loci.load_windows` takes
        them. A frame can be given only with ``cache_dir=False``, an
        entry being keyed by its files.
    :param negatives: The negative loci, likewise, or ``None`` for
        none, which needs a ``negative_ratio`` of ``0``.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`. A letter other than the four bases, in either
        case, reads as an all-zero column; any other character is
        refused, with `ValueError`, by the read of a window that holds
        it, which under workers is a worker's read. Where ``cache_dir``
        is ``False``, a reference with no index and nowhere to write one
        is refused with `OSError` by the first read, as
        `quadrat.stores.SequenceStore` states.
    :param signals: The signal tracks, as
        `quadrat.groups.normalize_signal_groups` takes them, each track
        of any shape `quadrat.stores.track_frame` accepts, an in-memory
        one only with ``cache_dir=False``.
    :param controls: The control tracks, in the same form, or ``None``
        or empty for none.
    :param chroms: The contigs to keep, as `quadrat.loci.load_windows`
        takes them.
    :param in_window: The input window's width, in bases.
    :param out_window: The output window's width, in bases.
    :param max_jitter: The jitter bound, as
        `quadrat.sampling.PeakNegativeMixSampler` takes it, and the
        margin a peak's windows are widened by on both sides.
    :param negative_ratio: Negative draws per peak, per epoch.
    :param reverse_complement: Whether to mirror about half the draws.
    :param shuffle: Whether to permute the peak order every epoch.
    :param min_counts: The lowest count a kept locus may have, or
        ``None`` for no floor.
    :param max_counts: The highest count a kept locus may have, or
        ``None`` for no ceiling.
    :param summits: Whether peaks are centred on their summits, as
        `quadrat.loci.load_windows` takes it. Negatives never are.
    :param exclusion_lists: The excluded regions, as
        `quadrat.loci.exclusion_mask` takes them, or ``None`` to
        exclude nothing. Files only, unless ``cache_dir=False``.
    :param random_state: The sampler's seed, or ``None`` to draw one.
    :param pin_memory: Whether the loader pins the batches it yields.
    :param num_workers: How many worker processes read samples; ``0``
        reads in the calling process. Workers persist between epochs.
    :param batch_size: How many samples a batch holds. With
        ``world_size`` it is the global batch, summed over ranks, and
        not the batch this rank's loader yields, which holds
        ``batch_size // world_size``.
    :param verbose: Whether to print how many loci were kept, or that
        an entry was reused.
    :param cache_residency: Where the caches keep their arrays, as
        `quadrat.cache.WindowSignalCache.build` takes ``residency``.
    :param cache_dir: The directory holding entries, ``None`` for
        `quadrat.cache.default_cache_dir`, or ``False`` to neither
        read nor write one. A build that persists nothing, whether for
        ``False`` or because its entry could not be committed, keeps
        its caches in a scratch directory until the process exits
        where ``cache_residency`` resolves to ``"mapped"``, as
        `quadrat.cache.WindowSignalCache.from_arrays` states.
    :param fetch_size: How many samples a worker hands over at once, a
        whole multiple of the batches the loader yields, or ``None``
        to choose one as above. Equal to that batch, it fetches one
        batch at a time.
    :param rank: This process's rank, counting from ``0``, or ``None``
        not to shard. Given together with ``world_size`` or not at all.
    :param world_size: How many ranks share each global batch, or
        ``None`` not to shard. Above ``1``, every rank has to draw the
        same epochs, so ``random_state`` is required.
    :param max_fetch_bytes: The ceiling on one fetch, in bytes, as
        `auto_fetch_size` takes it, which a fetch size chosen by
        default exceeds only where one batch does. Unused where
        ``fetch_size`` is given.
    :param scratch_dir: Where a build that persists nothing makes its
        scratch directory, as `quadrat.cache.WindowSignalCache.build`
        takes it.
    :param timeout: How long to wait for a fetch from a worker, in
        seconds, as `torch.utils.data.DataLoader` takes it; ``0`` waits
        for as long as it takes.
    :returns: A `RechunkedLoader` whose ``dataset`` is the
        `quadrat.dataset.LocusDataset`, whose ``sampler`` is the
        sampler it draws through, whose ``batch_size`` is the batch it
        yields, this rank's under sharding, and whose length is the
        batches it yields per pass.
    :raises ValueError: if ``signals`` is ``None`` or empty; for a
        group in ``signals`` or ``controls``
        `quadrat.groups.normalize_signal_groups` refuses; if only one of
        ``rank`` and ``world_size`` is given; if ``world_size`` is above
        ``1`` without a ``random_state``; if ``fetch_size`` is not a
        positive whole multiple of the batches the loader yields; for
        anything
        `quadrat.sampling.PeakNegativeMixSampler.validate_arguments`
        refuses of ``negative_ratio`` and ``max_jitter``, ``negatives``
        being ``None`` counting as no negative; for anything
        `quadrat.sampling.ShardedEpochSampler.validate_arguments`
        refuses of ``batch_size``, ``rank`` and ``world_size``, not
        sharding counting as rank ``0`` of a world of ``1``; for a
        ``cache_residency`` `quadrat.cache.resolve_residency` refuses;
        for a ``num_workers`` or ``timeout``
        `torch.utils.data.DataLoader` refuses; if no peak is kept; if
        ``negative_ratio`` is positive and no negative is kept; or for
        anything `quadrat.loci.load_windows` or
        `quadrat.loci.require_track_contigs` refuses by value. Each of
        these refusals but the last three is made before anything is
        read or keyed, and the last three before anything is grouped or
        persisted.
    :raises TypeError: if ``signals`` or ``controls`` is anything but
        ``None``, a `list` or a `tuple`; if a locus set, an exclusion
        list or a track is held in memory while ``cache_dir`` is not
        ``False``; or for anything `quadrat.loci.load_windows` refuses
        by type.
    :raises FileNotFoundError: for a reference, a locus file or an
        exclusion list `quadrat.loci.readable_windows` finds nothing
        at, or a track `quadrat.loci.require_track_contigs` finds
        nothing at, whether or not ``cache_dir`` is ``False``; or for a
        ``scratch_dir`` that does not exist, where a build that persists
        nothing makes its scratch directory there.
    :raises OSError: for a reference `quadrat.cache.window_cache_key`
        refuses, where ``cache_dir`` is not ``False``.
    :raises ImportError: if a build is needed and the ``pileup`` extra
        is not installed.

    .. rubric:: Implementation notes

    Everything that decides how many loci there are happens before the
    first batch, in the calling process, because the sampler sizes
    every draw of an epoch from the peak count. One peak's
    disagreement about a filter would therefore change every batch.

    The outlier test is over peaks only: negatives are drawn from a
    background set, where an extreme locus is not the same kind of
    problem.

    The track check is placed where `quadrat.loci.extract_loci` places
    it, for the reason its notes give.

    A fetch of one batch is still wrapped, its rechunk an identity, so
    the type returned does not depend on the worker count and the
    epoch size that decide the default fetch.

    Workers persist for the cost of starting one rather than for
    correctness, the epoch being the sampler's and the sampler the
    calling process's.

    Sharded, the dataset still holds the
    `quadrat.sampling.PeakNegativeMixSampler`, and resolves each
    ``(epoch, index)`` pair the shard yields against its plan, so the
    shard decides which slots this rank reads and the plan what they
    hold. That is why every rank needs one seed: each plans the epoch
    for itself, and ranks with different seeds would read slices of
    different epochs.
    """

    if (rank is None) != (world_size is None):
        raise ValueError(
            "rank and world_size are given together or not at all; got "
            "rank={!r} and world_size={!r}".format(rank, world_size)
        )
    # Refused for the reason in the notes.
    if world_size is not None and world_size > 1 and random_state is None:
        raise ValueError(
            "a world_size of {} needs a random_state every rank shares; "
            "without one each rank draws its own seed and reads a slice "
            "of a different epoch".format(world_size)
        )
    PeakNegativeMixSampler.validate_arguments(
        negative_ratio, max_jitter, 0 if negatives is None else None
    )
    ShardedEpochSampler.validate_arguments(
        batch_size,
        0 if rank is None else rank,
        1 if world_size is None else world_size,
    )
    step = batch_size // (1 if world_size is None else world_size)
    if fetch_size is not None and (
        isinstance(fetch_size, bool)
        or not isinstance(fetch_size, numbers.Integral)
        or fetch_size < 1
        or fetch_size % step != 0
    ):
        raise ValueError(
            "fetch_size must be a positive whole multiple of the {} samples "
            "a batch holds here, got {!r}".format(step, fetch_size)
        )
    resolve_residency(cache_residency)
    # A loader over nothing, for the refusals `torch.utils.data.DataLoader`
    # makes of these two, so they come before the build and not after.
    torch.utils.data.DataLoader((), num_workers=num_workers, timeout=timeout)
    signal_tracks, signal_groups = normalize_signal_groups(signals)
    control_tracks, control_groups = normalize_signal_groups(controls)
    if not signal_tracks:
        raise ValueError("signals is required; there is nothing to predict")
    if not control_tracks:
        control_tracks, control_groups = None, None
    if negatives is None:
        negatives = []
    # Walked more than once below, by the key and by each locus set.
    if chroms is not None and not isinstance(chroms, (str, bytes)):
        chroms = list(chroms)

    key = None
    if cache_dir is not False:
        key = window_cache_key(
            signal_tracks,
            sequences,
            peaks,
            negatives,
            chroms=chroms,
            in_window=in_window,
            out_window=out_window,
            max_jitter=max_jitter,
            summits=summits,
            exclusion_lists=exclusion_lists,
            min_counts=min_counts,
            max_counts=max_counts,
            signal_groups=signal_groups,
            controls=control_tracks,
            control_groups=control_groups,
        )

    def readable(loci, jitter, centred):
        windows, keep = readable_windows(
            loci,
            sequences,
            in_window,
            out_window,
            chroms=chroms,
            summits=centred,
            max_jitter=jitter,
            exclusion_lists=exclusion_lists,
        )
        return windows.filter(keep)

    def counted(windows):
        if min_counts is None and max_counts is None:
            return windows
        totals = window_group_sums(windows, signal_tracks[:1], [1])[:, 0]
        return windows.filter(within_counts(totals.numpy(), min_counts, max_counts))

    def kept_windows():
        peak_windows = readable(peaks, max_jitter, summits)
        negative_windows = readable(negatives, 0, False)

        on = sorted(
            set(peak_windows["chrom"].to_list())
            | set(negative_windows["chrom"].to_list())
        )
        require_track_contigs(on, signal_tracks, "signals")
        if control_tracks is not None:
            require_track_contigs(on, control_tracks, "controls")

        peak_windows = counted(peak_windows)
        negative_windows = counted(negative_windows)

        outliers = outlier_mask(
            window_group_sums(peak_windows, signal_tracks, signal_groups)
        )
        peak_windows = peak_windows.filter(~outliers.numpy())
        # Refused here, ahead of the grouping and the commit, so a
        # configuration that keeps nothing persists no entry.
        if len(peak_windows) == 0:
            raise ValueError(
                "no peak is kept by the filters; there is nothing to draw from"
            )
        PeakNegativeMixSampler.validate_arguments(
            negative_ratio, max_jitter, len(negative_windows)
        )
        return peak_windows, negative_windows, int(outliers.sum())

    peak_windows, negative_windows, signal_cache, control_cache, n_outliers = (
        window_cache(
            key,
            kept_windows,
            signal_tracks,
            control_tracks,
            cache_dir=None if cache_dir is False else cache_dir,
            residency=cache_residency,
            scratch_dir=scratch_dir,
        )
    )

    if verbose and n_outliers is not None:
        print(
            "\nTraining Peaks: {} ({} dropped as outliers)".format(
                len(peak_windows), n_outliers
            )
        )
        print("Training Negatives: {}".format(len(negative_windows)))
    elif verbose:
        print(
            "\nReusing the window cache at {}".format(
                os.path.join(cache_dir or default_cache_dir(), key)
            )
        )

    sampler = PeakNegativeMixSampler(
        n_peaks=len(peak_windows),
        n_negatives=len(negative_windows),
        negative_ratio=negative_ratio,
        max_jitter=max_jitter,
        reverse_complement=reverse_complement,
        shuffle=shuffle,
        random_state=random_state,
    )
    dataset = LocusDataset(
        peak_windows,
        negative_windows,
        SequenceStore(sequences, ignore=_IGNORED_RESIDUES),
        signal_cache,
        in_window,
        out_window,
        control_cache=control_cache,
        signal_groups=signal_groups,
        control_groups=control_groups,
        sampler=sampler,
    )

    draws = sampler
    if world_size is not None:
        draws = ShardedEpochSampler(
            len(dataset), batch_size, rank=rank, world_size=world_size
        )
    if fetch_size is None:
        fetch_size = auto_fetch_size(
            len(draws),
            step,
            num_workers,
            sample_nbytes(dataset),
            max_fetch_bytes=max_fetch_bytes,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=draws,
        batch_size=int(fetch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        timeout=timeout,
    )
    return RechunkedLoader(loader, step)


def outlier_mask(group_counts, quantile=0.99, scale=1.2):
    """Return which loci are extreme in any one signal group.

    A locus is flagged when its count in some group is strictly above
    ``scale`` times that group's ``quantile`` over all the loci. Each
    group has its own ceiling and the flags are OR-ed over groups.

    :param group_counts: The total signal per locus per group, as a
        floating `torch.Tensor` of shape ``(loci, groups)``, e.g., from
        `quadrat.cache.window_group_sums`.
    :param quantile: The quantile each group's ceiling is taken from,
        in ``[0, 1]``.
    :param scale: The multiplier applied to that quantile.
    :returns: A boolean `torch.Tensor` of shape ``(loci,)``, true where
        the locus is an outlier. Empty for no loci.

    .. rubric:: Implementation notes

    One ceiling over the groups' total would let a high-coverage
    modality set it for a quiet one read beside it, e.g., a stranded
    factor two orders of magnitude above an accessibility track would
    decide the filter for both.
    """

    if group_counts.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool)
    thresholds = torch.quantile(group_counts, quantile, dim=0) * scale
    return (group_counts > thresholds).any(dim=1)


def auto_fetch_size(
    n_draws,
    batch_size,
    num_workers,
    bytes_per_sample,
    prefetch_factor=2,
    max_fetch_bytes=MAX_FETCH_BYTES,
):
    """Return how many samples a loader should fetch per transfer.

    Always a whole multiple of ``batch_size``, so a `RechunkedLoader`
    can slice it back down, and ``batch_size`` itself wherever fetching
    in bulk cannot help: with no workers, there being no transfer to
    amortise, or with too few draws to divide.

    The multiple is the largest that still leaves an epoch
    `FETCHES_PER_PREFETCH_SLOT` fetches for each of the ``num_workers *
    prefetch_factor`` prefetch slots and keeps one fetch within
    ``max_fetch_bytes``. It is never less than ``1``, so a single batch
    larger than the ceiling is still fetched whole.

    :param n_draws: How many samples one epoch holds.
    :param batch_size: The batch the consumer steps on.
    :param num_workers: How many worker processes the loader runs.
    :param bytes_per_sample: The size of one sample's tensors, in
        bytes, as `sample_nbytes` gives it.
    :param prefetch_factor: How many fetches each worker keeps ready,
        as `torch.utils.data.DataLoader` takes it.
    :param max_fetch_bytes: The ceiling on one fetch, in bytes.
    :returns: The fetch size, as an `int`.
    """

    if num_workers <= 0 or n_draws <= 0:
        return batch_size

    n_batches = n_draws // batch_size
    budget = max(1, num_workers * prefetch_factor * FETCHES_PER_PREFETCH_SLOT)
    multiple = max(1, n_batches // budget)

    by_memory = max(1, int(max_fetch_bytes // (bytes_per_sample * batch_size)))
    return int(min(multiple, by_memory) * batch_size)


def sample_nbytes(dataset):
    """Return how many bytes one sample of a dataset occupies once collated.

    Taken from the shapes and dtypes of the tensors
    `quadrat.dataset.LocusDataset` reads rather than from a sample, so
    nothing is read; the label collates to one ``int64``.

    :param dataset: A `quadrat.dataset.LocusDataset` over a
        `quadrat.stores.SequenceStore` of the four bases and caches
        that report ``n_channels``, as `build_training_loader` builds
        one.
    :returns: The size, as an `int`.
    """

    floats = dataset.signal_cache.n_channels * dataset.out_window
    if dataset.control_cache is not None:
        floats += dataset.control_cache.n_channels * dataset.in_window
    return int(
        4 * dataset.in_window * torch.int8.itemsize
        + floats * torch.float32.itemsize
        + torch.int64.itemsize
    )
