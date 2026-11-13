"""The samplers: which draw fills each slot of an epoch, and which rank reads it.

Provides `LocusDraw`, one sampling decision; `EpochPlan`, every
decision of one epoch; `PeakNegativeMixSampler`, which plans an epoch
from a seed; and `ShardedEpochSampler`, which hands one rank its share
of each batch.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`. Importing this module
imports ``torch``, which `import quadrat` does not.
"""

import typing

import numpy
import torch

#: The multiplier that spreads consecutive epochs across the seed
#: space, and the modulus that keeps a seed one
#: `numpy.random.RandomState` accepts. Both are part of the frozen draw
#: sequence `PeakNegativeMixSampler.plan` documents.
_EPOCH_STRIDE = 2654435761
_SEED_MODULUS = 2**31 - 1


class LocusDraw(typing.NamedTuple):
    """Hold one sampling decision: which locus to read, and how.

    The whole of what a sampler tells a dataset. Every field is a plain
    Python scalar.

    :param is_peak: Whether the locus is a peak rather than a
        negative.
    :param index: The locus, as its row in the set ``is_peak`` picks.
    :param jitter: How far the draw is shifted into its window, in
        bases.
    :param reverse_complement: Whether the draw is mirrored.

    .. rubric:: Implementation notes

    Plain scalars rather than numpy ones because a draw crosses into a
    loader worker once per sample, and numpy scalars pickle at several
    times the cost for no gain.

    The jitter is carried as a decision rather than left for the reader
    to redraw, which is what lets a reader hold no random state.
    """

    is_peak: bool
    index: int
    jitter: int
    reverse_complement: bool


class EpochPlan(object):
    """Hold every draw of one epoch, as parallel arrays.

    Slot ``i`` of the epoch is entry ``i`` of each array. Indexing
    answers that slot as a `LocusDraw`, iterating answers every slot in
    order, and the length is the slot count.

    :param epoch: The epoch the plan is for.
    :param peak_ordering: The order the peaks are drawn in, i.e., the
        peak rows in the order the peak slots take them.
    :param is_peak: Whether each slot draws a peak, as a boolean array.
    :param index: The row each slot draws, in the set its ``is_peak``
        picks.
    :param jitter: The shift of each slot's draw, in bases.
    :param reverse_complement: Whether each slot's draw is mirrored, as
        a boolean array.
    """

    __slots__ = (
        "epoch",
        "peak_ordering",
        "is_peak",
        "index",
        "jitter",
        "reverse_complement",
    )

    def __init__(self, epoch, peak_ordering, is_peak, index, jitter, reverse_complement):
        self.epoch = epoch
        self.peak_ordering = peak_ordering
        self.is_peak = is_peak
        self.index = index
        self.jitter = jitter
        self.reverse_complement = reverse_complement

    def __len__(self):
        return len(self.is_peak)

    def __getitem__(self, slot):
        return LocusDraw(
            bool(self.is_peak[slot]),
            int(self.index[slot]),
            int(self.jitter[slot]),
            bool(self.reverse_complement[slot]),
        )

    def __iter__(self):
        for slot in range(len(self)):
            yield self[slot]


class PeakNegativeMixSampler(torch.utils.data.Sampler):
    """Plan an epoch as a mixture of peak and negative draws.

    An epoch draws every peak exactly once and fills
    ``int(n_peaks * negative_ratio)`` further slots with negatives,
    drawn with replacement, the two interleaved at random. The jitter
    and the orientation of every draw are decided here as well, so a
    reader of the draws needs no random state.

    **The plan is a pure function of the seed and the epoch.**
    `plan` answers the same draws for the same ``(random_state,
    epoch)`` whatever was planned before, in any process, so slot ``i``
    of epoch ``e`` is the same draw wherever it is resolved.

    **Iteration.** Starting an iteration advances the epoch by one,
    whether or not the pass is then finished, and the pass yields the
    `LocusDraw` of every slot of the epoch it started at, in order. A
    fresh sampler starts at epoch ``0``. The length is the slot count
    of one epoch.

    Pickles, holding nothing but its arguments, its seed, its epoch,
    and the last plan it made.

    :param n_peaks: How many peaks there are to draw.
    :param n_negatives: How many negatives there are to draw from.
    :param negative_ratio: Negative slots per peak, per epoch. ``0``
        draws no negative.
    :param max_jitter: The jitter bound. A peak's jitter is drawn
        uniformly from ``[0, 2 * max_jitter)``, the range that keeps a
        draw inside a window made ``max_jitter`` wider on both sides,
        and a negative's is always ``0``.
    :param reverse_complement: Whether to mirror about half the draws,
        each chosen independently. ``False`` mirrors none.
    :param shuffle: Whether to permute the peak order every epoch.
        ``False`` draws the peaks in row order; the slots negatives
        take among them are random either way.
    :param random_state: The seed, as an `int`, or ``None`` to draw
        one from numpy's global generator at construction.
    :raises ValueError: for anything `validate_arguments` refuses.

    .. rubric:: Implementation notes

    A sampler rather than state on a dataset, and the distinction is
    load-bearing. A `torch.utils.data.DataLoader` iterates its sampler
    in the parent process, so the epoch counter lives in exactly one
    place and survives workers being torn down. A dataset is copied
    into every worker, so an epoch counter kept there is advanced in
    the copy and discarded with it, and every epoch replays the first
    unless the workers persist.
    """

    def __init__(
        self,
        n_peaks,
        n_negatives,
        negative_ratio=0.1,
        max_jitter=0,
        reverse_complement=False,
        shuffle=True,
        random_state=None,
    ):
        self.validate_arguments(negative_ratio, max_jitter, int(n_negatives))

        self.n_peaks = int(n_peaks)
        self.n_negatives = int(n_negatives)
        self.negative_ratio = negative_ratio
        self.max_jitter = max_jitter
        self.reverse_complement = reverse_complement
        self.shuffle = shuffle

        if random_state is None:
            random_state = int(numpy.random.randint(0, _SEED_MODULUS))
        self._base_seed = int(random_state) % _SEED_MODULUS

        self._epoch = 0
        self._cached = None

    def __len__(self):
        return self.n_peaks + int(self.n_peaks * self.negative_ratio)

    def __iter__(self):
        plan = self.plan(self._epoch)
        self._epoch += 1
        return iter(plan)

    @property
    def epoch(self):
        """Return the epoch the next iteration emits."""

        return self._epoch

    @staticmethod
    def validate_arguments(negative_ratio, max_jitter, n_negatives=None):
        """Refuse the arguments a sampler cannot be constructed from.

        Takes the arguments of the same names, so a caller can refuse
        them before it knows how many loci there are.

        :param negative_ratio: As the class takes it.
        :param max_jitter: As the class takes it.
        :param n_negatives: As the class takes it, or ``None`` where the
            count is not known yet, which skips the one refusal that
            needs it.
        :raises ValueError: if ``max_jitter`` or ``negative_ratio`` is
            negative, or if ``negative_ratio`` is positive while
            ``n_negatives`` is ``0``.

        .. rubric:: Implementation notes

        A positive ratio over no negatives is refused because the draw
        itself would be legal and would answer row ``0`` of an empty
        set, which fails somewhere far from the cause.
        """

        if max_jitter < 0:
            raise ValueError(
                "max_jitter must be non-negative, got {}".format(max_jitter)
            )
        if negative_ratio < 0:
            raise ValueError(
                "negative_ratio must be non-negative, got {}".format(negative_ratio)
            )
        # Refused for the reason in the notes.
        if negative_ratio > 0 and n_negatives == 0:
            raise ValueError(
                "negative_ratio is {} but there are no negatives to draw from".format(
                    negative_ratio
                )
            )

    def set_epoch(self, epoch):
        """Choose the epoch the next iteration emits.

        Needed only to resume a run partway through or to pin one
        epoch, since iterating advances the epoch by itself.

        :param epoch: The epoch, counting from ``0``.
        """

        self._epoch = int(epoch)

    def plan(self, epoch):
        """Return the `EpochPlan` of ``epoch``.

        The purity the class documents is this method's: it reads
        ``epoch``, the seed and the sampler's other arguments, and
        nothing else, and does not move the epoch iteration is at.

        :param epoch: The epoch to plan, counting from ``0``.
        :returns: The plan. The same object is answered while the same
            epoch is asked for again, so its arrays must not be
            written to.

        .. rubric:: Implementation notes

        Epoch ``e`` is drawn from a fresh generator seeded with ``(base
        + e * _EPOCH_STRIDE) % _SEED_MODULUS``, where ``base`` is
        ``random_state``, or the seed drawn in its place, reduced modulo
        ``_SEED_MODULUS``. The formula is frozen with the generator
        calls, for the reason given below: changing either constant, or
        how the base is reduced, changes the epochs a seed draws.

        The sequence of generator calls below is frozen, including the
        branches that skip a call: no permutation without ``shuffle``,
        no jitter draw at a ``max_jitter`` of ``0``, no orientation
        draw without ``reverse_complement``. Each skipped call shifts
        every later one, and every run launched against a seed depends
        on the stream as it is. For the same reason the generator is
        `numpy.random.RandomState`, the legacy stream, and must not
        become `numpy.random.Generator`: the change would alter every
        batch of every run while still looking correct.
        """

        if self._cached is not None and self._cached.epoch == epoch:
            return self._cached

        seed = (self._base_seed + epoch * _EPOCH_STRIDE) % _SEED_MODULUS
        rng = numpy.random.RandomState(seed)
        n = len(self)

        peak_ordering = (
            rng.permutation(self.n_peaks) if self.shuffle else numpy.arange(self.n_peaks)
        )

        is_peak = rng.permutation(numpy.arange(n) < self.n_peaks)

        index = numpy.empty(n, dtype=numpy.int64)
        index[is_peak] = peak_ordering
        index[~is_peak] = rng.randint(
            0, max(1, self.n_negatives), size=int((~is_peak).sum())
        )

        if self.max_jitter > 0:
            jitter = rng.randint(0, self.max_jitter * 2, size=n)
            jitter[~is_peak] = 0
        else:
            jitter = numpy.zeros(n, dtype=numpy.int64)

        if self.reverse_complement:
            reverse_complement = rng.randint(0, 2, size=n).astype(bool)
        else:
            reverse_complement = numpy.zeros(n, dtype=bool)

        self._cached = EpochPlan(
            epoch, peak_ordering, is_peak, index, jitter, reverse_complement
        )
        return self._cached


class ShardedEpochSampler(torch.utils.data.Sampler):
    """Hand one rank its share of every global batch, as ``(epoch, index)`` pairs.

    Global batch ``s`` of an epoch covers the indices ``[s *
    batch_size, (s + 1) * batch_size)``, and rank ``r`` takes the
    ``r``-th contiguous slice of it, ``batch_size // world_size``
    indices long. A trailing partial global batch is dropped. So with a
    ``world_size`` of ``1`` this is the sequential order less the
    partial batch, and for any world size the ranks together read, step
    for step, exactly the indices one rank would.

    Every index is paired with the epoch `set_epoch` last named, ``0``
    until it is called, and iterating does not advance it: the epoch is
    the caller's to set before each pass, as it is for
    `torch.utils.data.distributed.DistributedSampler`. A dataset
    addressed by the pair is therefore told its epoch outright.

    The length is the index count of one rank's epoch.

    :param n: How many indices one epoch holds.
    :param batch_size: The global batch size, summed over ranks.
    :param rank: This process's rank, counting from ``0``.
    :param world_size: How many ranks there are.
    :raises ValueError: for anything `validate_arguments` refuses.
    """

    def __init__(self, n, batch_size, rank=0, world_size=1):
        self.validate_arguments(batch_size, rank, world_size)

        self.n = n
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = batch_size // world_size
        self.epoch = 0

    def __len__(self):
        return (self.n // self.batch_size) * self.local_batch_size

    def __iter__(self):
        epoch = self.epoch
        for step in range(self.n // self.batch_size):
            start = step * self.batch_size + self.rank * self.local_batch_size
            for index in range(start, start + self.local_batch_size):
                yield epoch, index

    @staticmethod
    def validate_arguments(batch_size, rank=0, world_size=1):
        """Refuse the arguments a shard cannot be constructed from.

        Takes the arguments of the same names, so a caller can refuse
        them before it knows how many indices an epoch holds.

        :param batch_size: As the class takes it.
        :param rank: As the class takes it.
        :param world_size: As the class takes it.
        :raises ValueError: if ``batch_size`` or ``world_size`` is less
            than ``1``, if ``world_size`` does not divide
            ``batch_size``, or if ``rank`` is not in
            ``range(world_size)``.

        .. rubric:: Implementation notes

        A rank outside the world is refused because it would be
        answered: its slice starts past its own global batch, so it
        reads indices another step owns and the ranks no longer
        partition the epoch.
        """

        if batch_size < 1 or world_size < 1:
            raise ValueError(
                "batch_size and world_size must be at least 1, got {} and {}".format(
                    batch_size, world_size
                )
            )
        if batch_size % world_size != 0:
            raise ValueError(
                "batch_size ({}) must be divisible by the number of ranks ({})".format(
                    batch_size, world_size
                )
            )
        # Refused for the reason in the notes.
        if not 0 <= rank < world_size:
            raise ValueError(
                "rank must be in range({}), got {}".format(world_size, rank)
            )

    def set_epoch(self, epoch):
        """Choose the epoch every index of the next pass is paired with.

        :param epoch: The epoch, counting from ``0``.
        """

        self.epoch = epoch
