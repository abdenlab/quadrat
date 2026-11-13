"""The dataset: one draw in, one sample's tensors out.

Provides `LocusDataset`.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`. Importing this module
imports ``torch``, which `import quadrat` does not.
"""

import numpy
import torch

from quadrat.cache import joint_windows, require_windows
from quadrat.groups import channel_permutation_from_groups
from quadrat.sampling import PeakNegativeMixSampler


class LocusDataset(torch.utils.data.Dataset):
    """Turn one draw into the tensors of one training sample.

    Indexed by a draw rather than by a position. The draw is either a
    `quadrat.sampling.LocusDraw`, or an ``(epoch, index)`` pair, which
    names slot ``index`` of ``sampler``'s plan for ``epoch`` and reads
    as that slot's draw. A bare integer is refused, since it names
    neither a locus set nor an epoch.

    A read answers ``(sequence, signal, label)``, or ``(sequence,
    control, signal, label)`` with a control cache:

    ``sequence``
        What ``sequence_store`` reads over the draw's input window:
        from a `quadrat.stores.SequenceStore` over the four bases,
        ``int8`` of shape ``(4, in_window)``.

    ``signal``
        What ``signal_cache`` reads over its output window: from a
        `quadrat.cache.WindowSignalCache`, ``float32`` of shape
        ``(channels, out_window)``.

    ``control``
        What ``control_cache`` reads over the input window, in the
        same form at ``in_window`` wide.

    ``label``
        ``1`` for a peak and ``0`` for a negative, as an `int`.

    A draw's window starts ``jitter`` bases into the window its locus
    was given, so a jitter of ``j`` reads the same bases one further
    along than ``j - 1`` does.

    **Reverse-complement.** A mirrored draw reads the same window
    turned around. The sequence is flipped along both axes, which for
    the alphabet ``A, C, G, T`` complements as it reverses. The signal
    equals ``plain[signal_perm].flip(-1)`` of the unmirrored read, and
    the control the same under ``control_perm``; without a grouping
    only the length is flipped.

    **Stateless reads.** No read consults an epoch counter or a random
    state carried between reads: a draw carries every decision, and a
    pair is read at the epoch it names, whatever epoch ``sampler`` has
    reached. The stores keep their handles per process, so a copy in a
    loader worker, pickled or inherited, reads what the original reads.

    The length is ``len(sampler)``, the draws of one epoch, with a
    sampler, and the loci that can be addressed without one.

    :param peak_windows: The peak windows, as a `polars.DataFrame`
        carrying ``chrom`` and ``in_start``, one row per peak, as
        `quadrat.loci.load_windows` derives them.
    :param negative_windows: The negative windows, likewise. They may
        be empty where no negative is drawn. A negative is drawn at a
        jitter of ``0``, so these are windows made with no jitter
        margin.
    :param sequence_store: The sequence reader, i.e., anything
        answering ``window(chrom, start, end)`` as
        `quadrat.stores.SequenceStore` does.
    :param signal_cache: The signal reader, i.e., anything answering
        ``n_channels`` and ``window(index, jitter, length,
        reverse_complement, perm)`` as `quadrat.cache.WindowSignalCache`
        does, built over the `quadrat.cache.joint_windows` of the two
        frames between their ``sig_start`` and ``sig_end``.
    :param in_window: The input window's width, in bases.
    :param out_window: The output window's width, in bases.
    :param control_cache: The control reader, of the same form, built
        over the same windows between ``in_start`` and ``in_end``, or
        ``None`` for no controls.
    :param signal_groups: The size of every signal group in channel
        order, as `quadrat.groups.validate_signal_groups` accepts
        them, or ``None`` to leave the channels alone under a mirror.
        Kept as the `tuple` attribute of the same name, beside
        ``signal_perm``, the ``int64`` tensor
        `quadrat.groups.channel_permutation_from_groups` returns for
        it, or ``None``.
    :param control_groups: The same for the control, kept likewise
        beside ``control_perm``.
    :param sampler: The `quadrat.sampling.PeakNegativeMixSampler` an
        ``(epoch, index)`` pair is resolved against, whose
        ``n_peaks`` and ``n_negatives`` are this dataset's, or
        ``None``. An attribute of the same name, which may be assigned
        later and is checked as it is here.
    :raises TypeError: if either window frame is not a
        `polars.DataFrame`; if ``sampler`` is neither ``None`` nor a
        `quadrat.sampling.PeakNegativeMixSampler`; or for a grouping
        `quadrat.groups.validate_signal_groups` refuses by type.
    :raises ValueError: if ``peak_windows`` is empty; if
        ``control_groups`` is given without ``control_cache``; if a
        grouping's sizes do not sum to its cache's ``n_channels``; if
        ``sampler`` draws from another number of peaks or negatives
        than the frames hold; or for a grouping
        `quadrat.groups.validate_signal_groups` refuses by value.

    .. rubric:: Implementation notes

    Resolving an ``(epoch, index)`` pair here rather than ahead of the
    worker boundary sends two integers to a worker instead of a draw,
    and costs the worker one plan per epoch, which the sampler keeps.

    The readers mirror while they scatter, so they want the scatter
    form of each permutation, i.e., where an input channel goes. A
    permutation from groups is its own inverse, so that is the
    permutation itself. It is handed over only for a mirrored draw and
    not at all where it is the identity, because a reader checks a
    permutation on every read it is given one for.

    The window bounds are unpacked into plain arrays once, because a
    row lookup on a frame costs more than the read it prepares, and
    the contigs as integer codes into a short tuple of names, because a
    `str` per window is a reference count per window, and under
    ``fork`` touching those copies their pages into every worker.

    A sampler is refused where it is given, rather than at the first
    ``(epoch, index)`` pair, which would fail inside a loader worker far
    from the mistake, or, for a sampler sized for other windows, not
    fail at all: it would read negatives as peaks and label them so.
    """

    def __init__(
        self,
        peak_windows,
        negative_windows,
        sequence_store,
        signal_cache,
        in_window,
        out_window,
        control_cache=None,
        signal_groups=None,
        control_groups=None,
        sampler=None,
    ):
        require_windows("LocusDataset", peak_windows)
        require_windows("LocusDataset", negative_windows)
        if len(peak_windows) == 0:
            raise ValueError("peak_windows is empty; there is nothing to draw from")
        if control_groups is not None and control_cache is None:
            raise ValueError("control_groups was supplied but control_cache is None")

        self.sequence_store = sequence_store
        self.signal_cache = signal_cache
        self.control_cache = control_cache
        self.in_window = in_window
        self.out_window = out_window
        self.signal_groups, self.signal_perm = _grouping(
            "signal_groups", signal_groups, signal_cache
        )
        self.control_groups, self.control_perm = _grouping(
            "control_groups", control_groups, control_cache
        )

        self._n_peaks = len(peak_windows)
        self._n_negatives = len(negative_windows)
        self.sampler = sampler
        windows = joint_windows(peak_windows, negative_windows)
        names, codes = numpy.unique(windows["chrom"].to_numpy(), return_inverse=True)
        self._chrom_names = tuple(str(name) for name in names)
        self._chrom_codes = codes.astype(numpy.int32)
        self._in_starts = numpy.asarray(windows["in_start"], dtype=numpy.int64)
        # Handed to the readers as the notes say.
        self._signal_scatter = _scatter_form(self.signal_perm)
        self._control_scatter = _scatter_form(self.control_perm)

    def __len__(self):
        if self.sampler is not None:
            return len(self.sampler)
        return self._n_peaks + self._n_negatives

    def __getitem__(self, draw):
        if isinstance(draw, tuple) and len(draw) == 2:
            if self.sampler is None:
                raise TypeError(
                    "LocusDataset was given an (epoch, index) pair but has "
                    "no sampler to resolve it against; give it one, or "
                    "index it with a LocusDraw."
                )
            epoch, index = draw
            draw = self.sampler.plan(int(epoch))[int(index)]
        elif isinstance(draw, (int, numpy.integer)):
            raise TypeError(
                "LocusDataset is indexed by a LocusDraw or an (epoch, "
                "index) pair, not a bare integer: an integer does not say "
                "which locus set to read from, nor which epoch's plan."
            )

        # Negatives follow the peaks, as `joint_windows` lays them out.
        row = draw.index if draw.is_peak else self._n_peaks + draw.index
        start = int(self._in_starts[row]) + draw.jitter
        mirrored = draw.reverse_complement

        sequence = self.sequence_store.window(
            self._chrom_names[self._chrom_codes[row]], start, start + self.in_window
        )
        if mirrored:
            sequence = torch.flip(sequence, [0, 1])
        signal = self.signal_cache.window(
            row,
            draw.jitter,
            self.out_window,
            reverse_complement=mirrored,
            perm=self._signal_scatter if mirrored else None,
        )
        if self.control_cache is None:
            return sequence, signal, int(draw.is_peak)

        control = self.control_cache.window(
            row,
            draw.jitter,
            self.in_window,
            reverse_complement=mirrored,
            perm=self._control_scatter if mirrored else None,
        )
        return sequence, control, signal, int(draw.is_peak)

    @property
    def n_peaks(self):
        """Return how many peaks there are to draw."""

        return self._n_peaks

    @property
    def n_negatives(self):
        """Return how many negatives there are to draw from."""

        return self._n_negatives

    @property
    def sampler(self):
        """Return the sampler an ``(epoch, index)`` pair is resolved against, or ``None``."""

        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        # Refused here for the reason in the notes.
        if sampler is not None:
            if not isinstance(sampler, PeakNegativeMixSampler):
                raise TypeError(
                    "LocusDataset resolves an (epoch, index) pair through a "
                    "PeakNegativeMixSampler, got a {}; give it one or "
                    "None.".format(type(sampler).__name__)
                )
            if (sampler.n_peaks, sampler.n_negatives) != (
                self._n_peaks,
                self._n_negatives,
            ):
                raise ValueError(
                    "LocusDataset holds {} peaks and {} negatives, and its "
                    "sampler draws from {} peaks and {} negatives; build the "
                    "sampler over the same loci.".format(
                        self._n_peaks,
                        self._n_negatives,
                        sampler.n_peaks,
                        sampler.n_negatives,
                    )
                )
        self._sampler = sampler


def _grouping(label, groups, reader):
    """A grouping as the sizes and the gather-form permutation the dataset keeps.

    Both ``None`` without a grouping. Refuses sizes that do not sum to
    the reader's channel count.
    """

    if groups is None:
        return None, None
    perm = channel_permutation_from_groups(groups)
    if len(perm) != reader.n_channels:
        raise ValueError(
            "{} {} describes {} channels and its cache holds {}.".format(
                label, list(groups), len(perm), reader.n_channels
            )
        )
    return tuple(int(size) for size in groups), perm


def _scatter_form(perm):
    """A permutation as the numpy array a reader is handed, or ``None`` for an identity.

    A copy rather than a view of the tensor: pickling a dataset into a
    worker moves a tensor's storage into shared memory, which would
    leave a view pointing at the freed original.
    """

    if perm is None:
        return None
    perm = perm.numpy().copy()
    if (perm == numpy.arange(len(perm))).all():
        return None
    return perm
