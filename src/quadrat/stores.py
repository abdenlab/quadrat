"""The per-window readers: one-hot sequence, and dense signal.

A store is a reader addressed by coordinate. It holds no data between
calls, opens its handles on first use in the process that uses them,
and answers one window of one interval at a time.

That shape is set by where these are used. A dataset is re-created in
every loader worker, so anything a store keeps is kept once per worker;
and the eager alternative -- pulling whole chromosomes through the
engine and holding the residues -- costs about 248 MB for hg38 chr1
alone, in each of them. See `_ProcessLocalHandles` for what a worker
does to a live file handle, which is the sharper half of the same
problem.

Reading a track and reading a reference are what the base install is
for, so nothing here reaches for the ``pileup`` extra: a store reads
Parquet through polars and sequence through pyfaidx, both base
dependencies, and neither is imported until a store is built.
"""

import os

import numpy
import torch
from tangermeme.utils import one_hot_encode

from quadrat._engine import fspath_or_none, import_on_first_use
from quadrat.tables import TRACK_SUFFIX, require_complete, track_chrom_path


# Defined ahead of the stores that use it, against the public-first
# ordering, because a base class has to exist when the subclass
# statement runs.
class _ProcessLocalHandles(object):
    """Keep a reader's handles from outliving the process that opened them.

    A dataset is re-created in every loader worker -- pickled under
    ``spawn``, inherited copy-on-write under ``fork`` -- and neither
    mode tolerates a live file handle. A ``pyfaidx.Fasta`` cannot be
    pickled at all, so ``spawn`` fails loudly; ``fork`` is the worse of
    the two, because an inherited descriptor appears to work while two
    processes seek the same file out from under each other and neither
    read looks wrong afterwards.

    A polars scan is a plan rather than a descriptor, and current
    polars pickles one, so it is not the same hazard -- the reason it
    is process-local too is that a plan is built for and cached by the
    process that will run it, and shipping a parent's cache into every
    worker pays for plans the worker may never ask for. Upstream
    recorded both as unpicklable; only the first is.

    So handles open on first use, keyed to the pid that opened them,
    and are dropped on the way into a pickle. A subclass keeps
    everything process-local in the mapping `_handles` answers and
    reads nothing else across a fork.

    The two halves of that dropping are each sufficient on their own:
    with the state pruned there is nothing for the restore to clear,
    and with the restore clearing there is nothing stale to read even
    unpruned. Only the pruning is load-bearing for picklability, so
    only the pruning is strictly required -- but a test of the cache
    being fresh passes with either one present, which is worth knowing
    before deleting whichever looks redundant.
    """

    def _handles(self):
        """The current process's handle mapping, emptied if the pid moved."""

        pid = os.getpid()
        local = getattr(self, "_local", None)
        if local is None or local.get("pid") != pid:
            local = self._local = {"pid": pid}
        return local

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = None


class SequenceStore(_ProcessLocalHandles):
    """Read one-hot sequence windows from an indexed FASTA.

    One window per call, holding nothing between them. The reference
    must carry a faidx index beside it; without one every read would
    scan from the start of the contig.

    :param path: The FASTA, as a `str`, `bytes` or `os.PathLike`.
    :param alphabet: The residues that get a row, in row order.
    :param ignore: Residues that encode to an all-zero column. The
        ambiguity codes belong here, so an ambiguous base reads as no
        base rather than as one of them.
    :raises TypeError: if ``path`` is not a path.

    .. rubric:: Implementation notes

    Random access through ``pyfaidx`` is a flat ~34 us per 2 kb window
    and holds nothing between calls. That is not noise beside the
    signal read: the sequence read is channel-independent, so it is
    about 52% of a sample at one channel and still about 39% at twenty,
    where the signal read finally matches it (34.8 us against 36.0 us
    of 89.1 us total).

    Caching the one-hot windows instead would cost on the order of
    1.3 GB for a human peak set, far less than densifying the signal
    would, so it is the better of the two trades if the per-sample read
    ever becomes the limit. Measured, it is not: an epoch is bound by
    the loader's collate and IPC, which saturates at eight workers, and
    all the reading in an epoch is about a quarter of it.
    """

    def __init__(self, path, alphabet=("A", "C", "G", "T"), ignore=("N",)):
        converted = fspath_or_none(path)
        if converted is None:
            raise TypeError(
                "SequenceStore reads sequence from a FASTA: path must be "
                "a str, bytes or os.PathLike, got {}.".format(type(path).__name__)
            )
        self.path = converted
        self.alphabet = list(alphabet)
        self.ignore = list(ignore)
        self._local = None

    def window(self, chrom, start, end):
        """One-hot encode ``[start, end)`` of ``chrom``.

        :returns: A `torch.Tensor` of shape ``(len(alphabet), end -
            start)`` and dtype ``int8``, with an all-zero column
            wherever the residue is one of ``ignore``.
        """

        residues = self._fasta()[str(chrom)][int(start) : int(end)].seq.upper()
        return one_hot_encode(residues, alphabet=self.alphabet, ignore=self.ignore)

    def _fasta(self):
        """This process's open reference, opened on first use."""

        local = self._handles()
        fasta = local.get("fasta")
        if fasta is None:
            pyfaidx = import_on_first_use("pyfaidx")
            fasta = local["fasta"] = pyfaidx.Fasta(self.path)
        return fasta


class TrackStore(_ProcessLocalHandles):
    """Read dense signal windows from a set of tracks, one per channel.

    Each source is one channel and the order is the channel axis, so
    two stores built from the same tracks in different orders are
    different stores.

    Every source is verified complete once, here, rather than per read.
    A truncated channel would otherwise surface as nothing at all: a
    contig whose file never landed reads as a contig with no signal,
    which is indistinguishable from one that is genuinely empty.

    :param sources: One track per channel, each a `str`, `bytes` or
        `os.PathLike`. A single path is refused rather than treated as
        a one-channel set.
    :raises TypeError: if ``sources`` is itself a path, or if any
        element is not one.
    :raises ValueError: if any source does not verify complete. See
        `require_complete`.

    .. rubric:: Implementation notes

    The scan plan for a contig is cached per process. It is a plan and
    not data, and reusing it saves about 20% of a window read
    (0.76 ms to 0.61 ms) now that the read itself is small enough for
    plan construction to show up in it.
    """

    def __init__(self, sources):
        if fspath_or_none(sources) is not None:
            raise TypeError(
                "TrackStore reads one track per channel: sources must be a "
                "sequence of tracks, not a single path. Pass [{!r}] for one "
                "channel.".format(sources)
            )
        self.sources = []
        for index, source in enumerate(sources):
            converted = fspath_or_none(source)
            if converted is None:
                raise TypeError(
                    "TrackStore reads tracks from disk: source {} must be a "
                    "str, bytes or os.PathLike, got {}.".format(
                        index, type(source).__name__
                    )
                )
            require_complete(converted)
            self.sources.append(converted)
        self._local = None

    def __len__(self):
        return len(self.sources)

    @property
    def n_channels(self):
        """How many channels this store reads, which is its source count."""

        return len(self.sources)

    def window(self, chrom, start, end):
        """Dense signal over ``[start, end)`` of ``chrom``, every channel.

        :returns: A `torch.Tensor` of shape ``(n_channels, end -
            start)`` and dtype ``float32``, zero at every position
            where the channel recorded nothing. A contig absent from a
            channel reads as that channel's row of zeros rather than as
            an error, which is routine for the small unplaced contigs.
        """

        polars = import_on_first_use("polars")
        start, end = int(start), int(end)
        out = numpy.zeros((self.n_channels, end - start), dtype=numpy.float32)

        for channel in range(self.n_channels):
            plan = self._frame(channel, chrom)
            if plan is None:
                continue
            hits = plan.filter(
                (polars.col("pos") >= start) & (polars.col("pos") < end)
            ).collect()
            if hits.height:
                out[channel, hits["pos"].to_numpy() - start] = hits["value"].to_numpy()

        return torch.from_numpy(out)

    def _frame(self, channel, chrom):
        """This process's scan plan for one channel's view of one contig."""

        local = self._handles()
        plans = local.setdefault("plans", {})
        key = (channel, str(chrom))
        if key not in plans:
            plans[key] = chrom_plan(self.sources[channel], chrom)
        return plans[key]


def chrom_plan(source, chrom):
    """A scan of one track's ``(pos, value)`` rows for one contig, or ``None``.

    ``None`` means the contig has nothing in this track, which is
    routine for the small unplaced contigs and is not an error: a
    caller leaves that window at zero.

    Both track layouts are read here, and this is the only place either
    is recognised. A track written by this package is a directory, so
    the read opens one contig's file and pays for nothing else. A table
    written before the format became a directory is a single file
    holding every contig, and is filtered instead.

    :param source: The track, as a `str`.
    :param chrom: The contig to read.
    :returns: A polars ``LazyFrame`` selecting ``pos`` and ``value``,
        unsorted, or ``None``.
    """

    polars = import_on_first_use("polars")
    if os.path.isdir(source):
        path = track_chrom_path(source, chrom)
        if not os.path.exists(path):
            return None
        return polars.scan_parquet(path).select("pos", "value")
    return (
        polars.scan_parquet(source)
        .filter(polars.col("chrom") == str(chrom))
        .select("pos", "value")
    )


def chrom_track(source, chrom):
    """One track's sorted ``(pos, value)`` arrays for one contig, or ``None``.

    What `chrom_plan` describes, collected and sorted. ``None`` means
    the contig has nothing in this track, for the reasons stated there.

    Sorted because both callers search it: the grouping takes the rows
    inside a window with two ``searchsorted`` calls, and the group sums
    take a prefix sum over it. The format writes each contig's file
    sorted already, so this is a no-op on a track this package wrote
    and a real sort only on a single-file table it did not.

    :param source: The track, as a `str`.
    :param chrom: The contig to read.
    :returns: A ``(positions, values)`` pair of numpy arrays, or
        ``None`` where the contig has no rows.
    """

    plan = chrom_plan(source, chrom)
    if plan is None:
        return None
    table = plan.sort("pos").collect()
    if table.height == 0:
        return None
    return table["pos"].to_numpy(), table["value"].to_numpy()


def track_frame(source):
    """One track as a single frame covering every contig it holds.

    The whole-table counterpart of `chrom_plan`, for the callers that
    filter on ``chrom`` themselves and want every contig visible at
    once. A directory is globbed back together and verified complete; a
    single-file table is scanned as it is; an in-memory polars frame is
    taken as given, which is what lets a caller go from alignments to
    tensors without a track on disk.

    :param source: A track directory, a single Parquet file, or a
        polars ``DataFrame`` or ``LazyFrame`` carrying ``chrom``,
        ``pos`` and ``value``.
    :returns: A polars ``LazyFrame``.
    :raises TypeError: if ``source`` is none of those shapes.
    :raises ValueError: if a track directory does not verify complete.
    """

    polars = import_on_first_use("polars")
    path = fspath_or_none(source)
    if path is not None:
        if os.path.isdir(path):
            require_complete(path)
            return polars.scan_parquet(os.path.join(path, "*.parquet"))
        return polars.scan_parquet(path)
    if isinstance(source, polars.LazyFrame):
        return source
    if isinstance(source, polars.DataFrame):
        return source.lazy()
    raise TypeError(
        "track_frame reads a track: a {} directory, a single Parquet "
        "file, or an in-memory polars frame with chrom, pos and value "
        "columns. Got {}.".format(TRACK_SUFFIX, type(source).__name__)
    )
