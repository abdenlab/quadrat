"""The sparse window cache, the grouping that fills it, and its entries on disk.

Provides `WindowSignalCache`, `group_track` that produces the arrays a
cache holds, `window_group_sums` for per-window totals over the same
rows, the functions that key, write and read a persisted entry,
`window_cache` that reads an entry or builds it, `joint_windows` that
lays peaks and negatives out as one window set,
`resolve_residency` that decides where a cache keeps its arrays, and
`require_windows`, `track_list` and `window_columns`, the argument
checks and the window decomposition this package's readers of window
sets share.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`.
"""

import atexit
import collections.abc
import functools
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile

import numpy

from quadrat._engine import (
    TRACK_TABLE,
    WINDOWS_TABLE,
    fspath_or_none,
    import_on_first_use,
    item_list,
    registered_frames,
    transpile_intervals,
)
from quadrat.stores import ProcessLocalHandles, SequenceStore, track_frame
from quadrat.tables import (
    MANIFEST_VERSION,
    commit_directory,
    read_manifest,
    require_complete,
    stage_directory,
)

#: Bumped when the layout of an entry changes, or when anything that
#: shapes the cached bytes without being part of the key does. An
#: entry built under a different value is a miss rather than an error,
#: so bumping it retires every entry without touching any of them.
CACHE_VERSION = 2


#: The legal domain of the ``residency`` argument `WindowSignalCache`
#: takes. A module-level name rather than a literal inside the
#: function that validates it, because a caller cannot check an
#: argument against a membership test written somewhere private.
#: ``None`` is also accepted there and is not a member, because it
#: names a resolution rather than a residency -- see
#: `WindowSignalCache.build`.
#:
#: ``"mapped"`` is named for what the arrays are, not for where they
#: sit: a cold build puts them in a scratch directory it owns and a
#: hit maps a persisted entry it must never delete, and both are this
#: residency. ``scratch_dir`` and ``QUADRAT_SCRATCH_DIR`` keep the
#: other word, for the only thing here that is actually scratch.
RESIDENCIES = ("memory", "mapped")


#: Where entries go when no directory is named. Read on every call
#: rather than once, so a process that sets it after importing this
#: package is not left pointed at the real user cache.
_CACHE_DIR_ENV = "QUADRAT_CACHE_DIR"


#: Where a scratch directory is made when none is named. Read on every
#: call, as `_CACHE_DIR_ENV` is and for the same reason. It exists
#: because the platform temporary directory is a tmpfs on many compute
#: nodes -- i.e., RAM -- and an entry is gigabytes, so the mapped
#: residency would silently become the in-memory one plus a filesystem
#: there, inverting the measurement that chose it.
_SCRATCH_DIR_ENV = "QUADRAT_SCRATCH_DIR"


#: The four arrays a grouped result is written as, in the order
#: `group_track` returns them, each with its byte width and whether
#: its length is one more than the window count rather than the site
#: count.
#:
#: One table because it is one file layout. Together with the ``.npy``
#: header -- a fixed 128 bytes for every one of these, because every
#: dtype and shape here is fixed -- a name's width gives the only size
#: that name's file can legitimately have, which is what lets
#: `WindowSignalCache.adopt` refuse a truncated entry without reading
#: a byte of it. The order is `write_cache_arrays`' argument order and
#: has to stay that way.
_ARRAY_WIDTHS = (
    ("offsets", 8, True),
    ("pos", 4, False),
    ("val", 4, False),
    ("chan", 2, False),
)


#: Derived rather than written out again. A name in one list and not
#: the other would be written by nothing and read as absent, or
#: verified by half the mechanism that exists to verify it; deriving
#: makes both impossible rather than merely documented.
_ARRAY_NAMES = tuple(name for name, _, _ in _ARRAY_WIDTHS)


_NPY_HEADER_BYTES = 128

#: The bounds a control is grouped over, i.e., the input window, where
#: `group_track` defaults to the output window a signal is grouped
#: over.
_CONTROL_BOUND_COLUMNS = ("in_start", "in_end")

#: Prefixed to every key. `window_cache_key`'s notes say why.
_KEY_DOMAIN = b"quadrat-window-cache\x00"


class WindowSignalCache(ProcessLocalHandles):
    """Hold signal for one fixed set of windows, read by draw rather than by place.

    A read is a slice and a scatter. The sampler has already chosen the
    window and the jitter offset, and this cache was built for exactly
    those windows, so there is nothing to look up.

    The layout is the one `group_track` returns and documents.

    Never authoritative. A cache depends on the locus set, the contig
    split, the exclusion lists and the jitter bound, and can always be
    rebuilt from the tracks, so a persisted entry is an optimisation
    and a miss is never an error.

    Built through `build`, `from_arrays` or `adopt` rather than
    directly; the constructor takes whatever those three assembled.

    :param offsets: The window boundaries into the other three arrays,
        as `group_track`'s return value lays them out, or ``None``
        when ``mapped_dir`` holds the arrays.
    :param pos: Each row's offset within its window, likewise.
    :param val: Each row's value, likewise.
    :param chan: Each row's channel index, likewise.
    :param n_channels: How many channels a read answers with, which
        the arrays cannot recover themselves.
    :param mapped_dir: A directory holding the four arrays, which
        every process memory-maps, or ``None`` to hold them in memory.
        Named for what it is rather than for one of its two sources:
        `from_arrays` makes a scratch directory and owns it, `adopt`
        is handed a persisted entry and must never delete it, and both
        arrive here.
    :param meta: What a reader may want to know without touching the
        arrays, or ``None``.

    .. rubric:: Implementation notes

    Looking nothing up also means there is no binary search to degrade
    once the data outgrows page cache, as a search over one large
    sorted array does.

    Sparse because the dense form does not fit. One entry per recorded
    position costs about 5.6 GB of proportional set size over a whole
    process tree for 57,742 windows of 2,000 bp over 68 channels, where
    the dense ``windows x span x channels`` array for the same inputs
    is 31.4 GB. That is also what makes the cost insensitive to channel
    count: pooling changes how positions are labelled rather than how
    many there are, so ten channels cost 4.86 GB against sixty-eight at
    5.71 GB, where dense would be 4.6 GB against 31.4 GB.
    """

    def __init__(
        self,
        offsets=None,
        pos=None,
        val=None,
        chan=None,
        n_channels=None,
        mapped_dir=None,
        meta=None,
    ):
        self.mapped_dir = mapped_dir
        self.meta = meta
        self._offsets = offsets
        self._pos = pos
        self._val = val
        self._chan = chan
        self._n_channels = n_channels
        self._local = None

    @property
    def n_channels(self):
        """Return how many channels a read of this cache answers with."""

        return int(self._n_channels)

    @classmethod
    def build(
        cls,
        windows,
        tracks,
        bound_columns=("sig_start", "sig_end"),
        residency=None,
        scratch_dir=None,
    ):
        """Group ``tracks`` over ``windows`` and hold the result.

        The grouping is `group_track`'s, and as it states needs the
        ``pileup`` extra and is unsupported concurrently in one process.

        :param windows: As `group_track` takes it.
        :param tracks: One track per channel, in channel order, as
            `group_track` takes them.
        :param bound_columns: The window's start and end column names.
        :param residency: ``"memory"``, ``"mapped"``, or ``None`` to
            choose from the multiprocessing start method. ``None``
            resolves to ``"memory"`` under ``fork`` and to ``"mapped"``
            under any other start method, and a program that has not
            yet chosen one resolves as the platform's default method
            would. The choice is made on every call, so a start method
            set after this package was imported is the one that counts.
        :param scratch_dir: Where a scratch directory is made, or
            ``None`` to read ``QUADRAT_SCRATCH_DIR`` and fall back to
            the platform's temporary directory -- which on many compute
            nodes is a tmpfs not sized for a cache.
        """

        tracks = track_list("WindowSignalCache.build", "tracks", tracks)
        return cls.from_arrays(
            group_track(windows, tracks, bound_columns=bound_columns),
            residency=residency,
            scratch_dir=scratch_dir,
        )

    @classmethod
    def from_arrays(cls, arrays, residency=None, scratch_dir=None):
        """Hold arrays `group_track` has already produced.

        Separate from `build` so a caller that wants the arrays
        themselves (e.g., to persist them) does not group twice to also
        get a cache.

        **A scratch directory's lifetime is the building process's.**
        It is removed when that process exits, not when the cache is
        collected, so dropping the cache while workers still hold
        copies of it is safe and a caller has nothing to keep alive.
        The cost of that is the other direction: a process that builds
        many scratch-resident caches holds every directory until it
        ends, so a long-lived process building caches in a loop should
        persist them instead.

        For the multiprocess case, prefer `write_window_cache` and
        `adopt`, which write the bytes once where persisting a
        scratch-resident cache writes them twice; `adopt` states what it
        verifies and what it never deletes, and the example below shows
        the call.

        :param arrays: The five values `group_track` returns, or five
            a caller assembled to the same invariants: ``pos``, ``val``
            and ``chan`` of one length, ``offsets`` non-decreasing from
            zero to that length, and every ``chan`` inside
            ``range(n_channels)``.
        :param residency: As `build` takes it.
        :param scratch_dir: As `build` takes it.
        :raises ValueError: if the arrays break one of those
            invariants.

        Persisting and adopting instead::

            root = write_window_cache(cache_dir, key, ...)
            signal = WindowSignalCache.adopt(
                os.path.join(root, "signal"), n_channels, meta
            )

        where ``meta`` is what `array_meta` returned for those arrays,
        which `read_window_cache` reads back out of the entry's
        record.

        .. rubric:: Implementation notes

        The directory is reaped by an `atexit` hook registered against
        the building process, which is what keeps the API design
        guide's rule that a value must not own a resource: the cache is
        a value, and nothing about its lifetime decides the
        directory's. The guide's own remedy -- a caller-supplied
        directory this function never creates -- would have cost more
        than it bought: ``residency`` resolves to ``"mapped"`` whenever
        the start method is not ``fork``, so requiring the directory
        would make the two-argument `build` raise by default, which
        costs the trivial case the ceremony the same guide forbids it.

        Ownership is one process's, as `_reap_scratch` documents. What a
        hook cannot cover is a process that never exits normally (e.g.,
        ``SIGKILL``, an OOM kill, ``os._exit``), and for those the
        directory outlives the run. That is a leak of scratch space rather than a wrong
        answer, and the same is true of the entry directory, whose
        growth `default_cache_dir` documents.
        """

        resolved = resolve_residency(residency)
        offsets, pos, val, chan, n_channels = arrays
        _refuse_inconsistent_arrays(offsets, pos, val, chan, n_channels)
        if resolved == "memory":
            return cls(
                offsets=offsets,
                pos=pos,
                val=val,
                chan=chan,
                n_channels=n_channels,
                meta=array_meta(arrays),
            )

        if scratch_dir is None:
            scratch_dir = os.environ.get(_SCRATCH_DIR_ENV) or None
        root = tempfile.mkdtemp(prefix="quadrat-cache-", dir=scratch_dir)
        # Registered before the arrays are written, not after the cache
        # exists, so that every exit path from here on is covered: the
        # write failing, `array_meta` failing, or the process ending
        # normally with the cache still alive. A `weakref.finalize`
        # cannot stand here, having nothing yet to watch.
        #
        # The pid makes ownership this process's; see `_reap_scratch`.
        #
        # A `functools.partial` and not ``atexit.register(_reap_scratch,
        # root, pid)``, because the handler below has to be removable
        # on its own: `atexit.unregister` matches on the function and
        # ignores the arguments, so unregistering the bare function
        # would drop the handler of every *other* scratch directory
        # this process still owns. Partials compare by identity, so one
        # per call is precisely removable.
        reap = functools.partial(_reap_scratch, root, os.getpid())
        atexit.register(reap)
        try:
            write_cache_arrays(root, offsets, pos, val, chan)
            cache = cls(mapped_dir=root, n_channels=n_channels, meta=array_meta(arrays))
        except BaseException:
            # Swept now rather than left to the hook above, which would
            # also do it but not until the process ended. A full disk
            # is the realistic way to get here, and that is the worst
            # moment to hold gigabytes until exit.
            atexit.unregister(reap)
            shutil.rmtree(root, ignore_errors=True)
            raise
        return cache

    @classmethod
    def adopt(cls, root, n_channels, meta):
        """Read arrays already on disk without taking responsibility for them.

        The persistent-entry constructor. Nothing here registers
        ``root`` for reaping and nothing here deletes it, so the
        directory outlives every cache that reads it.

        The arrays are verified against ``meta`` before any of them is
        mapped, so a truncated entry or one numpy refuses the header of
        is refused by this call rather than by the first draw.

        :param root: The directory holding the four arrays.
        :param n_channels: The channel count the arrays were grouped
            at, which they cannot recover themselves.
        :param meta: The counts the entry recorded, as `array_meta`
            returns them.
        :raises ValueError: if ``meta`` carries no array counts, if an
            array's size disagrees with them, or if numpy refuses an
            array's header.
        :raises OSError: if the directory is missing an array.

        .. rubric:: Implementation notes

        An entry is read by every run whose key matches it, so a cache
        that deleted its directory would have the first run to finish
        delete the entry every later run is counting on.

        ``meta`` is required rather than optional because an `adopt`
        that cannot verify is one whose result a caller has no way to
        trust.
        """

        directory = str(root)
        _refuse_wrong_sizes(directory, meta)
        # Mapped now, for the refused-by-this-call guarantee above.
        for name in _ARRAY_NAMES:
            numpy.load(os.path.join(directory, name + ".npy"), mmap_mode="r")
        return cls(mapped_dir=directory, n_channels=int(n_channels), meta=meta)

    def nbytes(self):
        """Return how many bytes the four arrays occupy."""

        return int(sum(array.nbytes for array in self._arrays()))

    def window(self, index, jitter, length, reverse_complement=False, perm=None):
        """Return the dense signal for window ``index`` at ``jitter``, every channel.

        :param index: The window, as its row in the set this cache was
            built for, counting from ``0``. A negative index is neither
            supported nor refused.
        :param jitter: How far the draw is shifted. A position shifted
            outside ``[0, length)`` is dropped, which is what makes a
            window near a contig edge read as zeros there rather than
            raise.
        :param length: The span to answer, in bases.
        :param reverse_complement: Whether to mirror the draw.
        :param perm: The channel permutation a mirrored draw applies,
            or ``None`` to leave channels alone. A stranded pair swaps
            under reverse-complement; an unstranded channel does not.
            It must be a permutation of ``range(n_channels)``.
        :returns: A `torch.Tensor` of shape ``(n_channels, length)``
            and dtype ``float32``. Two rows landing on one position --
            which a mirrored draw can also produce from a malformed
            track -- are added, as every other reader of a track adds
            them.
        :raises ValueError: if ``perm`` is given and is not a
            permutation of ``range(n_channels)``, whether or not the
            draw is mirrored.
        :raises IndexError: if ``index`` is past the last window.

        .. rubric:: Implementation notes

        ``perm`` is checked rather than trusted because a mirrored draw
        would otherwise answer for almost every wrong permutation: a
        negative entry indexes from the end, a repeated entry sums two
        channels into one row and leaves another empty, and a long one
        has its extra entries ignored. Only an entry past the last
        channel, or a permutation too short for a channel the window
        holds, can raise, and then from numpy, naming nothing.

        Every channel scatters in one vectorised write, which the layout
        exists to allow; `group_track`'s notes give the cost of the
        alternative.

        The reverse-complement is applied here, by writing each row to
        its mirrored offset and permuted channel, rather than afterwards
        by flipping the dense result. Afterwards means two passes over
        ``(channels, length)`` floats for the sake of the hundred or so
        rows per channel that are not zero.
        """

        torch = import_on_first_use("torch")
        if perm is not None:
            perm = numpy.asarray(perm)
            # Checked rather than trusted; the notes say why.
            if perm.shape != (self.n_channels,) or not numpy.array_equal(
                numpy.sort(perm), numpy.arange(self.n_channels)
            ):
                raise ValueError(
                    "perm must be a permutation of range({}), got {!r}.".format(
                        self.n_channels, numpy.asarray(perm).tolist()
                    )
                )

        offsets, pos, val, chan = self._arrays()
        out = numpy.zeros((self.n_channels, length), dtype=numpy.float32)

        first = int(offsets[index])
        last = int(offsets[index + 1])
        if last > first:
            offset = numpy.asarray(pos[first:last]) - jitter
            keep = (offset >= 0) & (offset < length)
            offset = offset[keep]
            channels = numpy.asarray(chan[first:last])[keep]
            if reverse_complement:
                offset = (length - 1) - offset
                if perm is not None:
                    channels = perm[channels]
            # `add.at` rather than an assignment, so two rows on one
            # position sum instead of one of them winning. Unbuffered,
            # and proportional to the rows in the run rather than to
            # the span, which is what keeps the single vectorised write
            # above worth having.
            #
            # Over a flat view with one computed index rather than over
            # `out` with a `(channels, offset)` pair. The two write the
            # same cells, sum duplicates alike and refuse the same
            # out-of-range channel; the pair makes numpy build a
            # multi-dimensional iterator per call, which measured about
            # four times the cost at 68 channels.
            numpy.add.at(
                out.reshape(-1),
                channels.astype(numpy.intp) * length + offset,
                numpy.asarray(val[first:last])[keep],
            )

        return torch.from_numpy(out)

    def _arrays(self):
        """The four arrays, memory-mapped per process when on disk."""

        if self.mapped_dir is None:
            return self._offsets, self._pos, self._val, self._chan

        local = self._handles()
        arrays = local.get("arrays")
        if arrays is None:
            arrays = local["arrays"] = tuple(
                numpy.load(os.path.join(self.mapped_dir, name + ".npy"), mmap_mode="r")
                for name in _ARRAY_NAMES
            )
        return arrays


def default_cache_dir():
    """Return where persisted entries go when the caller names no directory.

    Resolved from the environment on every call. ``QUADRAT_CACHE_DIR``
    wins, then the XDG convention, then a home-relative default -- so
    caching needs no configuration and relocating it needs one
    variable.

    **Nothing in this package reclaims what accumulates here.** An
    entry is keyed by the whole of what decides its contents, window
    geometry included, so a sweep over geometries mints a fresh entry
    per configuration and keeps every one of them; each is the size of
    the grouped arrays, which is gigabytes at a realistic locus set.
    There is no eviction, no age limit and no size cap, deliberately:
    a retention policy for someone else's disk is the caller's to set,
    and a sweep that deletes gigabytes should not be something a
    library does on its own initiative. A caller who needs the space
    bounded should point ``QUADRAT_CACHE_DIR`` somewhere they manage
    and delete entry directories outright -- each is self-contained,
    and a missing one is a miss rather than an error.
    """

    named = os.environ.get(_CACHE_DIR_ENV)
    if named:
        return os.path.join(os.path.expanduser(named), "windows")
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = (
        os.path.expanduser(xdg)
        if xdg
        else os.path.join(os.path.expanduser("~"), ".cache")
    )
    return os.path.join(root, "quadrat", "windows")


def window_cache_key(
    signals,
    sequences,
    peaks,
    negatives,
    chroms=None,
    in_window=2114,
    out_window=1000,
    max_jitter=500,
    summits=False,
    exclusion_lists=None,
    min_counts=None,
    max_counts=None,
    signal_groups=None,
    controls=None,
    control_groups=None,
    bound_columns=("sig_start", "sig_end"),
):
    """Return a content hash of everything that decides what an entry holds.

    Taken from the inputs and never from anything derived, so that a
    hit costs no computation to recognise.

    What is deliberately *not* in it is as much the point: the batch
    size, the worker count, the shuffle and its seed, the negative
    ratio and the reverse-complement flag all change what a run draws
    and none of them change what the cache holds, so runs differing
    only in those share one entry.

    A stale hit would train on the wrong data and say nothing, which is
    worse than rebuilding every time. So this is over-specified on
    purpose: every locus file is hashed by content, and the order of
    ``signals`` and ``controls`` is part of the identity because it is
    the channel axis.

    Every input is a file, and one held in memory, whether a locus
    set, an exclusion list or a track, is refused rather than keyed.

    **The reference is indexed before it is hashed.** A reference with
    no FASTA index has one written beside it, as
    `quadrat.stores.SequenceStore.contig_lengths` writes it, so the key
    is the one a build over the reference, which indexes it, leaves
    behind. A reference that exists and cannot be indexed, nothing
    being writable beside it, is refused, since a training read takes
    its sequence through the index. A reference that does not exist is
    keyed by its path and an absent index, and left for whatever reads
    it to refuse.

    :param signals: The signal tracks, in channel order, as
        `track_list` takes them, each a path. A single path is refused
        rather than read as one track.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`, indexed as stated above.
    :param peaks: The peak locus file, as a `str`, `bytes` or
        `os.PathLike`, or a `list` or `tuple` of them. A single path
        is one file, never its characters. ``None`` is no file.
    :param negatives: The background locus file or files, as
        ``peaks``.
    :param chroms: The contig subset, or ``None`` for all.
    :param in_window: The input window's width, in bases.
    :param out_window: The output window's width, in bases.
    :param max_jitter: The jitter margin, in bases, as
        `quadrat.loci.load_windows` takes it.
    :param summits: Whether the windows are centred on the peaks'
        summits.
    :param exclusion_lists: The exclusion locus file or files, as
        ``peaks``. Each is hashed by content, as a peak file is.
    :param min_counts: The lower count threshold, or ``None``. A
        number, a per-group sequence, or anything `numpy.asarray`
        reads as either, so a threshold taken from
        `window_group_sums` is accepted as it comes.
    :param max_counts: The upper count threshold, as ``min_counts``.
    :param signal_groups: How the signal channels are grouped, or
        ``None``.
    :param controls: The control tracks, as ``signals``, or ``None``.
    :param control_groups: How the control channels are grouped, as
        ``signal_groups``.
    :param bound_columns: The window's start and end column names, as
        `group_track` takes them. Part of the key because it decides
        which window the signal is grouped over, so two entries built
        from the same inputs over different windows are different
        entries.
    :returns: The key, as a hex `str`.
    :raises TypeError: if ``sequences`` is not a path; if a locus set,
        an exclusion list or a track is given as anything but a path; or
        if ``signals`` or ``controls`` is anything `track_list` refuses.
    :raises OSError: if the reference exists and has no usable index,
        and none can be written beside it, or cannot be read.

    .. rubric:: Implementation notes

    A track is hashed by its record and a stat of every Parquet rather
    than by its bytes: a track runs to gigabytes and hashing it would
    cost more than the work being cached. The record already carries
    the row count of every file and the pileup arguments that produced
    them, so a regenerated track almost always changes it, and the
    sizes and modification times catch one rewritten to the same shape.

    The reference is hashed through its index, not its sequence --
    tens of kilobytes against gigabytes -- and `chrom_lengths` reads
    the index too, so it is the part that decides anything here.

    It is indexed first because the build writes an index where there
    is none, after the key is taken: a key over the absent index would
    file the first build's entry under a key no later run, which finds
    the index, ever takes again.

    The domain tag `_KEY_DOMAIN` names this package, so two packages
    that group the same tracks the same way never read each other's
    entries. That is the deliberate opposite of the track format, which
    is an interoperability contract: a track costs minutes to rebuild
    and is shared on purpose, while an entry costs a small fraction of
    that, so sharing one buys nothing and risks serving bytes another
    package's code laid out.
    """

    reference = fspath_or_none(sequences)
    if reference is None:
        raise TypeError(
            "window_cache_key hashes the reference it was given: "
            "sequences must be a str, bytes or os.PathLike, got "
            "{}.".format(type(sequences).__name__)
        )

    digest = hashlib.sha256()
    digest.update(_KEY_DOMAIN)
    digest.update(str(CACHE_VERSION).encode())

    for label, loci in (("peaks", peaks), ("negatives", negatives)):
        digest.update(label.encode())
        for path in _key_paths(label, [] if loci is None else item_list(loci)):
            digest.update(os.path.abspath(path).encode())
            _hash_file(digest, path)

    digest.update(b"sequences")
    digest.update(os.path.abspath(reference).encode())
    # Indexed first, as the docstring states.
    if os.path.exists(reference):
        SequenceStore(reference).contig_lengths()
    # The index rather than `reference` itself, for the reason in the
    # notes, and through the normalized path: a name spelled with `str`
    # from a `bytes` reference is a repr naming no file, and every
    # absent index hashes alike, so two genomes would collide.
    _hash_file(digest, reference + ".fai")

    digest.update(
        json.dumps(
            {
                "chroms": list(chroms) if chroms is not None else None,
                "in_window": int(in_window),
                "out_window": int(out_window),
                "max_jitter": int(max_jitter),
                "summits": bool(summits),
                "min_counts": _threshold(min_counts),
                "max_counts": _threshold(max_counts),
                "signal_groups": list(signal_groups) if signal_groups else None,
                "control_groups": list(control_groups) if control_groups else None,
                # In the key for the reason its ``:param:`` gives.
                "bound_columns": list(bound_columns),
            },
            sort_keys=True,
        ).encode()
    )

    digest.update(b"exclusion")
    excluded = [] if exclusion_lists is None else item_list(exclusion_lists)
    for path in _key_paths("exclusion_lists", excluded):
        digest.update(os.path.abspath(path).encode())
        _hash_file(digest, path)

    for label, tracks in (("signals", signals), ("controls", controls)):
        digest.update(label.encode())
        listed = [] if tracks is None else track_list("window_cache_key", label, tracks)
        for path in _key_paths(label, listed):
            _hash_track(digest, path)

    return digest.hexdigest()


def read_window_cache(cache_dir, key, residency=None):
    """Return one persisted entry, or ``None`` where there is no verifiable one.

    Anything unreadable, unverifiable, or built under another version
    is reported as a miss, so the caller rebuilds rather than trusts
    it. A miss is not an error: the entry is derived and rebuilding is
    the correct answer to every doubt about it.

    :param cache_dir: The directory holding entries.
    :param key: The key, from `window_cache_key`.
    :param residency: As `WindowSignalCache.build` takes it. A hit
        answers in the residency a cold build of the same arguments
        would have chosen, so the two paths are indistinguishable
        downstream.
    :returns: ``(peak_windows, negative_windows, signal_cache,
        control_cache)``, the last ``None`` where the entry carries no
        controls; or ``None`` for a miss.
    """

    polars = import_on_first_use("polars")
    root = os.path.join(cache_dir, key)

    try:
        # The record's shape, the listed files' presence and the schema
        # version are `require_complete`'s to check; `write_window_cache`
        # documents the record an entry carries. Its refusal is
        # converted rather than raised, because this answers a miss
        # where it answers an error -- and every message it emits talks
        # about a track and tells the caller to pile up the alignments
        # again, which is the wrong instruction for an entry and must
        # not reach anyone.
        #
        # `PolarsError` is in the catch and not implied by the other
        # two: a truncated Parquet raises `ComputeError`, which derives
        # straight from `Exception`, so the obvious tuple lets a
        # corrupt window table out of here as an exception rather than
        # answering the miss this function promises.
        require_complete(root)
        manifest = read_manifest(root)
        if manifest.get("cache_version") != CACHE_VERSION:
            return None
        # The key the entry claims, against the one asked for, so a
        # directory that was moved or renamed is a miss rather than a
        # silently wrong hit.
        if manifest.get("key") != key:
            return None
        peaks = polars.read_parquet(os.path.join(root, "peaks.parquet"))
        negatives = polars.read_parquet(os.path.join(root, "negatives.parquet"))
        # Inside the block with everything else that can be wrong about
        # an entry, because the two caches are built from fields
        # `require_complete` does not know exist. It verifies the files
        # the record lists; it cannot tell that a record omits a channel
        # count, carries one that is not a count, or declares controls
        # whose arrays were never written, and each of those has to
        # answer a miss like every other doubt about an entry.
        signal = _adopted(root, "signal", manifest, residency=residency)
        control = (
            None
            if manifest.get("control_channels") is None
            else _adopted(root, "control", manifest, residency=residency)
        )
    except (OSError, ValueError, polars.exceptions.PolarsError):
        return None

    return peaks, negatives, signal, control


def write_window_cache(
    cache_dir, key, peak_windows, negative_windows, signal_arrays, control_arrays=None
):
    """Commit one entry atomically and return its root, or ``None``.

    Every failure is swallowed, which is the contract rather than an
    oversight: a cache that cannot be written is a slower run, and an
    unwritable directory must not take a training job down.

    The entry is a directory named ``key`` under ``cache_dir``, holding
    ``peaks.parquet``, ``negatives.parquet``, and the four grouped
    arrays under ``signal`` and, with controls, under ``control``. Its
    record is a JSON object beside it, at `manifest_path` of the entry,
    with exactly these fields:

    ``version``
        `MANIFEST_VERSION`, the schema version a track's record carries.

    ``files``
        Every file the entry holds, as a path relative to its root,
        each mapped to ``null``: the record lists files without counts.

    ``cache_version``
        The `CACHE_VERSION` the entry was written under.

    ``key``
        The key the entry was written under.

    ``signal_channels``
        The signal channel count, the fifth value of ``signal_arrays``.

    ``control_channels``
        The control channel count, or ``null`` without controls.

    ``signal_meta``
        The signal arrays' counts, as `array_meta` returns them. An
        adopted cache is verified against them and carries them as its
        ``meta``.

    ``control_meta``
        The same for the control arrays, or ``null`` without controls.

    It carries no ``rows``, which is what lets `require_complete`
    accept files listed without counts, and no ``parameters``.

    :param cache_dir: The directory to hold the entry. Created if
        absent.
    :param key: The key, from `window_cache_key`.
    :param peak_windows: The peak window frame, as a `polars.DataFrame`.
    :param negative_windows: The background window frame, as a
        `polars.DataFrame`.
    :param signal_arrays: The five values `group_track` returns.
    :param control_arrays: The same for the controls, or ``None``.
    :returns: The entry root, or ``None`` if anything failed.

    .. rubric:: Implementation notes

    Commits through `stage_directory` and `commit_directory`, whose
    docstrings carry the ordering argument and what each crash point
    leaves behind.

    The window frames are written with polars rather than pandas.
    pandas reaches Parquet through pyarrow, and pyarrow arrives in this
    tree only under the ``pileup`` extra, so through pandas an entry
    could be neither written nor read on a base install, and because
    this swallows its failures that would be a silent permanent miss
    rather than an error.
    """

    root = os.path.join(cache_dir, key)
    staging = None
    try:
        # The cache directory itself is not created here: `makedirs` on
        # the staging path creates every parent it needs, so a cache
        # directory several levels below anything that exists comes
        # into being with the first entry written into it.
        staging = stage_directory(root)
        os.makedirs(os.path.join(staging, "signal"))

        peak_windows.write_parquet(os.path.join(staging, "peaks.parquet"))
        negative_windows.write_parquet(os.path.join(staging, "negatives.parquet"))
        write_cache_arrays(os.path.join(staging, "signal"), *signal_arrays[:4])
        files = ["peaks.parquet", "negatives.parquet"]
        files += [os.path.join("signal", name + ".npy") for name in _ARRAY_NAMES]
        if control_arrays is not None:
            os.makedirs(os.path.join(staging, "control"))
            write_cache_arrays(os.path.join(staging, "control"), *control_arrays[:4])
            files += [os.path.join("control", name + ".npy") for name in _ARRAY_NAMES]

        commit_directory(
            root,
            staging,
            {
                # The entry record this function's docstring documents.
                "version": MANIFEST_VERSION,
                "files": {name: None for name in files},
                "cache_version": CACHE_VERSION,
                "key": key,
                "signal_channels": int(signal_arrays[4]),
                "control_channels": (
                    None if control_arrays is None else int(control_arrays[4])
                ),
                "signal_meta": array_meta(signal_arrays),
                "control_meta": (
                    None if control_arrays is None else array_meta(control_arrays)
                ),
            },
        )
        return root
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        return None


def window_cache(
    key,
    windows_factory,
    signals,
    controls=None,
    cache_dir=None,
    residency=None,
    bound_columns=("sig_start", "sig_end"),
    scratch_dir=None,
):
    """Return the entry ``key`` names, read where one is persisted and built where not.

    A verifiable entry under ``key`` is answered as `read_window_cache`
    answers it, and ``windows_factory`` is not called. Otherwise the
    two window frames are taken from ``windows_factory``, the tracks
    are grouped over their `joint_windows`, and the result is committed
    as `write_window_cache` commits it. A build answers the values a
    later hit on its entry answers, in the residency ``residency``
    resolves to: ``"memory"`` holds the arrays it grouped, and
    ``"mapped"`` maps the committed entry as a hit would, or, where the
    commit fails or ``key`` is ``None``, a scratch directory as
    `WindowSignalCache.from_arrays` makes one. Nothing is persisted
    where the commit fails.

    The signals are grouped over each window's ``bound_columns`` and
    the controls over its ``in_start`` and ``in_end``.

    A hit needs nothing of the ``pileup`` extra. A build needs it, and
    is unsupported concurrently in one process, as `group_track` states.

    Whatever ``windows_factory`` or `group_track` raises on a build
    propagates.

    :param key: The key, from `window_cache_key` over the same
        ``bound_columns``, or ``None`` to neither read nor write an
        entry.
    :param windows_factory: A callable taking no argument and returning
        ``(peak_windows, negative_windows, report)``: two frames with
        the columns `quadrat.loci.load_windows` returns, and whatever
        the caller wants back from a build. Called once, and only where
        no entry is read.
    :param signals: One track per signal channel, in channel order, as
        `group_track` takes them. Read only where no entry is.
    :param controls: The same for the controls, or ``None``.
    :param cache_dir: The directory holding entries, or ``None`` for
        `default_cache_dir`. Unused where ``key`` is ``None``.
    :param residency: As `WindowSignalCache.build` takes it.
    :param bound_columns: The window's start and end column names the
        signals are grouped over, as `group_track` takes them.
    :param scratch_dir: As `WindowSignalCache.build` takes it, for a
        build answered from a scratch directory.
    :returns: ``(peak_windows, negative_windows, signal_cache,
        control_cache, report)``: the fourth ``None`` without controls,
        and the fifth the ``report`` ``windows_factory`` returned, or
        ``None`` where an entry was read. Window ``i`` of either cache
        is row ``i`` of the `joint_windows` of the two frames.
    :raises TypeError: if ``windows_factory`` is not callable.
    :raises ValueError: if ``residency`` is neither a member of
        `RESIDENCIES` nor ``None``.

    .. rubric:: Implementation notes

    A ``"memory"`` build is not read back from the entry it wrote,
    because reading loads every array again while the grouped ones are
    still held, which doubles the peak for nothing the grouped arrays
    do not already answer.
    """

    # Both ahead of the read, which would otherwise answer a factory
    # that is not one on a hit and an unknown residency as a miss.
    if not callable(windows_factory):
        raise TypeError(
            "windows_factory is called for the window frames on a build, "
            "so it must be callable; got {}.".format(type(windows_factory).__name__)
        )
    resolved = resolve_residency(residency)
    if key is not None:
        if cache_dir is None:
            cache_dir = default_cache_dir()
        entry = read_window_cache(cache_dir, key, residency=resolved)
        if entry is not None:
            return entry + (None,)

    peak_windows, negative_windows, report = windows_factory()
    joint = joint_windows(peak_windows, negative_windows)
    signal_arrays = group_track(joint, signals, bound_columns=bound_columns)
    control_arrays = (
        None
        if controls is None
        else group_track(joint, controls, bound_columns=_CONTROL_BOUND_COLUMNS)
    )

    committed = key is not None and (
        write_window_cache(
            cache_dir,
            key,
            peak_windows,
            negative_windows,
            signal_arrays,
            control_arrays,
        )
        is not None
    )
    if committed and resolved == "mapped":
        entry = read_window_cache(cache_dir, key, residency=resolved)
        if entry is not None:
            return entry + (report,)

    return (
        peak_windows,
        negative_windows,
        WindowSignalCache.from_arrays(
            signal_arrays, residency=resolved, scratch_dir=scratch_dir
        ),
        (
            None
            if control_arrays is None
            else WindowSignalCache.from_arrays(
                control_arrays, residency=resolved, scratch_dir=scratch_dir
            )
        ),
        report,
    )


def joint_windows(peak_windows, negative_windows):
    """Return peak and negative windows as the one window set a cache is built over.

    The peaks come first, in row order, and the negatives follow in
    theirs, so negative ``i`` is row ``len(peak_windows) + i``.

    :param peak_windows: The peak window frame, as a
        `polars.DataFrame`.
    :param negative_windows: The negative window frame, of the same
        columns and dtypes.
    :returns: A `polars.DataFrame` of ``len(peak_windows) +
        len(negative_windows)`` rows.
    """

    polars = import_on_first_use("polars")
    return polars.concat([peak_windows, negative_windows])


def array_meta(arrays):
    """Return what a reader can learn about grouped arrays without reading them.

    One spelling for the three counts, which a cache carries in memory
    and an entry carries in its record, so the two cannot drift.

    :param arrays: The five values `group_track` returns.
    :returns: A dict of ``n_channels``, ``n_windows`` and ``n_sites``.
    """

    offsets, pos = arrays[0], arrays[1]
    return {
        "n_channels": int(arrays[4]),
        # One fewer window than there are boundaries.
        "n_windows": int(offsets.shape[0]) - 1,
        "n_sites": int(pos.shape[0]),
    }


def write_cache_arrays(root, offsets, pos, val, chan):
    """Write the four grouped arrays into ``root``, by their fixed names."""

    for name, array in zip(_ARRAY_NAMES, (offsets, pos, val, chan)):
        numpy.save(os.path.join(root, name + ".npy"), array)


def require_windows(caller, windows):
    """Raise `TypeError` unless ``windows`` is a `polars.DataFrame`.

    :param caller: The name of the function the windows were given to,
        for the message.
    :param windows: The value given as a window frame.
    :raises TypeError: if it is anything else, a frame of another
        library included.
    """

    polars = import_on_first_use("polars")
    if not isinstance(windows, polars.DataFrame):
        raise TypeError(
            "{} takes a polars DataFrame of windows. Got {}.".format(
                caller, type(windows).__name__
            )
        )


def track_list(caller, label, tracks):
    """Return a one-track-per-channel argument as a list, refusing a single track.

    The elements are not looked at: each is refused or read by
    `quadrat.stores.track_frame` when it is opened.

    :param caller: The name of the function the tracks were given to,
        for the message.
    :param label: The name of the argument they were given as.
    :param tracks: A sequence of tracks, one per channel, in channel
        order: a `list`, a `tuple`, or any other iterable that yields
        its elements in an order of its own, a generator included. It
        is walked once.
    :returns: The tracks, as a `list`.
    :raises TypeError: if ``tracks`` is itself a path, i.e., a `str`,
        `bytes` or `os.PathLike`; itself a frame, polars or pandas; or
        a `set`, a `frozenset` or a mapping, none of which states the
        order its tracks are in.

    .. rubric:: Implementation notes

    This is where a track argument's answer to a single item is
    decided, and it is the opposite of a locus argument's, for the
    reason `quadrat._engine.item_list` gives. Tracks are taken as any
    ordered iterable, and how many there are is the channel axis of everything
    read from them, so one track is written ``[track]``, as
    `quadrat.stores.TrackStore` also requires. It has to be refused by
    type rather than left to fail, because every single track walks as
    something: a path a character at a time, each character then
    looked up as a track of its own, and a frame a column at a time.

    A pandas frame is told by `sys.modules`, since one cannot have been
    passed in without pandas already imported, and importing it here
    would charge every grouping for a library it does not use.

    A set is refused because it would be accepted otherwise and walk
    in an order that changes between processes with the hash seed, so
    the channel axis, the track an index names and the cache key would
    all change with it and nothing would raise. A mapping walks as its
    keys, which drops the tracks if they are its values.
    """

    polars = import_on_first_use("polars")
    if fspath_or_none(tracks) is not None or isinstance(
        tracks, (polars.DataFrame, polars.LazyFrame)
    ):
        raise TypeError(
            "{} reads one track per channel: `{}` must be a sequence of "
            "tracks, not a single one. Pass a one-element list for one "
            "channel.".format(caller, label)
        )
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(tracks, pandas.DataFrame):
        raise TypeError(
            "{} reads one track per channel: `{}` must be a sequence of "
            "tracks, and a pandas DataFrame is neither that nor a track. "
            "Pass a list of track paths or polars frames.".format(caller, label)
        )
    if isinstance(tracks, (set, frozenset, collections.abc.Mapping)):
        raise TypeError(
            "{} reads one track per channel, and the order of `{}` is the "
            "channel axis: a {} states no order for its tracks. Pass a "
            "list or a tuple in channel order.".format(
                caller, label, type(tracks).__name__
            )
        )
    return list(tracks)


def window_columns(windows, bound_columns):
    """Return a window set's bounds, and its rows grouped by contig.

    The decomposition every reader of a window set begins with, written
    once so that no two readers can disagree about which rows belong to
    a contig or which columns bound a window. They differ in what they
    do per contig and not in how they get there.

    :param windows: The window frame, as `group_track` takes it.
    :param bound_columns: Its start and end column names.
    :returns: ``(starts, ends, groups)``. ``starts`` and ``ends`` are
        ``int64`` over every window. ``groups`` pairs each contig with
        the ascending row indices on it, in ascending order of contig
        name, and is empty for an empty frame.

    .. rubric:: Implementation notes

    The contig column is taken through ``to_numpy`` rather than a
    comprehension over the Series, which is the same values built in
    Rust instead of one Python call per window -- about five times
    faster at every size measured.

    The grouping is one stable ``argsort`` with the runs sliced out of
    it, rather than a scan of the whole column per distinct contig.
    That trades O(windows x contigs) for O(windows log windows): a
    little slower at a two-dozen-contig locus set and nine times
    faster at three hundred, which is the direction a scaffold-bearing
    reference goes. The sort is stable, so the indices inside a run
    stay ascending without re-sorting them.
    """

    lo_column, hi_column = bound_columns
    chroms = windows["chrom"].to_numpy().astype(str)
    starts = numpy.asarray(windows[lo_column], dtype=numpy.int64)
    ends = numpy.asarray(windows[hi_column], dtype=numpy.int64)

    if len(chroms) == 0:
        return starts, ends, []

    order = numpy.argsort(chroms, kind="stable")
    ordered = chroms[order]
    edges = numpy.flatnonzero(
        numpy.concatenate(([True], ordered[1:] != ordered[:-1], [True]))
    )
    groups = [
        (ordered[edges[index]], order[edges[index] : edges[index + 1]])
        for index in range(len(edges) - 1)
    ]
    return starts, ends, groups


def group_track(windows, tracks, bound_columns=("sig_start", "sig_end")):
    """Group every track's rows by the window they fall inside.

    Keeps only what the windows cover, which is the point: the windows
    are known before any signal is read, so a locus set that touches a
    quarter of the genome reads a quarter of the rows. A position
    covered by two overlapping windows is stored once per window, and
    that duplication is the price of a read that needs no search.

    Needs the ``pileup`` extra, which installs the engine the grouping
    runs on. Concurrent calls in one process are unsupported;
    `quadrat._engine` says why.

    :param windows: A `polars.DataFrame` carrying ``chrom`` and the
        two columns ``bound_columns`` names, one row per window. Row
        order is the window index every reader of the result addresses
        by. Refused by type: a frame of another library groups
        correctly here and then cannot be persisted, so every read of
        its entry would be a miss.
    :param tracks: One track per channel, in channel order, as
        `track_list` takes them, each of any shape
        `quadrat.stores.track_frame` accepts.
    :param bound_columns: The window's start and end column names.
        The signal is grouped over the output window and a control over
        the wider input window, which is the whole of why this is a
        parameter.
    :returns: ``(offsets, pos, val, chan, n_channels)``.

        ``offsets`` is one-dimensional over windows, ``int64``, of
        length ``len(windows) + 1``. Window ``i``'s rows are
        ``pos[offsets[i]:offsets[i + 1]]`` and the same slice of
        ``val`` and ``chan``. There is no channel axis in it.

        Every channel's rows for one window live in that single run,
        interleaved, with the channel index carried per row in
        ``chan`` (``int16``). ``pos`` is ``int32`` and holds the offset
        from the window's own ``bound_columns[0]``, not an absolute
        coordinate; ``val`` is ``float32``.

        The arrays are a function of the rows alone: the same windows
        and tracks reproduce them byte for byte whatever order the
        rows arrive in. Within a run, rows are ordered by offset, then
        channel, then value.

        ``n_channels`` is returned because the arrays cannot recover
        it: a trailing channel with no rows in any window leaves no
        trace in ``chan``.
    :raises TypeError: if ``windows`` is not a `polars.DataFrame`, if
        ``tracks`` is a single track or an unordered collection of
        them, or if a track is of no shape a track can take.
    :raises ValueError: if there are more channels than ``int16`` can
        index, or if a track directory does not verify complete.
    :raises FileNotFoundError: if a track path does not exist.
    :raises ImportError: if the ``pileup`` extra is not installed.

    .. rubric:: Implementation notes

    Grouped by window rather than by ``(channel, window)``. Grouping
    the other way would make a read one Python iteration per channel,
    each over a handful of elements and so almost all interpreter
    overhead; this way every channel scatters in a single vectorised
    write, which is 4.3x faster at ten channels and the difference
    between the read being linear in channel count or not.

    One GIQL ``INTERSECTS`` join per contig, covering every channel:
    each channel's rows on the contig are stacked into one frame, and
    the channel is carried through the join as a projected column
    (`_signal_sql` says why that costs the join nothing). The join
    returns ``(window index, offset within the window, value,
    channel)``, which is the grouped form already, so the engine does
    the work this would otherwise do with two ``searchsorted`` calls
    per window. For 68 channels over 24 contigs that is 24 joins rather
    than 1,632, and the count is what matters: a query's cost is
    dominated by planning rather than by binding, so the rows per query
    do not.

    The join is the one implementation of the grouping. Two
    implementations of one function agreeing is a thing to test rather
    than a thing to have. `window_group_sums` reads the same rows
    another way, for the reason its own notes give.

    Memory peaks near 28 bytes a site, a site being one row of the
    result. Every contig's hits are held until the last contig is
    joined, at 18 bytes a site: an ``int64`` window index, an ``int32``
    offset, a ``float32`` value and an ``int16`` channel. The 10 bytes a
    site returned, ``pos``, ``val`` and ``chan`` at the dtypes above,
    are filled from them while they are still held. Beside that sit
    ``offsets``, at 8 bytes a window, and one contig's join answer at a
    time.
    """

    polars = import_on_first_use("polars")
    tracks = track_list("group_track", "tracks", tracks)
    n_windows = len(windows)
    n_channels = len(tracks)

    # Checked before any scan, because the limit belongs to the layout
    # and no input is worth reading once it is exceeded.
    if n_channels > numpy.iinfo(numpy.int16).max:
        raise ValueError(
            "more channels than the grouped layout's channel index can "
            "hold: {} against a limit of {}.".format(
                n_channels, numpy.iinfo(numpy.int16).max
            )
        )

    # Refused by type, for the reason the ``windows`` parameter gives.
    require_windows("group_track", windows)

    starts, ends, groups = window_columns(windows, bound_columns)

    sql = _signal_sql()
    frames = [track_frame(track) for track in tracks]
    counts = numpy.zeros(n_windows, dtype=numpy.int64)
    gathered = []

    for chrom, rows in groups:
        placed = polars.DataFrame(
            {
                "idx": rows.astype(numpy.int64),
                "chrom": numpy.repeat(str(chrom), len(rows)),
                "wstart": starts[rows],
                "wend": ends[rows],
            },
            schema={
                "idx": polars.Int64,
                "chrom": polars.Utf8,
                "wstart": polars.Int64,
                "wend": polars.Int64,
            },
        )

        # Every channel in one join per contig; see the notes.
        rowsets = []
        for channel, frame in enumerate(frames):
            rowset = (
                frame.filter(polars.col("chrom") == str(chrom))
                .select(
                    polars.col("chrom").cast(polars.Utf8),
                    polars.col("pos").cast(polars.Int64).alias("cstart"),
                    (polars.col("pos").cast(polars.Int64) + 1).alias("cend"),
                    polars.col("value").cast(polars.Float64),
                    polars.lit(channel, dtype=polars.Int16).alias("chan"),
                )
                .collect()
            )
            # Skipped rather than concatenated. Every part is built by
            # the select above and so carries the same schema, empty or
            # not, which `concat` would accept -- this drops the copy
            # and the pass over it, not a failure. The `continue` below
            # is the one that matters: a contig no channel records
            # leaves nothing to concatenate at all.
            if rowset.height:
                rowsets.append(rowset)
        if not rowsets:
            continue
        rowset = rowsets[0] if len(rowsets) == 1 else polars.concat(rowsets)

        with registered_frames({WINDOWS_TABLE: placed, TRACK_TABLE: rowset}) as pb:
            hits = pb.sql(sql).collect()
        if not hits.height:
            continue

        index = hits["idx"].to_numpy()
        # Sorted per contig rather than once over every contig's hits
        # together. A window belongs to one contig, so these index
        # ranges are disjoint between iterations and each can be
        # ordered and ranked on its own -- which matters because a
        # sort is super-linear: one pass over everything measured
        # nearly three times the cost of the same rows sorted in
        # tenths.
        #
        # Every column is a key, for the ordering guarantee in the
        # docstring.
        order = numpy.lexsort(
            (
                hits["value"].to_numpy(),
                hits["chan"].to_numpy(),
                hits["offset"].to_numpy(),
                index,
            )
        )
        # Narrowed here rather than at the scatter. The join answers in
        # the widths it computes in, and holding every contig's hits at
        # those widths to the end costs 24 bytes a site against the 10
        # the result keeps -- so the cast that has to happen anyway
        # happens while the arrays are small.
        gathered.append(
            (
                index[order],
                hits["offset"].to_numpy()[order].astype(numpy.int32),
                hits["value"].to_numpy()[order].astype(numpy.float32),
                hits["chan"].to_numpy()[order].astype(numpy.int16),
            )
        )
        # Per window over every channel at once, which is the only
        # breakdown anything reads: the result has no channel axis.
        counts += numpy.bincount(index, minlength=n_windows)

    offsets = numpy.zeros(n_windows + 1, dtype=numpy.int64)
    numpy.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])

    pos = numpy.empty(total, dtype=numpy.int32)
    val = numpy.empty(total, dtype=numpy.float32)
    chan = numpy.empty(total, dtype=numpy.int16)

    for index, offset, value, channel_of in gathered:
        # `index` is sorted, so `searchsorted` gives each row's first
        # occurrence and the distance from it is its rank in the
        # window. Every channel is in this one pass, so the rank is the
        # position in the whole run and the window's own start is the
        # base -- there is no per-channel write head to carry, and no
        # contig can reach a window another contig wrote.
        rank = numpy.arange(len(index)) - numpy.searchsorted(index, index)
        slots = offsets[index] + rank
        pos[slots] = offset
        val[slots] = value
        chan[slots] = channel_of

    return offsets, pos, val, chan, n_channels


def window_group_sums(
    windows, tracks, channel_group_sizes, bound_columns=("sig_start", "sig_end")
):
    """Return the total signal per window per channel group, without densifying.

    The arithmetic is exact rather than approximate: a prefix sum per
    contig and two searches per window, in ``float64``, so a locus's
    total does not depend on how the windows were batched.

    :param windows: A `polars.DataFrame` carrying ``chrom`` and the
        two columns ``bound_columns`` names, one row per window.
    :param tracks: One track per channel, in channel order, as
        `group_track` takes them.
    :param channel_group_sizes: How many consecutive channels each
        group pools, in group order. Must sum to the channel count.
    :param bound_columns: The window's start and end column names, as
        `group_track` takes them.
    :returns: A `torch.Tensor` of shape ``(len(windows),
        len(channel_group_sizes))`` and dtype ``float32``.
    :raises ValueError: if ``channel_group_sizes`` does not sum to
        the channel count, or if a track is a directory that cannot be
        verified complete.
    :raises TypeError: if ``tracks`` is a single track or an unordered
        collection of them, or if a track is of no shape a track can
        take.
    :raises FileNotFoundError: if a track path does not exist.

    .. rubric:: Implementation notes

    This replaces summing a dense ``(windows, channels, span)`` tensor
    to take a percentile of it, which is the eager form of the same
    question and costs the memory this module exists to avoid.

    ``tracks`` accepts what `group_track` accepts because a caller who
    can build a cache over a track set should be able to total it, and
    ``bound_columns`` is the same parameter there and here because
    totalling a control over the wider input window is the same
    question as grouping one over it. ``channel_group_sizes`` is named
    for the axis it pools, because `group_track` groups by window
    instead.

    A second reader of the same rows as `group_track`, kept because
    the two are reachable from different installs rather than because
    the arithmetic differs. This one touches ``polars`` and ``torch``,
    both base dependencies; `group_track` reaches ``giql`` through
    `_signal_sql` and ``polars_bio`` through `registered_frames`, and
    those two are exactly what the ``pileup`` extra installs. So a
    caller who only has the base install can total a window set and
    cannot group one, which is the distinct reason this exists and
    the exemption from the rule its sibling's notes state.

    The per-contig sort is not incidental. The prefix sum below needs
    positions ascending, and a track read whole rather than per contig
    arrives in whatever order its files are globbed in.
    """

    polars = import_on_first_use("polars")
    torch = import_on_first_use("torch")
    tracks = track_list("window_group_sums", "tracks", tracks)
    n_windows = len(windows)
    totals = numpy.zeros((n_windows, len(channel_group_sizes)), dtype=numpy.float64)

    # Checked above the empty-window return: an inconsistent grouping
    # is the caller's mistake whether or not there is a window to apply
    # it to, and refusing it only sometimes is worse than either answer.
    group_of = []
    for group, size in enumerate(channel_group_sizes):
        group_of.extend([group] * size)
    if len(group_of) != len(tracks):
        raise ValueError(
            "channel_group_sizes sums to {} and there are {} tracks; "
            "every channel belongs to exactly one group.".format(
                len(group_of), len(tracks)
            )
        )

    if n_windows == 0:
        return torch.from_numpy(totals.astype(numpy.float32))

    starts, ends, groups = window_columns(windows, bound_columns)

    # Resolved once per track, before any contig, so a directory is
    # verified complete, and verified once rather than once per contig.
    frames = [track_frame(track) for track in tracks]

    for chrom, rows in groups:
        for channel, frame in enumerate(frames):
            read = (
                frame.filter(polars.col("chrom") == str(chrom))
                .select(
                    polars.col("pos").cast(polars.Int64),
                    polars.col("value").cast(polars.Float64),
                )
                .sort("pos")
                .collect()
            )
            if read.height == 0:
                continue
            positions = read["pos"].to_numpy()
            values = read["value"].to_numpy()
            cumulative = numpy.concatenate(
                ([0.0], numpy.cumsum(values, dtype=numpy.float64))
            )
            lo = numpy.searchsorted(positions, starts[rows], side="left")
            hi = numpy.searchsorted(positions, ends[rows], side="left")
            totals[rows, group_of[channel]] += cumulative[hi] - cumulative[lo]

    return torch.from_numpy(totals.astype(numpy.float32))


def resolve_residency(residency=None):
    """Return where a cache keeps its arrays, resolving ``None`` from the start method.

    `WindowSignalCache.build` states, under ``residency``, the rule
    ``None`` resolves by.

    :param residency: A member of `RESIDENCIES`, or ``None`` to
        resolve from the start method.
    :returns: A member of `RESIDENCIES`.
    :raises ValueError: if ``residency`` is neither and not ``None``.

    .. rubric:: Implementation notes

    Both residencies are correct and which is faster depends entirely
    on how loader workers start, so ``None`` asks the start method
    rather than making the caller state it.

    Under ``fork`` the parent's arrays are already shared
    copy-on-write, so keeping them in memory costs nothing extra and
    avoids the page-fault indirection a mapping adds: 210 us per sample
    against 267 us mapped, on a ten-channel set. Under ``spawn`` there
    is no sharing to inherit and the arrays would be pickled into every
    worker, once each -- flat for a mapped cache and linear in cache
    size otherwise, 9.4 s against 34.7 s to the first batch at four
    channels, diverging from there.

    Per call rather than frozen at import, because
    ``multiprocessing.set_start_method`` is idiomatically called from
    inside ``if __name__ == "__main__":``, after the import that pulled
    this package in, so a frozen answer samples the platform default
    and a program that deliberately chose ``fork`` to get the sharing
    is handed the mapped form anyway.

    The read is ``allow_none=True``, which is a correctness
    requirement rather than a preference: ``allow_none=False``
    *assigns* the default context when none has been chosen, so asking
    the question would answer it, and a caller who had not yet reached
    their ``set_start_method`` could no longer call it.

    The platform's default, for a method not yet chosen, is read from
    `multiprocessing.get_all_start_methods`, which documents its first
    entry as that default and binds nothing. So the answer follows how
    the program will in fact start its workers even before the program
    has said so, and asking still does not decide it. The alternative
    -- treating unchosen as unknown and taking the residency that is
    correct under either method -- is only slower, but it is slower on
    the default path of every program that never calls
    ``set_start_method``, which is most of them.
    """

    if residency is None:
        method = multiprocessing.get_start_method(allow_none=True)
        if method is None:
            method = multiprocessing.get_all_start_methods()[0]
        return "memory" if method == "fork" else "mapped"
    if residency not in RESIDENCIES:
        raise ValueError(
            "residency must be one of {} or None, got {!r}.".format(
                ", ".join(repr(value) for value in RESIDENCIES), residency
            )
        )
    return residency


def _key_paths(label, items):
    """A key's file argument, already a list of items, as a list of path strings.

    :param label: The name of the argument, for the message.
    :param items: The argument's items, as `quadrat._engine.item_list`
        or `track_list` answers them.
    :raises TypeError: if an item is anything but a path.
    """

    paths = []
    for item in items:
        path = fspath_or_none(item)
        if path is None:
            raise TypeError(
                "a persisted window cache is keyed by files, so every item "
                "of `{}` must be a str, bytes or os.PathLike; got {}. An "
                "input held in memory has no key: pass cache_dir=False to "
                "build without an entry.".format(label, type(item).__name__)
            )
        paths.append(path)
    return paths


def _threshold(value):
    """A count threshold as the plain floats the key hashes.

    The key is sensitive to the threshold and not to its spelling: a
    ``float32`` out of `window_group_sums`, a tensor element, an
    ``int`` and a per-group array all reduce to the same canonical
    record that the equal Python float would.
    """

    if value is None:
        return None
    return numpy.asarray(value, dtype=float).tolist()


def _refuse_inconsistent_arrays(offsets, pos, val, chan, n_channels):
    """Refuse grouped arrays a draw could not read correctly.

    The cross-member invariants `group_track` satisfies by
    construction and a caller assembling the tuple by hand may not.
    Checked where the arrays are already in memory, once per cache;
    `_refuse_wrong_sizes` covers an entry read from disk, where this
    scan would be a full read of every mapped array.

    The channel bound is the one that matters most. A negative
    ``chan`` indexes from the end of the channel axis, so a draw over
    it answers signal on the wrong row and raises nothing.

    :raises ValueError: if ``pos``, ``val`` and ``chan`` differ in
        length, if ``offsets`` does not run non-decreasing from zero to
        that length, or if a ``chan`` falls outside
        ``range(n_channels)``.
    """

    sites = len(pos)
    if len(val) != sites or len(chan) != sites:
        raise ValueError(
            "pos, val and chan must be one length, got {}, {} and {}.".format(
                sites, len(val), len(chan)
            )
        )

    offsets = numpy.asarray(offsets)
    if (
        offsets.ndim != 1
        or len(offsets) == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != sites
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError(
            "offsets must run non-decreasing from 0 to the {} rows the "
            "other arrays hold.".format(sites)
        )

    if sites:
        chan = numpy.asarray(chan)
        low, high = int(chan.min()), int(chan.max())
        if low < 0 or high >= n_channels:
            raise ValueError(
                "chan must lie in range({}), got values from {} to {}.".format(
                    n_channels, low, high
                )
            )


def _refuse_wrong_sizes(directory, meta):
    """Refuse an entry's arrays whose sizes disagree with its recorded counts.

    Reads nothing: every array here has a fixed dtype and a length
    fixed by one of the two counts an entry records, so the file size
    is predictable exactly and `os.stat` settles it. `_adopted`'s notes
    say where this stands in verifying an entry.

    It is the only thing that catches an array longer than its counts
    allow: a `numpy.load` of an over-long file succeeds, mapped or
    not, so neither the eager map nor the in-memory read would notice.

    :param directory: The entry's ``signal`` or ``control`` directory.
    :param meta: The record's counts for it, as `array_meta` wrote
        them, or ``None`` for a record that carries no counts.
    :raises ValueError: if the counts are absent or are not counts, or
        if any array's size disagrees with them.
    :raises OSError: if an array is missing.
    """

    if not isinstance(meta, dict):
        raise ValueError(
            "an entry records no array counts to verify against, got {!r}.".format(meta)
        )
    try:
        windows, sites = int(meta["n_windows"]), int(meta["n_sites"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "an entry's array counts are not counts: {!r}.".format(meta)
        ) from None

    for name, width, per_window in _ARRAY_WIDTHS:
        expected = _NPY_HEADER_BYTES + width * (windows + 1 if per_window else sites)
        size = os.stat(os.path.join(directory, name + ".npy")).st_size
        if size != expected:
            raise ValueError(
                "an entry's {}.npy is {} bytes where its record implies {}.".format(
                    name, size, expected
                )
            )


def _adopted(root, kind, manifest, residency=None):
    """One of an entry's two caches, refusing a record it cannot trust.

    Raises rather than answering, and only `ValueError` or `OSError`,
    the two errors a read of an entry converts to a miss.

    :param root: The entry's directory.
    :param kind: ``"signal"`` or ``"control"``, which names both the
        subdirectory and the record's fields for it.
    :param manifest: The entry's record, as `read_manifest` returns it.
    :param residency: As `WindowSignalCache.build` takes it, and
        resolved the same way, so a hit is held where a cold build of
        the same arguments would hold it.
    :raises ValueError: if the record's channel count for ``kind`` is
        absent or is not a count, if the record carries no array
        counts, or if an array's size disagrees with them.
    :raises OSError: if the directory is missing an array.

    .. rubric:: Implementation notes

    Presence is what `require_complete` already establishes from the
    record's file list, and it says nothing about the array *bytes*.
    A size derived from the recorded counts is the cheap half of
    what is needed -- it catches every truncation without reading
    anything -- and opening the arrays rather than leaving them to the
    first draw is the other half, because numpy's refusal of a corrupt
    header is then a `ValueError` inside `read_window_cache`'s block
    and so one of its misses, instead of an exception raised from a
    loader worker halfway through an epoch.

    Both halves live in `WindowSignalCache.adopt`, which is where a
    caller reaching the entry without going through this function
    arrives, so the public path and this one verify the same way by
    construction rather than by being kept in step. The in-memory
    branch below calls the size half directly, being the one path that
    maps nothing.

    What neither half reaches is a corruption that preserved the file
    size. Nothing cheap does, and an entry is gigabytes, so the digest
    that would is not worth running before the work it guards.
    """

    count = manifest.get("{}_channels".format(kind))
    # `bool` is an `int` and `True` would survive the comparison below
    # as a channel count of one, so it is excluded by name.
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(
            "an entry's {}_channels must be a count, got {!r}.".format(kind, count)
        )

    directory = os.path.join(root, kind)
    meta = manifest.get("{}_meta".format(kind))

    if resolve_residency(residency) == "memory":
        # What `adopt` does on the other branch, less the mapping.
        # Only the size half is wanted here: a header numpy refuses
        # raises out of the `numpy.load` below by itself.
        _refuse_wrong_sizes(directory, meta)
        loaded = [
            numpy.load(os.path.join(directory, name + ".npy")) for name in _ARRAY_NAMES
        ]
        return WindowSignalCache(
            offsets=loaded[0],
            pos=loaded[1],
            val=loaded[2],
            chan=loaded[3],
            n_channels=count,
            meta=meta,
        )

    return WindowSignalCache.adopt(directory, count, meta=meta)


def _hash_file(digest, path):
    """Fold a file's bytes into ``digest``, or the fact of its absence."""

    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        digest.update(b"<absent>")


def _hash_track(digest, path):
    """Fold a track's identity into ``digest``, without reading its rows.

    Why the record and a stat rather than the bytes is `window_cache_key`'s
    to say; this is the mechanism.
    """

    digest.update(os.path.abspath(path).encode())
    manifest = read_manifest(path)
    digest.update(
        json.dumps(manifest, sort_keys=True).encode()
        if manifest is not None
        else b"<no manifest>"
    )
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        digest.update(b"<unreadable>")
        return
    for name in entries:
        if not name.endswith(".parquet"):
            continue
        info = os.stat(os.path.join(path, name))
        digest.update("{}:{}:{}".format(name, info.st_size, info.st_mtime_ns).encode())


def _reap_scratch(root, owner_pid):
    """Remove a scratch directory, in the process that made it and nowhere else.

    The pid is what makes the ownership one process's. ``fork`` copies
    `atexit` registrations into the child along with everything else,
    so without the check a child that reached a normal exit would
    delete arrays its parent and its siblings are still mapping -- and
    ``ignore_errors`` would make that silent. A `multiprocessing`
    worker does not reach it, exiting through ``os._exit``, but that
    is a fact about one library's workers rather than a guarantee
    about forked children, so it is not what this rests on.

    :param root: The scratch directory.
    :param owner_pid: The pid of the process that made it, captured
        when it was made.
    """

    if os.getpid() != owner_pid:
        return
    shutil.rmtree(root, ignore_errors=True)


@functools.cache
def _signal_sql():
    """Return the grouping join, transpiled once per process.

    Once per process because the query never varies: the contig varies
    by what is registered under the two table names, not by the text,
    and the channel is a column the join carries through rather than a
    thing the text knows about.

    ``chan`` is projected rather than joined on. giql rewrites the
    ``ON`` clause and leaves the select list alone, so an extra column
    costs the join nothing and saves registering one frame per channel
    per contig.
    """

    return transpile_intervals(
        "SELECT w.idx, c.cstart - w.wstart AS offset, c.value, "
        "c.chan FROM {} AS w JOIN {} AS c ON w.interval "
        "INTERSECTS c.interval".format(WINDOWS_TABLE, TRACK_TABLE),
        (WINDOWS_TABLE, "wstart", "wend"),
        (TRACK_TABLE, "cstart", "cend"),
    )
