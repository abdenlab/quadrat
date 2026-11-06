"""The sparse window cache, and the grouping that fills it.

A window cache holds the signal for one fixed set of windows, grouped
so that reading a draw is a slice and a scatter rather than a search.

It is sparse because the dense form does not fit. One entry per
recorded position costs about 5.6 GB of proportional set size over a
whole process tree for 57,742 windows of 2,000 bp over 68 channels,
where the dense ``windows x span x channels`` array for the same inputs
is 31.4 GB. That is also what makes the cost insensitive to channel
count: pooling changes how positions are labelled rather than how many
there are, so ten channels cost 4.86 GB against sixty-eight at
5.71 GB, where dense would be 4.6 GB against 31.4 GB.

Nothing here is a durable artifact. A cache is derived from the tracks,
the locus set, the contig split, the exclusion lists and the window
geometry, and rebuilds from them in under a second for a locus set of
order 10^5 windows. The tracks are the only thing worth keeping.
"""

import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
import time
import weakref

import numpy

from quadrat._engine import (
    TRACK_TABLE,
    WINDOWS_TABLE,
    fspath_or_none,
    import_on_first_use,
    registered_frames,
)
from quadrat.stores import ProcessLocalHandles, chrom_track, track_frame
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
CACHE_VERSION = 1


#: Where entries go when no directory is named. Read on every call
#: rather than once, so a process that sets it after importing this
#: package is not left pointed at the real user cache -- which is also
#: what the suite's own guard depends on.
_CACHE_DIR_ENV = "QUADRAT_CACHE_DIR"


#: The four arrays a grouped result is written as, in the order
#: `group_track` returns them. Named once because the tuple is the file
#: layout: a fifth array added without extending this would be written
#: by nothing and read as absent.
_ARRAY_NAMES = ("offsets", "pos", "val", "chan")


#: The transpiled grouping join, per process. A plain dict rather than
#: SCREAMING_CASE because the binding is constant and the referent is
#: not: `_signal_sql` fills it on first use.
_transpiled = {}


class WindowSignalCache(ProcessLocalHandles):
    """Signal for one fixed set of windows, read by draw rather than by place.

    A read is a slice and a scatter. The sampler has already chosen the
    window and the jitter offset, and this cache was built for exactly
    those windows, so there is nothing to look up -- which also means
    there is no binary search to degrade once the data outgrows page
    cache, as a search over one large sorted array does.

    The layout is the one `group_track` returns and documents.

    Derived, never durable. A cache depends on the locus set, the
    contig split, the exclusion lists and the jitter bound, and
    rebuilds from the tracks in under a second. The tracks are the only
    thing worth keeping, which is why nothing here writes one.

    Built through `build`, `from_arrays` or `adopt` rather than
    directly; the constructor takes whatever those three assembled.

    :param scratch: A directory holding the four arrays, which every
        process memory-maps, or ``None`` to hold them in memory.
    :param owns_scratch: Whether dropping this cache should delete
        ``scratch``. Only `build` and `from_arrays` set it, and only
        for a directory they made.
    :param meta: What a reader may want to know without touching the
        arrays, or ``None``.
    """

    def __init__(
        self,
        offsets=None,
        pos=None,
        val=None,
        chan=None,
        n_channels=None,
        scratch=None,
        meta=None,
        owns_scratch=False,
    ):
        self.scratch = scratch
        self.meta = meta
        # Pickled on purpose, so a worker's copy never owns anything: a
        # worker that exited first would otherwise delete the arrays
        # every other worker is still reading.
        self.owns_scratch = bool(owns_scratch)
        self._offsets = offsets
        self._pos = pos
        self._val = val
        self._chan = chan
        self._n_channels = n_channels
        self._local = None

    @property
    def n_channels(self):
        """How many channels a read of this cache answers with."""

        return int(self._n_channels)

    @classmethod
    def build(
        cls,
        windows,
        tracks,
        bounds=("sig_start", "sig_end"),
        residency=None,
        scratch_dir=None,
    ):
        """Group ``tracks`` over ``windows`` and hold the result.

        :param windows: As `group_track` takes it.
        :param tracks: One track per channel, in channel order.
        :param bounds: The window's start and end column names.
        :param residency: ``"memory"``, ``"scratch"``, or ``None`` to
            choose from the multiprocessing start method. See
            `_resolve_residency`.
        :param scratch_dir: Where a scratch directory is made, or
            ``None`` for the platform's temporary directory.
        """

        return cls.from_arrays(
            group_track(windows, tracks, bounds=bounds),
            residency=residency,
            scratch_dir=scratch_dir,
        )

    @classmethod
    def from_arrays(cls, arrays, residency=None, scratch_dir=None):
        """Hold arrays `group_track` has already produced.

        Separate from `build` so a caller that wants the arrays
        themselves -- to persist them, say -- does not group twice to
        also get a cache.

        :param arrays: The five values `group_track` returns.
        :param residency: As `build` takes it.
        :param scratch_dir: As `build` takes it.
        """

        resolved = _resolve_residency(residency)
        offsets, pos, val, chan, n_channels = arrays
        if resolved == "memory":
            return cls(
                offsets=offsets, pos=pos, val=val, chan=chan, n_channels=n_channels
            )

        root = tempfile.mkdtemp(prefix="quadrat-cache-", dir=scratch_dir)
        write_cache_arrays(root, offsets, pos, val, chan)
        cache = cls(
            scratch=root,
            n_channels=n_channels,
            owns_scratch=True,
            meta={
                "n_channels": n_channels,
                "n_windows": int(offsets.shape[0]) - 1,
                "n_sites": int(pos.shape[0]),
            },
        )
        # The directory must outlive this scope -- the loader holds the
        # cache and the files have to last as long as it does -- and a
        # finalizer is what ties the two together. Registered only for a
        # directory this call made; see `adopt`.
        weakref.finalize(cache, shutil.rmtree, root, True)
        return cache

    @classmethod
    def adopt(cls, root, n_channels, meta=None):
        """Read arrays already on disk without taking responsibility for them.

        The persistent-entry constructor. The directory outlives every
        cache that reads it, so no finalizer is registered and nothing
        here deletes it -- the first run to finish would otherwise
        delete the entry every later run is counting on.

        :param root: The directory holding the four arrays.
        :param n_channels: The channel count the arrays were grouped
            at, which they cannot recover themselves.
        :param meta: As the constructor takes it.
        """

        return cls(
            scratch=str(root),
            n_channels=int(n_channels),
            owns_scratch=False,
            meta=meta,
        )

    def nbytes(self):
        """How many bytes the four arrays occupy."""

        return int(sum(array.nbytes for array in self._arrays()))

    def window(self, index, jitter, length, reverse_complement=False, perm=None):
        """Dense signal for window ``index`` at ``jitter``, every channel.

        :param index: The window, as its row in the set this cache was
            built for.
        :param jitter: How far the draw is shifted. A position shifted
            outside ``[0, length)`` is dropped, which is what makes a
            window near a contig edge read as zeros there rather than
            raise.
        :param length: The span to answer, in bases.
        :param reverse_complement: Whether to mirror the draw.
        :param perm: The channel permutation a mirrored draw applies,
            or ``None`` to leave channels alone. A stranded pair swaps
            under reverse-complement; an unstranded channel does not.
        :returns: A `torch.Tensor` of shape ``(n_channels, length)``
            and dtype ``float32``.

        .. rubric:: Implementation notes

        Every channel scatters in one vectorised write, because a
        window's rows are stored together with a channel index beside
        them. Looping per channel cost 4.3x as much at ten channels:
        the work per channel is a handful of elements, so it was all
        interpreter overhead.

        The reverse-complement is applied here, by writing each row to
        its mirrored offset and permuted channel, rather than afterwards
        by flipping the dense result. Afterwards means two passes over
        ``(channels, length)`` floats for the sake of the hundred or so
        rows per channel that are not zero.
        """

        torch = import_on_first_use("torch")
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
            out[channels, offset] = numpy.asarray(val[first:last])[keep]

        return torch.from_numpy(out)

    def _arrays(self):
        """The four arrays, memory-mapped per process when on disk."""

        if self.scratch is None:
            return self._offsets, self._pos, self._val, self._chan

        local = self._handles()
        arrays = local.get("arrays")
        if arrays is None:
            arrays = local["arrays"] = tuple(
                numpy.load(os.path.join(self.scratch, name + ".npy"), mmap_mode="r")
                for name in _ARRAY_NAMES
            )
        return arrays


def default_cache_dir():
    """Where persisted entries go when the caller names no directory.

    Resolved from the environment on every call. ``QUADRAT_CACHE_DIR``
    wins, then the XDG convention, then a home-relative default -- so
    caching needs no configuration and relocating it needs one
    variable.
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
):
    """A content hash of everything that decides what an entry holds.

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

    :param signals: The signal tracks, in channel order.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`.
    :param peaks: The peak locus file, or an iterable of them.
    :param negatives: The background locus file, or an iterable.
    :param chroms: The contig subset, or ``None`` for all.
    :param controls: The control tracks, in channel order, or ``None``.
    :returns: The key, as a hex `str`.
    :raises TypeError: if ``sequences`` is not a path.

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

    The domain tag names this package. Upstream's entries and these are
    not interchangeable, which is the deliberate opposite of the track
    format's interoperability: a track costs minutes to rebuild and is
    shared on purpose, an entry rebuilds in under a second, so sharing
    buys nothing and risks serving bytes grouped by other code.
    """

    reference = fspath_or_none(sequences)
    if reference is None:
        raise TypeError(
            "window_cache_key hashes the reference it was given: "
            "sequences must be a str, bytes or os.PathLike, got "
            "{}.".format(type(sequences).__name__)
        )

    digest = hashlib.sha256()
    digest.update(b"quadrat-window-cache\x00")
    digest.update(str(CACHE_VERSION).encode())

    for label, loci in (("peaks", peaks), ("negatives", negatives)):
        digest.update(label.encode())
        named = fspath_or_none(loci)
        for path in [named] if named is not None else list(loci or ()):
            digest.update(os.path.abspath(path).encode())
            _hash_file(digest, path)

    digest.update(b"sequences")
    digest.update(os.path.abspath(reference).encode())
    # The index rather than `reference` itself, and through the
    # normalized path: upstream built this name with `str(sequences)`,
    # so a `bytes` reference hashed the index of a path spelled from a
    # repr, which is absent -- and two different genomes then collided.
    _hash_file(digest, reference + ".fai")

    digest.update(
        json.dumps(
            {
                "chroms": list(chroms) if chroms is not None else None,
                "in_window": int(in_window),
                "out_window": int(out_window),
                "max_jitter": int(max_jitter),
                "summits": bool(summits),
                "min_counts": min_counts,
                "max_counts": max_counts,
                "signal_groups": list(signal_groups) if signal_groups else None,
                "control_groups": list(control_groups) if control_groups else None,
            },
            sort_keys=True,
        ).encode()
    )

    digest.update(b"exclusion")
    for path in list(exclusion_lists or ()):
        digest.update(os.path.abspath(path).encode())
        _hash_file(digest, path)

    for label, tracks in (("signals", signals), ("controls", controls)):
        digest.update(label.encode())
        for path in list(tracks or ()):
            _hash_track(digest, path)

    return digest.hexdigest()


def read_window_cache(cache_dir, key):
    """One persisted entry, or ``None`` where there is no verifiable one.

    Anything unreadable, unverifiable, or built under another version
    is reported as a miss, so the caller rebuilds rather than trusts
    it. A miss is not an error: the entry is derived and rebuilding is
    the correct answer to every doubt about it.

    :param cache_dir: The directory holding entries.
    :param key: The key, from `window_cache_key`.
    :returns: ``(peak_windows, negative_windows, signal_cache,
        control_cache)``, the last ``None`` where the entry carries no
        controls; or ``None`` for a miss.
    """

    polars = import_on_first_use("polars")
    root = os.path.join(cache_dir, key)

    try:
        # The record's shape, the listed files' presence and the schema
        # version are `require_complete`'s to check, and it was written
        # to accept an entry's record: no row total, and files listed
        # with no count. Its refusal is converted rather than raised,
        # because this answers a miss where it answers an error -- and
        # every message it emits talks about a track and tells the
        # caller to pile up the alignments again, which is the wrong
        # instruction for an entry and must not reach anyone.
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
    except (OSError, ValueError, polars.exceptions.PolarsError):
        return None

    signal = WindowSignalCache.adopt(
        os.path.join(root, "signal"),
        manifest["signal_channels"],
        meta=manifest.get("signal_meta"),
    )
    control = (
        None
        if manifest.get("control_channels") is None
        else WindowSignalCache.adopt(
            os.path.join(root, "control"),
            manifest["control_channels"],
            meta=manifest.get("control_meta"),
        )
    )
    return peaks, negatives, signal, control


def write_window_cache(
    cache_dir, key, peak_windows, negative_windows, signal_arrays, control_arrays=None
):
    """Commit one entry, atomically. Answers its root, or ``None``.

    Every failure is swallowed, which is the contract rather than an
    oversight: a cache that cannot be written is a slower run, and an
    unwritable directory must not take a training job down.

    :param cache_dir: The directory to hold the entry. Created if
        absent.
    :param key: The key, from `window_cache_key`.
    :param peak_windows: The peak window frame, as polars.
    :param negative_windows: The background window frame, as polars.
    :param signal_arrays: The five values `group_track` returns.
    :param control_arrays: The same for the controls, or ``None``.
    :returns: The entry root, or ``None`` if anything failed.

    .. rubric:: Implementation notes

    Commits through `stage_directory` and `commit_directory`, the same
    pair a track is written with, so an entry and a track fail in the
    same direction and the ordering argument has one home.

    The window frames are written with polars rather than pandas,
    which is what upstream used. pandas reaches Parquet through
    pyarrow, and pyarrow arrives in this tree only under the ``pileup``
    extra -- so on a base install an entry could be neither written nor
    read, and because this swallows its failures that would have been
    a silent permanent miss rather than an error.
    """

    root = os.path.join(cache_dir, key)
    staging = None
    try:
        # Before staging, because the sweep globs and the directory
        # creation both run against this parent.
        os.makedirs(cache_dir, exist_ok=True)
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
                # No row total, and every file listed with no count.
                # That is the shape `require_complete` accepts for an
                # entry: an absent total is what excuses absent counts,
                # and a total of zero would be refused.
                "version": MANIFEST_VERSION,
                "files": {name: None for name in files},
                "cache_version": CACHE_VERSION,
                "key": key,
                "signal_channels": int(signal_arrays[4]),
                "control_channels": (
                    None if control_arrays is None else int(control_arrays[4])
                ),
                "n_windows": int(len(peak_windows) + len(negative_windows)),
                "created": time.time(),
            },
        )
        return root
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        return None


def write_cache_arrays(root, offsets, pos, val, chan):
    """Write the four grouped arrays into ``root``, by their fixed names."""

    for name, array in zip(_ARRAY_NAMES, (offsets, pos, val, chan)):
        numpy.save(os.path.join(root, name + ".npy"), array)


def group_track(windows, tracks, bounds=("sig_start", "sig_end")):
    """Group every track's rows by the window they fall inside.

    Keeps only what the windows cover, which is the point: the windows
    are known before any signal is read, so a locus set that touches a
    quarter of the genome reads a quarter of the rows. A position
    covered by two overlapping windows is stored once per window, and
    that duplication is the price of a read that needs no search.

    :param windows: A polars ``DataFrame`` carrying ``chrom`` and the
        two columns ``bounds`` names, one row per window. Row order is
        the window index every reader of the result addresses by.
    :param tracks: One track per channel, in channel order, each of any
        shape `quadrat.stores.track_frame` accepts.
    :param bounds: The window's start and end column names. The signal
        is grouped over the output window and a control over the wider
        input window, which is the whole of why this is a parameter.
    :returns: ``(offsets, pos, val, chan, n_channels)``.

        ``offsets`` is one-dimensional over windows, ``int64``, of
        length ``len(windows) + 1``. Window ``i``'s rows are
        ``pos[offsets[i]:offsets[i + 1]]`` and the same slice of
        ``val`` and ``chan``. There is no channel axis in it.

        Every channel's rows for one window live in that single run,
        interleaved, with the channel index carried per row in
        ``chan`` (``int16``). ``pos`` is ``int32`` and holds the offset
        from the window's own ``bounds[0]``, not an absolute
        coordinate; ``val`` is ``float32``.

        No order is guaranteed within a run. A consumer that scatters
        does not care, and one that does care must sort.

        ``n_channels`` is returned because the arrays cannot recover
        it: a trailing channel with no rows in any window leaves no
        trace in ``chan``.
    :raises ValueError: if there are more channels than ``int16`` can
        index.

    .. rubric:: Implementation notes

    Grouped by window rather than by ``(channel, window)``. Grouping
    the other way would make a read one Python iteration per channel;
    this way every channel scatters in a single vectorised write, which
    is 4.3x faster at ten channels and the difference between the read
    being linear in channel count or not.

    One GIQL ``INTERSECTS`` join per contig and per channel. The join
    returns ``(window index, offset within the window, value)``, which
    is the grouped form already, so the engine does the work this would
    otherwise do with two ``searchsorted`` calls per window.

    Upstream carried a second, hand-rolled grouping engine and chose it
    by default. It is not ported: two implementations of one function
    agreeing is a thing to test rather than a thing to have, and the
    join is what this package is built on.
    """

    polars = import_on_first_use("polars")
    lo_column, hi_column = bounds
    n_windows = len(windows)
    n_channels = len(tracks)

    # Checked before any scan, and before the engine dispatch upstream
    # put it after, because it applies to the layout rather than to one
    # way of filling it.
    if n_channels > numpy.iinfo(numpy.int16).max:
        raise ValueError(
            "more channels than the grouped layout's channel index can "
            "hold: {} against a limit of {}.".format(
                n_channels, numpy.iinfo(numpy.int16).max
            )
        )

    chroms = numpy.asarray([str(value) for value in windows["chrom"]])
    starts = numpy.asarray(windows[lo_column], dtype=numpy.int64)
    ends = numpy.asarray(windows[hi_column], dtype=numpy.int64)

    sql = _signal_sql()
    frames = [track_frame(track) for track in tracks]
    counts = numpy.zeros((n_channels, n_windows), dtype=numpy.int64)
    found = {}

    for chrom in numpy.unique(chroms):
        rows = numpy.flatnonzero(chroms == chrom)
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

        for channel, frame in enumerate(frames):
            rowset = (
                frame.filter(polars.col("chrom") == str(chrom))
                .select(
                    polars.col("chrom").cast(polars.Utf8),
                    polars.col("pos").cast(polars.Int64).alias("cstart"),
                    (polars.col("pos").cast(polars.Int64) + 1).alias("cend"),
                    polars.col("value").cast(polars.Float64),
                )
                .collect()
            )
            if rowset.height == 0:
                continue

            with registered_frames({WINDOWS_TABLE: placed, TRACK_TABLE: rowset}) as pb:
                hits = pb.sql(sql).collect()
            if not hits.height:
                continue

            index = hits["idx"].to_numpy()
            numpy.add.at(counts[channel], index, 1)
            found.setdefault(channel, []).append(
                (index, hits["offset"].to_numpy(), hits["value"].to_numpy())
            )

    offsets = numpy.zeros(n_windows + 1, dtype=numpy.int64)
    numpy.cumsum(counts.sum(axis=0), out=offsets[1:])
    total = int(offsets[-1])

    pos = numpy.empty(total, dtype=numpy.int32)
    val = numpy.empty(total, dtype=numpy.float32)
    chan = numpy.empty(total, dtype=numpy.int16)

    cursor = offsets[:-1].copy()
    for channel in range(n_channels):
        chunks = found.get(channel)
        if not chunks:
            continue
        index = numpy.concatenate([chunk[0] for chunk in chunks])
        offset = numpy.concatenate([chunk[1] for chunk in chunks])
        value = numpy.concatenate([chunk[2] for chunk in chunks])
        # The join answers in no particular order. Sorting by window and
        # then by offset is what makes `cursor` below a running write
        # position per window rather than a search.
        order = numpy.lexsort((offset, index))
        index, offset, value = index[order], offset[order], value[order]
        # `index` is sorted, so `searchsorted` gives each row's first
        # occurrence and the distance from it is its rank in the window.
        rank = numpy.arange(len(index)) - numpy.searchsorted(index, index)
        slots = cursor[index] + rank
        pos[slots] = offset.astype(numpy.int32)
        val[slots] = value.astype(numpy.float32)
        chan[slots] = channel
        numpy.add.at(cursor, index, 1)

    return offsets, pos, val, chan, n_channels


def window_group_sums(windows, tracks, group_sizes):
    """Total signal per window per channel group, without densifying.

    Replaces summing a dense ``(windows, channels, span)`` tensor to
    take a percentile of it, which is the eager form of the same
    question and costs the memory this module exists to avoid.

    The arithmetic is exact rather than approximate: a prefix sum per
    contig and two searches per window, in ``float64``, so a locus's
    total does not depend on how the windows were batched.

    :param windows: A polars ``DataFrame`` carrying ``chrom``,
        ``sig_start`` and ``sig_end``, one row per window.
    :param tracks: One track per channel, in channel order, each a
        path `quadrat.stores.chrom_track` can read.
    :param group_sizes: How many consecutive channels each group
        pools, in group order. Must sum to the channel count.
    :returns: A `torch.Tensor` of shape ``(len(windows),
        len(group_sizes))`` and dtype ``float32``.
    :raises ValueError: if ``group_sizes`` does not sum to the channel
        count.
    """

    torch = import_on_first_use("torch")
    n_windows = len(windows)
    totals = numpy.zeros((n_windows, len(group_sizes)), dtype=numpy.float64)

    # Checked above the empty-window return, not below it as upstream
    # had it: an inconsistent grouping is the caller's mistake whether
    # or not there is a window to apply it to, and refusing it only
    # sometimes is worse than either answer.
    group_of = []
    for group, size in enumerate(group_sizes):
        group_of.extend([group] * size)
    if len(group_of) != len(tracks):
        raise ValueError(
            "group_sizes sums to {} and there are {} tracks; every "
            "channel belongs to exactly one group.".format(len(group_of), len(tracks))
        )

    if n_windows == 0:
        return torch.from_numpy(totals.astype(numpy.float32))

    chroms = numpy.asarray([str(value) for value in windows["chrom"]])
    starts = numpy.asarray(windows["sig_start"], dtype=numpy.int64)
    ends = numpy.asarray(windows["sig_end"], dtype=numpy.int64)

    for chrom in numpy.unique(chroms):
        rows = numpy.flatnonzero(chroms == chrom)
        for channel, track in enumerate(tracks):
            read = chrom_track(track, chrom)
            if read is None:
                continue
            positions, values = read
            cumulative = numpy.concatenate(
                ([0.0], numpy.cumsum(values, dtype=numpy.float64))
            )
            lo = numpy.searchsorted(positions, starts[rows], side="left")
            hi = numpy.searchsorted(positions, ends[rows], side="left")
            totals[rows, group_of[channel]] += cumulative[hi] - cumulative[lo]

    return torch.from_numpy(totals.astype(numpy.float32))


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


def _resolve_residency(residency=None):
    """Where a cache should keep its arrays, resolved on every call.

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

    Resolved per call rather than frozen at import, which is what
    upstream did. ``multiprocessing.set_start_method`` is idiomatically
    called from inside ``if __name__ == "__main__":``, after the import
    that pulled this package in, so a frozen answer samples the
    platform default and a program that deliberately chose ``fork`` to
    get the sharing is handed the mapped form anyway. The same rule
    already governs the cache directory, which this package reads from
    the environment on every call for the same reason.

    :param residency: ``"memory"``, ``"scratch"``, or ``None`` to
        resolve from the start method.
    :returns: ``"memory"`` or ``"scratch"``.
    :raises ValueError: if ``residency`` is neither and not ``None``.
    """

    if residency is None:
        try:
            method = multiprocessing.get_start_method(allow_none=False)
        except RuntimeError:
            # No start method available at all. The mapped form is
            # correct everywhere and only slower, so it is the safe
            # answer rather than a refusal.
            return "scratch"
        return "memory" if method == "fork" else "scratch"
    if residency not in ("memory", "scratch"):
        raise ValueError(
            "residency must be 'memory', 'scratch' or None, got {!r}.".format(residency)
        )
    return residency


def _signal_sql():
    """The grouping join, transpiled for this engine once per process.

    Transpiled rather than written, because the join is an interval
    predicate and the target dialect has no such operator: giql is what
    turns ``INTERSECTS`` into the bound comparisons the engine runs.
    Once per process because the transpilation is pure and the query
    never varies -- the contig and the channel vary by what is
    registered under the two table names, not by the text.
    """

    if "signal" not in _transpiled:
        giql = import_on_first_use("giql")
        _transpiled["signal"] = giql.transpile(
            "SELECT w.idx, c.cstart - w.wstart AS offset, c.value "
            "FROM {} AS w JOIN {} AS c ON w.interval "
            "INTERSECTS c.interval".format(WINDOWS_TABLE, TRACK_TABLE),
            tables=[
                giql.Table(
                    WINDOWS_TABLE,
                    chrom_col="chrom",
                    start_col="wstart",
                    end_col="wend",
                    strand_col=None,
                ),
                giql.Table(
                    TRACK_TABLE,
                    chrom_col="chrom",
                    start_col="cstart",
                    end_col="cend",
                    strand_col=None,
                ),
            ],
            dialect="datafusion-bio",
        )
    return _transpiled["signal"]
