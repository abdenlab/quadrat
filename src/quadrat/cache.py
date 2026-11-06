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

import numpy
import torch

from quadrat._engine import (
    TRACK_TABLE,
    WINDOWS_TABLE,
    import_on_first_use,
    registered_frames,
)
from quadrat.stores import chrom_track, track_frame

#: The transpiled grouping join, per process. A plain dict rather than
#: SCREAMING_CASE because the binding is constant and the referent is
#: not: `_signal_sql` fills it on first use.
_transpiled = {}


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
