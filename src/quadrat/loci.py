"""Locus loading and extraction: windows from locus files, tensors from windows.

Provides `load_windows`, which turns locus files into a window frame;
`exclusion_mask` and `require_track_contigs`, which decide what of a
window frame may be read; `extract_signal` and `extract_sequences`,
which read it; and `extract_loci`, which composes all of them into
`ExtractedLoci`.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`.
"""

import csv
import functools
import os
import typing

import numpy

from quadrat._engine import (
    EXCLUSION_BINS_TABLE,
    LOCUS_BINS_TABLE,
    fspath_or_none,
    import_on_first_use,
    name_list,
    registered_frames,
    transpile_intervals,
)
from quadrat.cache import group_track, require_windows, track_list, window_columns
from quadrat.reference import chrom_lengths
from quadrat.stores import SequenceStore, track_frame
from quadrat.tables import track_contigs

#: Width of the bins `exclusion_mask` tests on, in bases.
EXCLUSION_BIN_WIDTH = 100


#: The start of a BED line that holds no interval: a comment, or a
#: track or browser line.
_BED_HEADER = r"#|(track|browser)(\s|$)"


class ExtractedLoci(typing.NamedTuple):
    """Hold the tensors one call to `extract_loci` read.

    The first axis of every tensor but ``kept`` is the kept loci, in
    locus order, so row ``i`` of each describes the same locus.

    :param sequences: The one-hot input windows, as `extract_sequences`
        returns them.
    :param signals: The signal over the output windows, as
        `extract_signal` returns it, or ``None`` where no signal track
        was given.
    :param controls: The control signal over the input windows, in the
        same form, or ``None`` where no control track was given.
    :param kept: A boolean `torch.Tensor` with one entry per locus the
        ``chroms`` filter left, in locus order, true where the locus
        was read. Its true entries number the rows of the other
        tensors.
    """

    sequences: typing.Any
    signals: typing.Any
    controls: typing.Any
    kept: typing.Any


def load_windows(
    loci, sequences, in_window, out_window, chroms=None, summits=False, max_jitter=0
):
    """Load loci and derive every window bound around each of them.

    The frame has one row per locus the ``chroms`` filter leaves, and
    drops nothing else: a window reaching off its contig stays, marked
    by ``in_bounds``. Several locus sets are interleaved, the first
    locus of each set in the order the sets were given, then the second
    of each, a shorter set simply running out. Row order is the locus
    order every reader of the frame addresses by.

    A locus is centred on ``start + (end - start) // 2``, or on its
    summit under ``summits``. Every bound is half-open, and an odd
    width puts its extra base on the right of the centre.

    ===============  ===========  =========================================
    column           dtype        contents
    ===============  ===========  =========================================
    ``chrom``        ``Utf8``     the contig
    ``mid``          ``Int64``    the centre
    ``outer_start``  ``Int64``    start of the union of both windows
    ``outer_end``    ``Int64``    end of that union
    ``in_bounds``    ``Boolean``  whether the union lies on the contig
    ``sig_start``    ``Int64``    start of the output window
    ``sig_end``      ``Int64``    end of the output window
    ``in_start``     ``Int64``    start of the input window
    ``in_end``       ``Int64``    end of the input window
    ===============  ===========  =========================================

    Each window is ``max_jitter`` wider on both sides than its nominal
    width, so every draw a sampler may make lies inside it. A union
    ending exactly at the contig's length is in bounds.

    Needs the ``pileup`` extra only where
    `quadrat.reference.chrom_lengths` falls back to a scan for the
    reference's lengths.

    :param loci: A BED file, as a `str`, `bytes` or `os.PathLike`; a
        `pandas.DataFrame` or `polars.DataFrame` whose first three
        columns are contig, start and end; or a `list` or `tuple` of
        any of those. A file may be compressed, the compression being
        inferred from its suffix, e.g., ``.gz``. Blank lines, comments,
        and ``track`` and ``browser`` lines are skipped, and a file
        with no interval in it holds no loci.
    :param sequences: The reference the loci are placed on, as
        `quadrat.reference.chrom_lengths` takes it.
    :param in_window: The input window's width, in bases.
    :param out_window: The output window's width, in bases. ``0`` where
        no signal is to be read, which leaves the input window alone to
        decide ``in_bounds``.
    :param chroms: The contigs to keep, as an iterable of names, or
        ``None`` for all. Applied to each locus set before the sets are
        interleaved.
    :param summits: Whether to centre each locus on its summit, i.e.,
        its start plus the offset in the tenth column. The centre is
        then the summit exactly, whatever the locus's width.
    :param max_jitter: The largest offset a draw may be shifted by, in
        bases.
    :returns: A `polars.DataFrame` with the columns above.
    :raises TypeError: if ``loci`` is of none of those shapes, if
        ``chroms`` is a single name rather than a collection of them,
        or if ``sequences`` is not a path.
    :raises FileNotFoundError: if nothing exists at ``sequences``.
    :raises ValueError: if ``chroms`` leaves no locus where the loci
        hold some, which is a naming mismatch between the two; if any
        locus is on a contig the reference does not name; or if a
        summit is negative or lies past its locus's end. Loci that
        hold no locus to begin with are not refused.
    :raises ImportError: if the reference has no usable index, none
        can be written beside it, and the ``pileup`` extra is not
        installed.

    .. rubric:: Implementation notes

    The two contig refusals exist because both mismatches are silent
    otherwise. A filter naming ``chr1`` over loci naming ``1`` answers
    an empty frame, which downstream is a training set with nothing in
    it. Telling that from a set that was empty beforehand costs a
    second read of the loci, which happens only on the way to raising.

    ``in_bounds`` counts the odd base of an odd-width window, so a
    window marked in bounds can always be read whole.

    A missing reference is refused here because `chrom_lengths` would
    not refuse it: it reads as a reference with no usable index, and
    the scan that follows fails in the engine's words, or for want of
    the extra.
    """

    pandas = import_on_first_use("pandas")
    polars = import_on_first_use("polars")
    sources = _locus_sources("loci", loci)
    chroms = _contig_names(chroms)
    # Ahead of `chrom_lengths`, for the reason in the notes. A
    # reference that is no path is left for it to refuse by type.
    path = fspath_or_none(sequences)
    if path is not None:
        try:
            os.stat(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "{} is not a reference; there is no such file or directory.".format(path)
            ) from None
    lengths = chrom_lengths(sequences)

    frame = _interleave_loci(sources, chroms, summits)
    if chroms is not None and not len(frame):
        unfiltered = _interleave_loci(sources, None, summits)
        if len(unfiltered):
            raise ValueError(
                "none of the {} loci in {} is on a contig `chroms` names: "
                "the loci name {} and `chroms` names {}. Every locus would "
                "be dropped and the window set would be empty. Contig names "
                "must match exactly, so `1` and `chr1` are different "
                "contigs. Pass `chroms` in the naming the loci use, or "
                "rename the loci.".format(
                    len(unfiltered),
                    _described(sources),
                    name_list(pandas.unique(unfiltered["chrom"]), "no contigs"),
                    name_list(chroms, "no contigs"),
                )
            )

    sizes = frame["chrom"].map(lengths)
    absent = sizes.isnull()
    if absent.any():
        raise ValueError(
            "{} of the {} loci in {} cannot be placed: {} does not name {}. "
            "The reference names {}. Contig names must match exactly, so "
            "`1` and `chr1` are different contigs. Use loci named as the "
            "reference names its contigs, or drop the others with "
            "`chroms`.".format(
                int(absent.sum()),
                len(frame),
                _described(sources),
                fspath_or_none(sequences),
                name_list(pandas.unique(frame["chrom"][absent]), "no contigs"),
                name_list(lengths, "no contigs"),
            )
        )
    sizes = sizes.to_numpy().astype(numpy.int64)

    start = frame["start"].to_numpy().astype(numpy.int64)
    end = frame["end"].to_numpy().astype(numpy.int64)
    mid = start + (end - start) // 2

    in_left, in_right = in_window // 2, in_window // 2 + in_window % 2
    out_left, out_right = out_window // 2, out_window // 2 + out_window % 2
    outer_start = mid - max(in_left, out_left) - max_jitter
    outer_end = mid + max(in_right, out_right) + max_jitter

    return polars.DataFrame(
        {
            "chrom": frame["chrom"].tolist(),
            "mid": mid,
            "outer_start": outer_start,
            "outer_end": outer_end,
            "in_bounds": (outer_start >= 0) & (outer_end <= sizes),
            "sig_start": mid - out_left - max_jitter,
            "sig_end": mid + out_right + max_jitter,
            "in_start": mid - in_left - max_jitter,
            "in_end": mid + in_right + max_jitter,
        },
        schema={
            "chrom": polars.Utf8,
            "mid": polars.Int64,
            "outer_start": polars.Int64,
            "outer_end": polars.Int64,
            "in_bounds": polars.Boolean,
            "sig_start": polars.Int64,
            "sig_end": polars.Int64,
            "in_start": polars.Int64,
            "in_end": polars.Int64,
        },
    )


def exclusion_mask(windows, exclusion_lists):
    """Return which windows touch an excluded region, on a grid of fixed bins.

    The test is made on bins `EXCLUSION_BIN_WIDTH` bases wide rather
    than on bases. A region marks every bin from the one holding its
    start through the one holding its end, and a window is flagged when
    any bin from the one holding ``outer_start`` through the one
    holding ``outer_end`` is marked. Both ends are the exclusive ones,
    so a region or a window ending exactly on a bin edge also reaches
    the bin that begins there.

    A region on a contig no window is on flags nothing, and neither
    does a window on a contig no region names.

    Needs the ``pileup`` extra unless there is no window or no region.
    Concurrent calls in one process are unsupported; `quadrat._engine`
    says why.

    :param windows: A `polars.DataFrame` carrying ``chrom``,
        ``outer_start`` and ``outer_end``, as `load_windows` returns.
    :param exclusion_lists: The excluded regions, in any shape
        `load_windows` takes loci in.
    :returns: A boolean `numpy.ndarray` with one entry per window, in
        row order, true where the window is excluded.
    :raises TypeError: if ``windows`` is not a `polars.DataFrame`, or
        if ``exclusion_lists`` is of no shape a locus set can take.
    :raises ImportError: if the ``pileup`` extra is not installed.

    .. rubric:: Implementation notes

    Both sides are half-open runs of bins, so the test is one interval
    join between two bin tables rather than a scan of either.
    """

    polars = import_on_first_use("polars")
    require_windows("exclusion_mask", windows)
    regions = _interleave_loci(_locus_sources("exclusion_lists", exclusion_lists))

    mask = numpy.zeros(len(windows), dtype=bool)
    if not len(regions) or not len(windows):
        return mask

    bins = {"chrom": polars.Utf8, "bstart": polars.Int64, "bend": polars.Int64}
    region_bins = polars.DataFrame(
        {
            "chrom": regions["chrom"].tolist(),
            "bstart": _bins(regions["start"].to_numpy()),
            "bend": _bins(regions["end"].to_numpy()) + 1,
        },
        schema=bins,
    )
    window_bins = polars.DataFrame(
        {
            "idx": numpy.arange(len(windows), dtype=numpy.int64),
            "chrom": windows["chrom"].cast(polars.Utf8),
            "bstart": _bins(windows["outer_start"].to_numpy()),
            "bend": _bins(windows["outer_end"].to_numpy()) + 1,
        },
        schema={"idx": polars.Int64, **bins},
    )

    with registered_frames(
        {EXCLUSION_BINS_TABLE: region_bins, LOCUS_BINS_TABLE: window_bins}
    ) as pb:
        hits = pb.sql(_exclusion_sql()).collect()
    mask[hits["idx"].to_numpy()] = True
    return mask


def require_track_contigs(chroms, tracks, label="signals"):
    """Raise unless some track holds a position on some contig in ``chroms``.

    What is refused is the whole grid being empty: every track holding
    nothing on every contig asked about. A track that holds nothing on
    any of them is accepted while another holds something, and so is a
    contig no track holds a position on while another contig has some.

    Every track is resolved before any is compared, so one that is no
    track is refused whichever track would have settled the question.

    Nothing is checked when ``chroms`` or ``tracks`` is empty, there
    being nothing left to disagree, and no track is then resolved.

    :param chroms: The contigs the windows to be read are on, as an
        iterable of names.
    :param tracks: The tracks to be read, as
        `quadrat.cache.track_list` takes them, each of any shape
        `quadrat.stores.track_frame` accepts.
    :param label: The name the caller knows ``tracks`` by, for the
        message.
    :raises ValueError: if no track holds a position on any contig in
        ``chroms``, or if a track directory does not verify complete.
    :raises TypeError: if ``chroms`` is a single name rather than a
        collection of them, if ``tracks`` is a single track or an
        unordered collection of them, or if a track is of no shape a
        track can take.
    :raises FileNotFoundError: if a track path does not exist.

    .. rubric:: Implementation notes

    The check is coarse because nothing finer separates a mismatch from
    ordinary data. A window on a contig the tracks never name reads as
    zeros, and so does a window where nothing was measured, so no
    single window can be refused. One quiet track is legitimate, e.g.,
    a low-signal modality or one strand of a stranded pair, and so is
    a window contig with no position on it, which is routine for the
    small unplaced contigs. Only the whole grid being empty is never
    anything but a naming mismatch, and it is the one that trains: the
    windows have the usual count and the usual shape, and every value
    in them is zero.
    """

    chroms = set(_contig_names(chroms))
    tracks = track_list("require_track_contigs", "tracks", tracks)
    if not chroms or not tracks:
        return

    # A loop of its own: resolved before any is compared, as the
    # docstring promises.
    for track in tracks:
        track_frame(track)

    held = set()
    for track in tracks:
        names = track_contigs(track)
        if names & chroms:
            return
        held |= names

    paths = [path for path in map(fspath_or_none, tracks) if path is not None]
    raise ValueError(
        "no `{}` track holds a position on any contig the windows are on: "
        "the tracks name {} and the windows name {}. Every window would "
        "read as zeros. Contig names must match exactly, so `1` and `chr1` "
        "are different contigs. Use tracks piled up against the reference "
        "the loci are named for. The tracks are {}.".format(
            label,
            name_list(sorted(held), "no contigs"),
            name_list(sorted(chroms), "no contigs"),
            name_list(paths, "in-memory frames"),
        )
    )


def extract_signal(windows, tracks, bound_columns=("sig_start", "sig_end")):
    """Return the dense signal of every window, one channel per track.

    A position recorded twice in a track is totalled, as
    `quadrat.stores.TrackStore.window` totals it. A contig a track does
    not hold reads as that channel's zeros, and so does every window
    when the tracks and the windows name their contigs differently:
    nothing here tells that mismatch from an absence of signal. A
    caller who needs it refused applies `require_track_contigs` first.

    Needs the ``pileup`` extra. Concurrent calls in one process are
    unsupported, as they are for `quadrat.cache.group_track`.

    :param windows: A `polars.DataFrame` carrying ``chrom`` and the two
        columns ``bound_columns`` names, one row per window, every
        window the same width.
    :param tracks: One track per channel, in channel order, as
        `quadrat.cache.group_track` takes them.
    :param bound_columns: The window's start and end column names. The
        signal is read over the output window and a control over the
        input window.
    :returns: A `torch.Tensor` of shape ``(len(windows), len(tracks),
        width)`` and dtype ``float32``, zero wherever a track records
        nothing. ``width`` is ``0`` for an empty window frame.
    :raises TypeError: if ``windows`` is not a `polars.DataFrame`, if
        ``tracks`` is a single track or an unordered collection of
        them, or if a track is of no shape a track can take.
    :raises ValueError: if the windows are not all one non-negative
        width, or if a track directory does not verify complete.
    :raises FileNotFoundError: if a track path does not exist.
    :raises ImportError: if the ``pileup`` extra is not installed.

    .. rubric:: Implementation notes

    The grouping is `quadrat.cache.group_track`'s, scattered dense. A
    second join answering the same rows in another layout would be a
    second implementation to keep in agreement with the first.

    Going through the grouping costs a transient the dense tensor does
    not need: the grouping holds every contig's hits until it returns,
    at the figure its own notes give, where a scatter made contig by
    contig would hold one contig's hits.
    """

    torch = import_on_first_use("torch")
    require_windows("extract_signal", windows)
    tracks = track_list("extract_signal", "tracks", tracks)
    width = _window_width("extract_signal", windows, bound_columns)

    signal = numpy.zeros((len(windows), len(tracks), width), dtype=numpy.float32)
    offsets, pos, val, chan, _ = group_track(windows, tracks, bound_columns)
    rows = numpy.repeat(numpy.arange(len(windows)), numpy.diff(offsets))
    # `add.at` rather than an assignment: the totalling the docstring
    # promises for a position recorded twice.
    numpy.add.at(signal, (rows, chan, pos), val)
    return torch.from_numpy(signal)


def extract_sequences(windows, sequences, alphabet=("A", "C", "G", "T"), ignore=("N",)):
    """Return the one-hot input window of every row of ``windows``.

    Each window is read through one `quadrat.stores.SequenceStore`,
    which states what a read holds, how its residues are encoded, what
    it writes beside the reference, and that it never scans it.
    Nothing here needs the ``pileup`` extra. An empty window frame is
    answered without the reference being opened.

    :param windows: A `polars.DataFrame` carrying ``chrom``,
        ``in_start`` and ``in_end``, one row per window, every window
        the same width and wholly on its contig.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`.
    :param alphabet: The residues that get a row, in row order.
    :param ignore: Residues that encode to an all-zero column.
    :returns: A `torch.Tensor` of shape ``(len(windows),
        len(alphabet), width)`` and dtype ``int8``. ``width`` is ``0``
        for an empty window frame.
    :raises TypeError: if ``windows`` is not a `polars.DataFrame`, or
        if ``sequences`` is not a path.
    :raises ValueError: if the windows are not all one non-negative
        width, if a window is on a contig the reference does not name,
        if a window reaches outside its contig, or for a residue
        `quadrat.stores.SequenceStore.window` refuses.
    :raises OSError: if the reference cannot be read, or has no index
        and none can be written beside it.

    .. rubric:: Implementation notes

    A window reaching off its contig is refused rather than left to
    the store, which would pad it with all-zero columns. Those read as
    ignored residues, so the tensor would not say which of its windows
    the reference never covered.

    The rows are stacked once rather than written one at a time into
    a tensor allocated beforehand, which measured at 54 ms against
    33 ms over 3,000 windows of 2,114 bases.
    """

    torch = import_on_first_use("torch")
    require_windows("extract_sequences", windows)
    path = fspath_or_none(sequences)
    if path is None:
        raise TypeError(
            "extract_sequences reads residues from a FASTA: `sequences` "
            "must be a str, bytes or os.PathLike, got {}.".format(
                type(sequences).__name__
            )
        )
    bound_columns = ("in_start", "in_end")
    width = _window_width("extract_sequences", windows, bound_columns)
    store = SequenceStore(path, alphabet, ignore)

    starts, ends, groups = window_columns(windows, bound_columns)
    if not groups:
        return torch.zeros((0, len(store.alphabet), width), dtype=torch.int8)

    lengths = store.contig_lengths()
    unknown = [chrom for chrom, _ in groups if chrom not in lengths]
    if unknown:
        raise ValueError(
            "{} does not name the contig{} {}. The reference names {}. "
            "Contig names must match exactly, so `1` and `chr1` are "
            "different contigs.".format(
                path,
                "" if len(unknown) == 1 else "s",
                name_list(unknown, "no contigs"),
                name_list(lengths, "no contigs"),
            )
        )

    for chrom, rows in groups:
        outside = rows[(starts[rows] < 0) | (ends[rows] > lengths[chrom])]
        if len(outside):
            row = int(outside[0])
            raise ValueError(
                "extract_sequences reads windows that lie on their contig: "
                "window {} is [{}, {}) on `{}`, which is {} bases long. "
                "Drop the windows `in_bounds` marks false before "
                "extracting.".format(
                    row, int(starts[row]), int(ends[row]), chrom, lengths[chrom]
                )
            )
    encoded = [None] * len(windows)
    for chrom, rows in groups:
        for row in rows:
            encoded[row] = store.window(chrom, starts[row], ends[row])
    return torch.stack(encoded)


def extract_loci(
    loci,
    sequences,
    in_window,
    out_window=None,
    *,
    signals=None,
    controls=None,
    chroms=None,
    max_jitter=0,
    min_counts=None,
    max_counts=None,
    target_idx=0,
    summits=False,
    alphabet=("A", "C", "G", "T"),
    ignore=("N",),
    exclusion_lists=None,
):
    """Extract the sequence and signal of every locus that can be read.

    The loci become windows as `load_windows` derives them, and a locus
    is read when all of these hold, tested in this order:

    1. its windows lie on its contig, i.e., ``in_bounds``;
    2. `exclusion_mask` does not flag it;
    3. its count is within ``min_counts`` and ``max_counts``, the count
       being the total of signal track ``target_idx`` over its output
       window. A count equal to either threshold is kept.

    Between the second test and the third, `require_track_contigs` is
    applied to ``signals`` and to ``controls`` over the contigs of the
    loci still standing. An exclusion list that removes every locus
    therefore raises nothing, and neither does a count filter that
    keeps none: both answer tensors with no rows.

    Controls are read over the input window, at the width the
    sequence is. The sequences and the controls are ``in_window + 2 *
    max_jitter`` wide and the signals ``out_window + 2 * max_jitter``,
    and they are those widths when no locus is read as well, so an
    empty result differs from a full one in its first axis alone.

    Needs the ``pileup`` extra to read ``signals`` and ``controls``
    and to apply ``exclusion_lists``, and for nothing else. The
    reference is opened as `quadrat.stores.SequenceStore` opens it
    before any locus is loaded, whether or not one is then read. One
    with no index and nowhere to write one is therefore refused,
    whatever else was asked for, where `load_windows` alone would scan
    it. Concurrent calls in one process are unsupported, as they are
    for `exclusion_mask` and `extract_signal`.

    :param loci: The loci, as `load_windows` takes them.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`.
    :param in_window: The input window's width, in bases.
    :param out_window: The output window's width, in bases. Required
        with ``signals`` and without effect otherwise, including on
        which loci are in bounds.
    :param signals: One track per signal channel, in channel order, as
        `extract_signal` takes them, or ``None`` to read no signal.
        Every track is opened before any locus is read, so one that is
        no track is refused whether or not a locus reaches it.
    :param controls: One track per control channel, in the same form,
        or ``None``.
    :param chroms: The contigs to keep, as `load_windows` takes them.
    :param max_jitter: The largest offset a draw may be shifted by, in
        bases; every window is that much wider on both sides.
    :param min_counts: The lowest count a kept locus may have, or
        ``None`` for no floor.
    :param max_counts: The highest count a kept locus may have, or
        ``None`` for no ceiling.
    :param target_idx: The signal track the count is taken on, as an
        index into ``signals``; negative values count from the end.
    :param summits: Whether loci are centred on their summits, as
        `load_windows` takes it.
    :param alphabet: The residues that get a row, in row order.
    :param ignore: Residues that encode to an all-zero column.
    :param exclusion_lists: The excluded regions, as `exclusion_mask`
        takes them, or ``None`` to exclude nothing.
    :returns: An `ExtractedLoci`.
    :raises TypeError: if ``signals`` or ``controls`` is a single
        track, an unordered collection of them, or holds something that
        is no track, or for any argument `load_windows` refuses by
        type.
    :raises ValueError: if ``signals`` is given without ``out_window``;
        if a count threshold is given without ``signals``; if
        ``target_idx`` names no signal track where ``signals`` holds
        one or a count threshold is given; if a track directory does
        not verify complete; for the two naming mismatches
        `load_windows` refuses; for the one `require_track_contigs`
        refuses; or for a residue `extract_sequences` refuses.
    :raises FileNotFoundError: if a track path does not exist.
    :raises OSError: if the reference cannot be read, or has no index
        and none can be written beside it.
    :raises ImportError: if the ``pileup`` extra is needed and not
        installed.

    .. rubric:: Implementation notes

    The placement of the track check is what keeps it free of false
    positives. After the exclusion mask, because a set the exclusion
    lists emptied has no contig left to compare; before the count
    filter, which runs on signal already read and so cannot have
    emptied the set by then.
    """

    torch = import_on_first_use("torch")
    track_sets = {}
    for label, tracks in (("signals", signals), ("controls", controls)):
        if tracks is None:
            continue
        track_sets[label] = track_list("extract_loci", label, tracks)
        # Opened here for the reason the ``signals`` parameter gives.
        for track in track_sets[label]:
            track_frame(track)
    signals, controls = track_sets.get("signals"), track_sets.get("controls")

    counting = min_counts is not None or max_counts is not None
    if signals is None:
        if counting:
            raise ValueError(
                "extract_loci counts over a signal track: min_counts and "
                "max_counts need `signals`, and none was given."
            )
        out_window = 0
    elif out_window is None:
        raise ValueError(
            "extract_loci reads signal over the output window: `signals` "
            "needs `out_window`, and none was given."
        )
    elif (signals or counting) and not -len(signals) <= target_idx < len(signals):
        raise ValueError(
            "extract_loci counts over one signal track: target_idx {} "
            "names none of the {} given.".format(target_idx, len(signals))
        )

    # Ahead of `load_windows`, as the docstring promises. A reference
    # that is no path is left for `load_windows` to refuse by type.
    if fspath_or_none(sequences) is not None:
        SequenceStore(sequences).contig_lengths()

    windows = load_windows(
        loci,
        sequences,
        in_window,
        out_window,
        chroms=chroms,
        summits=summits,
        max_jitter=max_jitter,
    )
    keep = windows["in_bounds"].to_numpy().copy()
    if exclusion_lists is not None:
        keep &= ~exclusion_mask(windows, exclusion_lists)

    # Placed for the reason in the notes.
    if keep.any():
        on = windows["chrom"].filter(keep).unique().to_list()
        for label, tracks in track_sets.items():
            require_track_contigs(on, tracks, label)

    signal = None
    if signals is not None and keep.any():
        signal = extract_signal(windows.filter(keep), signals)
        if min_counts is not None or max_counts is not None:
            counts = signal.numpy()[:, target_idx].sum(axis=1)
            within = numpy.ones(len(counts), dtype=bool)
            if min_counts is not None:
                within &= ~(counts < min_counts)
            if max_counts is not None:
                within &= ~(counts > max_counts)
            keep[numpy.flatnonzero(keep)[~within]] = False
            signal = signal[torch.from_numpy(within)]

    kept = windows.filter(keep)
    if not len(kept):
        # At the nominal widths, as the docstring promises: the
        # readers below answer no width for no windows, having only
        # the windows to take one from.
        in_width, out_width = in_window + 2 * max_jitter, out_window + 2 * max_jitter
        empty = {
            label: torch.zeros((0, len(tracks), width), dtype=torch.float32)
            for label, tracks, width in (
                ("signals", signals, out_width),
                ("controls", controls, in_width),
            )
            if tracks is not None
        }
        return ExtractedLoci(
            sequences=torch.zeros((0, len(list(alphabet)), in_width), dtype=torch.int8),
            signals=empty.get("signals"),
            controls=empty.get("controls"),
            kept=torch.from_numpy(keep),
        )

    control = None
    if controls is not None:
        control = extract_signal(kept, controls, bound_columns=("in_start", "in_end"))

    return ExtractedLoci(
        sequences=extract_sequences(kept, sequences, alphabet=alphabet, ignore=ignore),
        signals=signal,
        controls=control,
        kept=torch.from_numpy(keep),
    )


def _window_width(caller, windows, bound_columns):
    """Return the one width every window has, or ``0`` for no windows."""

    lo_column, hi_column = bound_columns
    widths = (windows[hi_column] - windows[lo_column]).to_numpy()
    if not len(widths):
        return 0
    width = int(widths[0])
    if width < 0 or (widths != width).any():
        raise ValueError(
            "{} answers one tensor, so every window must have the same "
            "non-negative width between `{}` and `{}`. Got widths from {} "
            "to {}.".format(
                caller, lo_column, hi_column, int(widths.min()), int(widths.max())
            )
        )
    return width


def _contig_names(chroms):
    """Return a contig filter as a list of `str` names, or ``None`` for none."""

    if chroms is None:
        return None
    if isinstance(chroms, (str, bytes)):
        raise TypeError(
            "`chroms` is a collection of contig names, and a single name "
            "would be read as its characters. Pass [{!r}].".format(chroms)
        )
    return [str(chrom) for chrom in chroms]


def _locus_sources(label, loci):
    """Return a locus argument as a list of path strings and pandas frames.

    A single path or frame is wrapped, because one file is the ordinary
    locus set, and only a `list` or `tuple` is read as several. How many
    sets there are is no axis of anything read from them: they are
    interleaved into one locus order. A track argument answers a single
    item the other way, for the reason `quadrat.cache.track_list`
    gives. A polars frame is answered as the pandas frame of the same
    columns.
    """

    pandas = import_on_first_use("pandas")
    polars = import_on_first_use("polars")
    items = loci if isinstance(loci, (list, tuple)) else [loci]
    sources = []
    for item in items:
        path = fspath_or_none(item)
        if path is not None:
            sources.append(path)
        elif isinstance(item, pandas.DataFrame):
            sources.append(item)
        elif isinstance(item, polars.DataFrame):
            # Through plain lists, which needs no pyarrow where
            # ``to_pandas`` does.
            sources.append(pandas.DataFrame(item.to_dict(as_series=False)))
        else:
            raise TypeError(
                "`{}` is a BED file path, a pandas or polars DataFrame "
                "whose first three columns are contig, start and end, or a "
                "list or tuple of those. Got {}.{}.".format(
                    label, type(item).__module__, type(item).__qualname__
                )
            )
    return sources


def _described(sources):
    """Return how a message names a locus argument: its paths, and its frames by count."""

    paths = [source for source in sources if isinstance(source, str)]
    frames = len(sources) - len(paths)
    if not paths:
        if frames == 1:
            return "the locus frame given"
        return "the {} locus frames given".format(frames)
    if not frames:
        return ", ".join(paths)
    return "{} and {} locus frame{}".format(
        ", ".join(paths), frames, "" if frames == 1 else "s"
    )


def _bins(coordinates):
    """Return the exclusion bin each coordinate falls in, as ``int64``."""

    return coordinates.astype(numpy.int64) // EXCLUSION_BIN_WIDTH


@functools.cache
def _exclusion_sql():
    """Return the exclusion join, transpiled once per process.

    Once per process because the query never varies: what varies is
    what is registered under the two table names.
    """

    return transpile_intervals(
        "SELECT DISTINCT w.idx FROM {} AS w JOIN {} AS e ON w.interval "
        "INTERSECTS e.interval".format(LOCUS_BINS_TABLE, EXCLUSION_BINS_TABLE),
        (LOCUS_BINS_TABLE, "bstart", "bend"),
        (EXCLUSION_BINS_TABLE, "bstart", "bend"),
    )


# `_read_bed` and `_interleave_loci` are adapted from tangermeme 1.6.0
# (``tangermeme/io.py``), which carries them as underscore-prefixed
# names this package must not import. They are reproduced under its
# licence:
#
# MIT License
#
# Copyright (c) 2024-2026 Jacob Schreiber
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def _read_bed(path, columns, names):
    """Return the named columns of a BED file's intervals.

    Adapted from tangermeme; the notice above this function applies.
    `load_windows` states which lines are skipped, how a compressed
    file is read, and what a file holding no interval answers.

    .. rubric:: Implementation notes

    pandas takes the column count from the first line, so a file with
    header lines is read a second time without them rather than once
    with its columns misassigned.
    """

    pandas = import_on_first_use("pandas")
    options = dict(
        sep="\t",
        usecols=columns,
        header=None,
        index_col=False,
        names=names,
        # Text, or a file naming only ``01`` would be read as the
        # integer 1, which is another contig.
        dtype={"chrom": str},
    )
    try:
        frame = pandas.read_csv(path, **options)
    except pandas.errors.ParserError:
        # No line is as wide as an interval, which a file of header
        # lines alone is; the line scan below settles which it is.
        frame = None
    if frame is not None and not frame["chrom"].astype(str).str.match(_BED_HEADER).any():
        return frame

    # One row per line, blank lines included, so that a row's index is
    # its line number, which is what ``skiprows`` takes.
    lines = pandas.read_csv(
        path,
        sep="\t",
        usecols=[0],
        header=None,
        names=["chrom"],
        dtype=str,
        skip_blank_lines=False,
        quoting=csv.QUOTE_NONE,
    )
    skip = numpy.flatnonzero(lines["chrom"].fillna("").str.match(_BED_HEADER))
    return pandas.read_csv(path, skiprows=skip.tolist(), **options)


def _interleave_loci(sources, chroms=None, summits=False):
    """Return every locus of ``sources`` as one frame, the sets interleaved.

    Adapted from tangermeme; the notice above `_read_bed` applies.
    `load_windows` states the order, the filter and the summit rule.
    Under ``summits`` a locus is re-centred at ``width // 2`` either
    side of its summit, so one of odd width comes back a base narrower
    with its centre unmoved, as tangermeme's does.

    :param sources: Path strings and pandas frames, as `_locus_sources`
        returns them.
    :param chroms: The contigs to keep, as a list of `str`, or ``None``.
    :param summits: Whether to re-centre each locus on its summit.
    :returns: A `pandas.DataFrame` with ``chrom`` as `str`, ``start``
        and ``end``.

    .. rubric:: Implementation notes

    The filter runs before the interleaving, so that the sets stay
    balanced over the contigs kept rather than over the ones dropped.
    """

    pandas = import_on_first_use("pandas")
    columns = [0, 1, 2] + ([9] if summits else [])
    names = ["chrom", "start", "end"] + (["summit"] if summits else [])

    frames = []
    for position, source in enumerate(sources):
        if isinstance(source, str):
            frame = _read_bed(source, columns, names)
        else:
            frame = source.iloc[:, columns].copy()
            frame.columns = names

        # A frame handed over may carry contigs named 1, 2, ... as
        # integers, which match no name a reference or a track carries.
        frame["chrom"] = frame["chrom"].astype(str)

        if summits:
            if frame["summit"].min() < 0:
                raise ValueError("Summits cannot be negative values.")
            if ((frame["summit"] + frame["start"]) > frame["end"]).any():
                raise ValueError("Summit + start cannot be larger than end.")

            mid = frame["start"] + frame["summit"]
            width = frame["end"] - frame["start"]
            frame["start"] = mid - width // 2
            frame["end"] = mid + width // 2
            frame = frame.drop(columns=["summit"])

        if chroms is not None:
            frame = frame[numpy.isin(frame["chrom"], chroms)].copy()

        frame["idx"] = numpy.arange(len(frame)) * len(sources) + position
        frames.append(frame)

    if not frames:
        return pandas.DataFrame({"chrom": [], "start": [], "end": []})
    return pandas.concat(frames).set_index("idx").sort_index().reset_index(drop=True)
