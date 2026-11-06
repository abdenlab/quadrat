"""Alignments to tracks.

One call per channel. ``pileup_alignments`` pools every alignment it is
given into one table, exactly as a single ``bam2bw`` invocation would,
and writes it as a directory of per-contig Parquet files.

The genome is read in contig batches rather than at once, and the batch
count is solved rather than configured -- see
`quadrat.pileup.plan`. Only one batch's positions are ever live,
so the peak follows the batch rather than the alignment, and because a
contig predicate reaches the BAM index the batch costs only the records
it selects.

A module of its own rather than the body of ``quadrat.pileup``, because
``run`` dispatches to this and ``quadrat.pileup`` re-exports both: with
the implementation in the package ``__init__`` the two could only be
wired together by importing inside a function, which is the circular
dependency the module split exists to remove.
"""

import collections
import contextlib
import logging
import os
import sys
import warnings

from quadrat._engine import (
    SIZES_TABLE,
    EngineOptionWarning,
    engine,
    exclusive_pileup,
    fspath_or_none,
    import_on_first_use,
    registered_frames,
)
from quadrat.alignments import (
    bam_contigs,
    contig_batches,
    deregister_alignments,
    register_alignments,
    require_reference_contigs,
    warn_unindexed_split,
)
from quadrat.pileup.plan import n_cpus, plan_pileup
from quadrat.pileup.sql import pileup_sql
from quadrat.reference import chrom_lengths
from quadrat.tables import TrackWriter, strand_slices, track_paths

#: Where both entry points report their progress. A logger rather than
#: `print`, because stdout cannot be filtered, redirected per caller or
#: captured by the logging configuration a training script already
#: has -- and because `verbose` as the only switch made the reporting
#: policy this package's rather than the caller's. The records are
#: emitted whatever `verbose` says; what it controls is whether this
#: package wires an output of its own.
log = logging.getLogger("quadrat.pileup")


@contextlib.contextmanager
def reporting(verbose):
    """Route this package's progress records to stdout, if asked.

    Here rather than in a module of its own because
    `pileup_alignments` is the unit every backend dispatches to, so
    `quadrat.pileup.run` already imports from this module and nothing
    else needs it.

    .. rubric:: Implementation notes

    The handler goes on `log` and never on the root logger,
    which is the distinction that makes this safe to do inside a
    library: a caller's own configuration is untouched, and the level
    is restored on the way out. A caller who has configured logging
    *and* passes ``verbose`` gets the records twice, which is the
    accepted cost of ``verbose`` being the switch for someone who has
    configured nothing.

    stdout rather than stderr, because that is where ``print`` put
    these and progress is not a diagnostic.

    This reaches only the process it runs in. Under
    ``pileup_many(backend="process")`` the records are emitted inside
    `loky` children, which inherit none of the parent's logging
    configuration -- so each child wires its own output here and the
    lines interleave, exactly as the prints did. Forwarding worker
    records to the parent is a separate piece of work and this does not
    claim it.
    """

    if not verbose:
        yield
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous = log.level
    log.addHandler(handler)
    log.setLevel(min(previous or logging.INFO, logging.INFO))
    try:
        yield
    finally:
        log.setLevel(previous)
        log.removeHandler(handler)
        handler.close()


# public
def pileup_alignments(
    alignments,
    sequences,
    name,
    *,
    pos_shift=0,
    neg_shift=0,
    unstranded=False,
    scale_factor=1.0,
    target_partitions=None,
    contig_groups=None,
    verbose=False,
):
    """Convert BAM/SAM files into ``{name}.track.parquet`` tracks.

    One position per record rather than coverage over the record's
    span, so which position -- a 5' end, a shifted one -- is the
    caller's choice through ``pos_shift`` and ``neg_shift``. The
    written table carries no assay.

    :param alignments: BAM/SAM paths, or one path on its own. All files
        are pooled into the same channel(s).
    :param sequences: FASTA path. Its contigs and lengths define which
        reads are kept.
    :param name: Output prefix. Unstranded data yields
        ``{name}.track.parquet``; stranded data yields
        ``{name}.+.track.parquet`` and ``{name}.-.track.parquet``.
    :param pos_shift: Shift applied to forward-strand 5' ends
        (bam2bw's ``-ps``).
    :param neg_shift: Shift applied to reverse-strand 5' ends
        (bam2bw's ``-ns``).
    :param unstranded: Pool both strands into a single table.
    :param scale_factor: Multiplier applied to every count.
    :param target_partitions: DataFusion's
        ``execution.target_partitions`` for this pileup, which bounds
        how many threads the scan and aggregate use. ``None`` uses
        every core, which is right for one pileup on an idle machine
        and wrong for several at once -- the setting is session-global
        and process-wide, so K concurrent pileups would otherwise ask
        for K times the cores. See `pileup_many`, which computes this
        from the worker count. Set for the duration of the call and
        restored afterwards.
    :param contig_groups: How many contig batches to run the genome in.
        One batch is the whole genome in a single aggregate. Splitting
        it bounds the peak, because only one batch's positions are
        ever live, and the output is byte-identical at any count.
        ``None`` solves for this machine *and* for the bytes this job
        reads: one batch on a small input, more when a single batch
        would scan past 8 GB however much memory the machine has, and
        more again when one batch would project past what memory is
        free, and never more than eight. Eight is the solver's ceiling
        and not the parameter's: a count you pass is honoured as
        given, because how much of the genome to hold at once is a
        decision this surface exposes rather than makes.
    :param verbose: Wire this package's progress records to stdout for
        the duration of the call. The records are emitted whatever this
        says; see `reporting`.
    :returns: The written track directory paths, in channel order. Each
        is a ``.track.parquet`` *directory* of per-contig files, not a
        Parquet file -- see the track format in the reference for what
        a reader may rely on.
    :raises RuntimeError: If a pileup is already running in this
        process. One call at a time per process is a precondition
        rather than a preference: the polars-bio session is
        process-global and every pileup registers its alignments under
        the same table names, so two that overlap read each other's
        reads and both tables still pass `require_complete`. Run them
        one after another, or use `pileup_many`'s ``process`` backend,
        which gives each pileup a process of its own. Calling it from
        two threads is therefore an error rather than a way to go
        faster -- the engine is already internally multithreaded, which
        is what ``target_partitions`` bounds.

    .. rubric:: Implementation notes

    The batch count's effect was measured on a 3.9 GB ENCODE
    alignment, before the aggregate fix: the peak falls from 12.65 GB
    at one batch to 5.95 GB at four and 3.06 GB at eight, with
    byte-identical output. Two and four are also *faster* than one;
    past eight the peak stops falling while the per-group overhead
    keeps accruing, which is where the ceiling comes from. Splitting
    is nearly free because a contig predicate reaches the BAM index as
    a region query, so a batch costs only the records it selects rather
    than a scan of the file -- a SAM input has no index and no header
    to read, so it is never split.

    Scaling inside a single pileup is weak, 1.9x from 14 cores on
    ENCSR483RKN, which is why dividing the machine between concurrent
    pileups costs little and is what `pileup_many` does.

    The two thresholds above are stated as numbers rather than as the
    constants `quadrat.pileup.plan` holds them in. Publishing a
    constant so that a docstring may name it gives a caller nothing the
    number does not.
    """

    # One alignment given as a bare path rather than as a list of one,
    # which is the mistake the plural name invites, and the trivial
    # case a caller should not have to wrap. Before the refusal below,
    # so an empty string is still no alignments.
    #
    # `fspath_or_none` is the package's one answer to "is this a path",
    # so the probe here and the conversion below cannot disagree about
    # which shapes count -- they were two spellings of the same set,
    # and `bytes` was in one of them and honoured by neither.
    if fspath_or_none(alignments) is not None:
        alignments = [alignments]

    if not alignments:
        raise ValueError(
            "pileup_alignments was given no alignments. A pileup over "
            "none of them reaches none of the guards -- no header to "
            "check against the reference, no contig batch to aggregate "
            "-- and still writes a manifest listing no files, which "
            "require_complete passes because every file it lists does "
            "exist. Training reads the result as a genome with no "
            "signal."
        )

    # The session this registers into is process-global and the
    # table names are fixed, so a second pileup overlapping this one
    # would read its reads. Held for the whole call rather than for
    # the registration, because the window is the whole call.
    #
    # The stack opens above the registration rather than beside the
    # writers. Everything this call puts into the shared session --
    # the alignment tables, the chrom
    # sizes frame, the one engine option -- is undone on the way out,
    # and a pileup that refuses partway through (an unknown contig, a
    # missing file) refuses *after* the registration, so a stack that
    # opened below it would leave exactly the failures that leak.
    # Normalized at the boundary, as every sibling that takes a path
    # does -- `chrom_lengths`, `register_alignments`, `bam_contigs` and
    # the three readers in `quadrat.tables`, all of them now through
    # the one normalizer, each with a test pinning it. Both failures
    # were a traceback from the middle of the call rather than a
    # refusal: `", ".join(alignments)` raised from inside a logging
    # call, and `name + TRACK_SUFFIX` from the path builder, both
    # *after* the alignment tables had been registered into the
    # process-global session.
    #
    # An element that is not a path is left as it was, so
    # `register_alignments` refuses it in its own words naming the
    # suffix rule and quoting what the caller passed. `name` has no
    # such owner below -- `track_paths` is where it lands, and the
    # refusal there would come after the registration -- so it is
    # refused here.
    normalized = []
    for alignment in alignments:
        # Explicitly rather than `fspath_or_none(a) or a`: an empty
        # string normalizes to a falsy value, and a fallback that reads
        # as "or the original" would be taking the wrong branch for it.
        converted = fspath_or_none(alignment)
        normalized.append(alignment if converted is None else converted)
    alignments = normalized

    # `sequences` too, and it is the one that fails latest. It is not
    # only handed to `chrom_lengths`, which normalizes for itself: it
    # is recorded in the manifest's `parameters`, through
    # `os.path.abspath`, which answers `bytes` for a `bytes` path. So a
    # `bytes` reference ran the whole pileup and then failed on
    # ``Object of type bytes is not JSON serializable`` while the
    # writer was committing -- past every guard, with the tables
    # already written. A non-path is left alone, because
    # `chrom_lengths` refuses it with a message about sequence
    # dictionaries that this one cannot improve on.
    converted = fspath_or_none(sequences)
    if converted is not None:
        sequences = converted
    path = fspath_or_none(name)
    if path is None:
        raise TypeError(
            "pileup_alignments writes tables under name: it must be a "
            "str, bytes or os.PathLike, got {}.".format(type(name).__name__)
        )
    name = path

    with exclusive_pileup(), reporting(verbose), contextlib.ExitStack() as stack:
        polars = import_on_first_use("polars")

        pb = engine()
        lengths = chrom_lengths(sequences)
        tables = register_alignments(alignments)
        stack.callback(deregister_alignments, tables)

        sizes = polars.DataFrame(
            {
                "chrom": list(lengths.keys()),
                "length": list(lengths.values()),
            },
            schema={"chrom": polars.Utf8, "length": polars.Int64},
        )
        stack.enter_context(registered_frames({SIZES_TABLE: sizes}))

        if target_partitions is None:
            # The cores this process may use, not the machine's, so a
            # pileup run on its own asks for the same parallelism one run
            # in a fan-out would.
            target_partitions = n_cpus()

        # Converted before the guard rather than inside it. A count
        # that is not a number is the caller's mistake and belongs in
        # their traceback; the guard below is for the engine refusing a
        # setting it does understand, which is a different failure with
        # a different owner.
        partitions = str(max(1, int(target_partitions)))
        try:
            # Restored on the way out, under the stack above. The
            # option is process-global and this is a shared session --
            # `quadrat._engine` makes that the module's central hazard
            # and enforces the table-name half of it -- so a pileup
            # that left it set would hand the next caller in this
            # process whatever parallelism this one chose, which at
            # `target_partitions=None` is the whole machine.
            previous = pb.get_option("datafusion.execution.target_partitions")
            pb.set_option("datafusion.execution.target_partitions", partitions)
            # Registered after the write rather than before it, so a
            # rejected setting leaves nothing to put back -- the
            # option never moved, and a restore that reached the same
            # refused key would raise during the unwind and mask
            # whatever the body was failing on.
            stack.callback(
                pb.set_option,
                "datafusion.execution.target_partitions",
                str(previous),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            # `BaseException`, not `Exception`: polars-bio surfaces a
            # Rust panic as `PanicException`, which derives from
            # `BaseException` directly, and a renamed config key is
            # exactly how this fails. An `except Exception` here reads
            # as a guard and is none: it catches every failure but the
            # one it exists for.
            warnings.warn(
                "the engine rejected target_partitions={} ({}: {}). This "
                "pileup will run at whatever parallelism the session "
                "already had. The table is unaffected -- the setting "
                "bounds how many threads the scan and aggregate use, not "
                "what they count -- so this costs time rather than "
                "correctness. Check that the installed polars-bio still "
                "accepts datafusion.execution.target_partitions.".format(
                    partitions, type(error).__name__, error
                ),
                EngineOptionWarning,
                stacklevel=2,
            )

        log.info("GIQL engine: piling up from %s", ", ".join(alignments))

        paths = track_paths(name, unstranded)
        parameters = {
            "alignments": [os.path.abspath(a) for a in alignments],
            "sequences": os.path.abspath(sequences),
            "pos_shift": int(pos_shift),
            "neg_shift": int(neg_shift),
            "unstranded": bool(unstranded),
            "scale_factor": float(scale_factor),
        }

        available = collections.OrderedDict(
            (table, bam_contigs(alignment))
            for table, alignment in zip(tables, alignments)
        )
        require_reference_contigs(lengths, available, alignments, sequences)
        if contig_groups is None:
            # Solved for one pileup having the machine to itself, which is
            # what a direct call is. `pileup_many` overrides this with a
            # count sized for the concurrency it is about to run.
            #
            # These are this job's exact bytes rather than a mean over a
            # set, so the scan cap applies to it precisely. This is also
            # the only planner call the serial backend makes, which is why
            # the cap has to live in `plan_pileup` and not only in the
            # per-job pass `pileup_many` runs.
            try:
                total = sum(os.path.getsize(a) for a in alignments)
            except OSError:
                total = None
            contig_groups = plan_pileup(1, total).contig_groups
        batches = contig_batches(lengths, available, contig_groups)
        warn_unindexed_split(alignments, available, batches)

        # A stack of its own for the writers, nested inside the one
        # above. They have to commit before the session is unwound,
        # and more importantly a writer must commit or abort before
        # this function returns rather than at the end of the whole
        # call, so the `with` boundary is what decides it.
        with contextlib.ExitStack() as writing:
            writers = [
                writing.enter_context(
                    TrackWriter(
                        path, parameters=dict(parameters, channel=os.path.basename(path))
                    )
                )
                for path in paths
            ]

            for batch in batches:
                _pileup_group(
                    pb,
                    writers,
                    tables,
                    batch,
                    unstranded,
                    pos_shift,
                    neg_shift,
                    scale_factor,
                    verbose,
                )

            for path, writer in zip(paths, writers):
                log.info("GIQL engine: wrote %s positions to %s", writer.rows, path)

        return paths


def _pileup_group(
    pb, writers, tables, batch, unstranded, pos_shift, neg_shift, scale_factor, verbose
):
    """Aggregate one contig group and stage it.

    A function rather than the body of a loop, and deliberately so. The
    slices `strand_slices` hands back are views over ``frame``, so any
    one of them still bound -- including a loop variable left over after
    the loop ends -- pins the entire group and gives back the memory
    that grouping exists to save. Returning drops all of them at once,
    where a scattering of ``del`` statements would not.
    """

    polars = import_on_first_use("polars")

    sql = pileup_sql(
        tables,
        pos_shift=pos_shift,
        neg_shift=neg_shift,
        unstranded=unstranded,
        scale_factor=scale_factor,
        contigs=batch,
    )
    if sql is None:
        # No alignment in this job has anything in this group.
        return

    frame = pb.sql(sql).collect()

    # The SQL casts pos to BIGINT and the count to DOUBLE, so this is a
    # no-op on every path that goes through pileup_sql. It stays as a
    # guard on the dtype contract the writer and the cache both rely on,
    # and fires only if a provider ever hands back something narrower --
    # an unconditional cast would copy the largest frame in the process
    # to no purpose.
    casts = [
        polars.col(column).cast(dtype)
        for column, dtype in (("pos", polars.Int64), ("value", polars.Float64))
        if frame.schema[column] != dtype
    ]
    if casts:
        frame = frame.with_columns(casts)

    for writer, part in zip(writers, strand_slices(frame, unstranded)):
        writer.write(part)
