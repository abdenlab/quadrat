"""Alignments to cut-site tables.

One call per channel. ``pileup_alignments`` pools every alignment it is
given into one table, exactly as a single ``bam2bw`` invocation would,
and writes it as a directory of per-contig Parquet files.

The genome is read in contig batches rather than at once, and the batch
count is solved rather than configured -- see
:mod:`quadrat.pileup.plan`. Only one batch's cut sites are ever live,
so the peak follows the batch rather than the alignment, and because a
contig predicate reaches the BAM index the batch costs only the records
it selects.
"""

import collections
import contextlib
import os

from quadrat._engine import _SIZES_TABLE, _pb, _register_frame
from quadrat.alignments import (
    _contig_batches,
    _require_reference_contigs,
    _warn_unindexed_split,
    bam_contigs,
    register_alignments,
)
from quadrat.pileup.plan import (
    MAX_CONTIG_GROUPS,
    default_partitions,
    plan_pileup,
)
from quadrat.pileup.run import BACKENDS, pileup_many
from quadrat.pileup.sql import pileup_sql
from quadrat.reference import chrom_lengths
from quadrat.tables import _cut_site_paths, _CutSiteWriter, _strand_slices

__all__ = [
    "BACKENDS",
    "MAX_CONTIG_GROUPS",
    "default_partitions",
    "pileup_alignments",
    "pileup_many",
    "plan_pileup",
]


# public
def pileup_alignments(
    alignments,
    sequences,
    name,
    pos_shift=0,
    neg_shift=0,
    unstranded=False,
    fragments=False,
    scale_factor=1.0,
    read_depth=False,
    target_partitions=None,
    contig_groups=None,
    verbose=False,
):
    """Convert BAM/SAM files into ``{name}.cuts.parquet`` cut-site tables.

    Parameters
    ----------
    alignments: list of str
        BAM/SAM paths. All files are pooled into the same channel(s).

    sequences: str
        FASTA path. Its contigs and lengths define which reads are kept.

    name: str
        Output prefix. Unstranded data yields ``{name}.cuts.parquet``;
        stranded data yields ``{name}.+.cuts.parquet`` and
        ``{name}.-.cuts.parquet``.

    pos_shift, neg_shift: int
        Shifts applied to forward-strand 5' ends and to reverse-strand 5'
        ends, respectively (bam2bw's ``-ps`` / ``-ns``).

    unstranded: bool
        Pool both strands into a single table.

    fragments, read_depth: bool
        Not supported (the ATAC-seq BAM recipe never sets them).

    scale_factor: float
        Multiplier applied to every count.

    target_partitions: int or None
        DataFusion's ``execution.target_partitions`` for this pileup, which
        bounds how many threads the scan and aggregate use. ``None`` uses
        every core, which is right for one pileup on an idle machine and
        wrong for several at once -- the setting is session-global and
        process-wide, so K concurrent pileups would otherwise ask for K
        times the cores. Scaling inside a single pileup is weak anyway
        (1.9x from 14 cores on ENCSR483RKN), so dividing the machine
        between concurrent pileups costs little and avoids the
        oversubscription. See :func:`quadrat.pileup.pileup_many`, which
        computes this from the worker count.

    contig_groups: int or None
        How many contig batches to run the genome in. One batch is the
        whole genome in a single aggregate, which is what this always
        did. Splitting it bounds the peak, because only one batch's cut
        sites are ever live: on a 3.9 GB ENCODE alignment, measured
        before the aggregate fix, the peak falls from 12.65 GB at one
        batch to 5.95 GB at four and 3.06 GB at eight, with
        byte-identical output. Two and four are also
        *faster* than one; past :data:`MAX_CONTIG_GROUPS` the peak stops
        falling and only the overhead keeps growing. ``None`` solves for
        this machine *and* for the bytes this job reads: one batch on a
        small input, more when a single batch would scan past
        :data:`quadrat.pileup.MAX_GROUP_INPUT_BYTES`, however much
        memory the machine has.

        This works because a contig predicate reaches the BAM index as a
        region query, so a batch costs only the records it selects
        rather than a scan of the file. A SAM input has no index and no
        header to read, so it is never split.

    Returns
    -------
    paths: list of str
        The written Parquet paths, in channel order.
    """

    if fragments:
        raise NotImplementedError(
            "fragments=True is not supported by the "
            "GIQL engine; use the bam2bw engine for fragment files."
        )
    if read_depth:
        raise NotImplementedError("read_depth=True is not supported by the GIQL engine.")

    import polars

    pb = _pb()
    lengths = chrom_lengths(sequences)
    tables = register_alignments(alignments)

    _register_frame(
        pb,
        _SIZES_TABLE,
        polars.DataFrame(
            {
                "chrom": list(lengths.keys()),
                "length": list(lengths.values()),
            },
            schema={"chrom": polars.Utf8, "length": polars.Int64},
        ),
    )

    if target_partitions is None:
        # The cores this process may use, not the machine's -- see
        # `quadrat.pileup.plan._n_cpus`, which this mirrors so a
        # pileup run on its own asks for the same parallelism one
        # run in a fan-out would.
        from quadrat.pileup.plan import _n_cpus

        target_partitions = _n_cpus()
    try:
        pb.set_option(
            "datafusion.execution.target_partitions", str(max(1, int(target_partitions)))
        )
    except Exception:
        pass

    if verbose:
        print("GIQL engine: piling up cut sites from {}".format(", ".join(alignments)))

    paths = _cut_site_paths(name, unstranded)
    parameters = {
        "alignments": [os.path.abspath(a) for a in alignments],
        "sequences": os.path.abspath(sequences),
        "pos_shift": int(pos_shift),
        "neg_shift": int(neg_shift),
        "unstranded": bool(unstranded),
        "scale_factor": float(scale_factor),
    }

    available = collections.OrderedDict(
        (table, bam_contigs(alignment)) for table, alignment in zip(tables, alignments)
    )
    _require_reference_contigs(lengths, available, alignments, sequences)
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
        contig_groups = plan_pileup(1, total)[1]
    batches = _contig_batches(lengths, available, contig_groups)
    _warn_unindexed_split(alignments, available, batches)

    with contextlib.ExitStack() as stack:
        writers = [
            stack.enter_context(
                _CutSiteWriter(
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

        if verbose:
            for path, writer in zip(paths, writers):
                print("GIQL engine: wrote {} cut sites to {}".format(writer.rows, path))

    return paths


def _pileup_group(
    pb, writers, tables, batch, unstranded, pos_shift, neg_shift, scale_factor, verbose
):
    """Aggregate one contig group and stage it.

    A function rather than the body of a loop, and deliberately so. The
    slices `_strand_slices` hands back are views over ``frame``, so any
    one of them still bound -- including a loop variable left over after
    the loop ends -- pins the entire group and gives back the memory
    that grouping exists to save. Returning drops all of them at once,
    where a scattering of ``del`` statements would not.
    """

    import polars

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

    for writer, part in zip(writers, _strand_slices(frame, unstranded)):
        writer.write(part)
