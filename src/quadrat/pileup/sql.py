"""The cut-site aggregate, as one query per contig batch.

Every mapped record contributes one cut site -- ``start + pos_shift``
forward, ``end - 1 + neg_shift`` reverse -- and the aggregate counts
them per position. The count is an exact integer scaled once
afterwards, which is what makes two backends byte-comparable rather
than approximately equal.

The join against contig sizes sits *above* the aggregate, not beside
the reads. Below it, the probe side carries one row per record, and at
291 M records the ``chrom`` array crosses Arrow's 2**31 offset ceiling
and the query dies. Above it, the same filter runs against one row per
distinct position. It is equivalent because ``length`` is a function of
``chrom`` alone and ``chrom`` is a grouping key, so the predicate is
constant across every row of a group: the join can only delete whole
groups, never alter a surviving group's count.

The sizes table is also named first in ``FROM``, because DataFusion
materialises the left input of a ``CollectLeft`` join whole and the
left input is whichever relation ``FROM`` names first.
"""

from quadrat._engine import _SIZES_TABLE, _sql_string
from quadrat.alignments import _BAM_START_OFFSET


def pileup_sql(
    bam_tables,
    pos_shift=0,
    neg_shift=0,
    unstranded=False,
    scale_factor=1.0,
    sizes_table=_SIZES_TABLE,
    contigs=None,
):
    """Build the cut-site aggregate over one or more registered BAM tables.

    Mirrors bam2bw: forward reads (``flags & 16 = 0``) cut at ``start +
    pos_shift``; reverse reads cut at ``end - 1 + neg_shift``; unmapped records
    (``flags & 4``) are skipped; records on contigs absent from the sizes
    table are dropped; several files are concatenated before counting. The
    one deliberate difference is that cut sites shifted outside ``[0, len)``
    are dropped here, whereas bam2bw hands them to pyBigWig unclipped; no
    training window can read those positions, so tensors are unaffected.

    ``contigs`` restricts each branch to the contigs that branch's table
    actually has, as ``{table: [chrom, ...]}``. The predicate goes on the
    scan rather than on the join because that is what reaches the BAM
    index: a contig equality is pushed down to an indexed region query,
    so a group costs only the records it selects. A table with nothing
    in the group is dropped from the ``UNION ALL`` -- asking its provider
    for a region absent from its header raises rather than returning
    nothing -- and when no table has anything, the whole query is
    ``None``.

    The bound on ``[0, len)`` is applied *after* the aggregate, and the
    sizes table is named first in the ``FROM`` clause. Both halves of
    that matter. ``length`` is a function of ``chrom`` alone and
    ``chrom`` is a grouping key, so the predicate is constant across
    every row of a group: the join can only delete whole ``(chrom,
    pos)`` groups, never alter a surviving group's count. Joining the
    grouped result is therefore the same answer over far fewer rows --
    16,898,696 groups rather than 291,063,148 reads on the alignment
    this was measured against, filtering to the 16,882,300 the pileup
    writes either way. Joining the reads used to die outright:
    DataFusion plans this as a ``CollectLeft`` hash join and
    materialises the left input whole, so one Arrow ``chrom`` array
    held every read's contig name and crossed the 32-bit offset
    ceiling just past 2^31. Naming the sizes table first keeps the
    collected side at one row per contig whatever sits below it.

    The join stays an inner one because it is also the contig filter:
    a record on a contig the reference does not name is dropped, and
    after the rewrite its whole group is dropped instead. The sizes
    table has one row per contig, so it cannot duplicate a grouped row
    either. One cost, paid only when ``contigs`` is ``None``: the
    aggregate now builds entries for off-reference contigs before the
    join deletes them. ``_pileup_group`` always passes a batch and
    :func:`_contig_batches` builds its groups from the reference's own
    lengths, so nothing off-reference reaches the SQL on that path.

    The ``ORDER BY`` is not cosmetic and is not for the reader, and it
    sits on the outermost query, above the join. It puts each strand,
    and each contig within a strand, in one contiguous run, which is
    what lets :func:`pileup_alignments` and :func:`_write_cut_sites`
    take every output file as a zero-copy ``slice`` of the result. The
    alternative -- ``filter`` per strand and ``partition_by`` per contig --
    allocates a second and third copy of the whole table, and at ENCODE
    scale those copies are the memory bound rather than an overhead.
    """

    strand = (
        "CASE WHEN (CAST(b.flags AS BIGINT) & 16) = 0 THEN '+' ELSE '-' END AS strand"
    )
    pos = (
        "CASE WHEN (CAST(b.flags AS BIGINT) & 16) = 0 "
        "THEN CAST(b.start AS BIGINT) + {offset} + ({pos_shift}) "
        'ELSE CAST(b."end" AS BIGINT) + {offset} + ({neg_shift}) - 1 END AS pos'
    ).format(
        offset=_BAM_START_OFFSET, pos_shift=int(pos_shift), neg_shift=int(neg_shift)
    )

    branches = []
    for table in bam_tables:
        where = "(CAST(b.flags AS BIGINT) & 4) = 0"
        if contigs is not None:
            names = list(contigs.get(table, ()))
            if not names:
                # This table has nothing in this group. Emitting a
                # branch anyway would ask its provider for a region it
                # has never heard of, which is an error rather than an
                # empty result.
                continue
            where += " AND b.chrom IN ({})".format(
                ", ".join(_sql_string(name) for name in names)
            )
        branches.append(
            "SELECT b.chrom, {pos}, {strand} FROM {table} AS b WHERE {where}".format(
                pos=pos, strand=strand, table=table, where=where
            )
        )

    if not branches:
        return None

    keys = ["chrom", "pos"] if unstranded else ["chrom", "pos", "strand"]
    group = ", ".join("c." + key for key in keys)
    select = ", ".join("g." + key for key in keys)
    order = ", ".join(
        "g." + key for key in (keys if unstranded else ["strand", "chrom", "pos"])
    )
    return (
        "WITH cuts AS ({branches}), "
        "counts AS (SELECT {group}, CAST(COUNT(*) AS DOUBLE) * {scale} "
        "AS value FROM cuts AS c WHERE c.pos >= 0 GROUP BY {group}) "
        "SELECT {select}, g.value "
        "FROM {sizes} AS s JOIN counts AS g ON s.chrom = g.chrom "
        "WHERE g.pos < s.length "
        "ORDER BY {order}"
    ).format(
        branches=" UNION ALL ".join(branches),
        group=group,
        select=select,
        order=order,
        scale=float(scale_factor),
        sizes=sizes_table,
    )
