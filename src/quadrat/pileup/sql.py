"""The track grouping, as one query per contig batch.

Which records count and what each contributes is an
`quadrat.pileup.aggregate.Aggregate`'s to say; this module turns one
into SQL. What accumulates is an exact integer per position: a count,
scaled once afterwards, or a sum of signed boundary events, left
unscaled for the reader to scale once it has summed them. Either way
two backends are byte-comparable rather than approximately equal.

Where the query joins the contig sizes, and which side of each join
DataFusion collects, decides whether it survives at scale;
`pileup_sql` states both.
"""

from quadrat._engine import SIZES_TABLE, sql_string
from quadrat.alignments import BAM_START_OFFSET
from quadrat.tables import COVERAGE_DELTAS


def pileup_sql(bam_tables, aggregate, contigs=None):
    """Build the track grouping over one or more registered BAM tables.

    ``aggregate`` is an `Aggregate`, which states which records
    qualify, the positions each contributes, what accumulates, and
    how a position outside its contig is treated. Several files are
    concatenated before accumulating. Grouped by ``(chrom, pos)``, and
    by ``strand`` too unless the aggregate is unstranded.

    ``contigs`` restricts each branch to the contigs that branch's table
    actually has, as ``{table: [chrom, ...]}``. The predicate goes on the
    scan rather than on the join because that is what reaches the BAM
    index: a contig equality is pushed down to an indexed region query,
    so a group costs only the records it selects. A table with nothing
    in the group is dropped from the ``UNION ALL`` -- asking its provider
    for a region absent from its header raises rather than returning
    nothing -- and when no table has anything, the whole query is
    ``None``.

    Building the query needs nothing the ``pileup`` extra installs;
    running it does.

    .. rubric:: Implementation notes

    Under counts the two halves of the ``[0, len)`` bound do not run in
    the same place. ``pos >= 0`` sits in the ``totals`` CTE's
    ``WHERE``, so it runs *at* the grouping: it needs no ``length``,
    and there is nothing to gain by carrying a negative position
    through the grouping. Only ``pos < length`` runs above the join,
    and that is the half the next two paragraphs apply to.

    So: the upper bound is applied after the grouping, and the sizes
    table is named first in the ``FROM`` clause. Both halves of that
    matter. ``length`` is a function of ``chrom`` alone and
    ``chrom`` is a grouping key, so the predicate is constant across
    every row of a group: the join can only delete whole ``(chrom,
    pos)`` groups, never alter a surviving group's count. Joining the
    grouped result is therefore the same answer over far fewer rows --
    16,898,696 groups rather than 291,063,148 reads on the alignment
    this was measured against, filtering to the 16,882,300 the pileup
    writes either way. Joining the reads instead, with them named
    first, does not merely cost more, it dies outright: DataFusion
    plans this as a
    ``CollectLeft`` hash join and
    materialises the left input whole, so one Arrow ``chrom`` array
    held every read's contig name and crossed the 32-bit offset
    ceiling just past 2^31. Naming the sizes table first keeps the
    collected side at one row per contig whatever sits below it.

    The join stays an inner one because it is also the contig filter:
    a record on a contig the reference does not name is dropped, and
    after the rewrite its whole group is dropped instead. The sizes
    table has one row per contig, so it cannot duplicate a grouped row
    either. One cost, paid only when ``contigs`` is ``None``: the
    grouping now builds entries for off-reference contigs before the
    join deletes them. ``_pileup_group`` always passes a batch and
    `contig_batches` builds its groups from the reference's own
    lengths, so nothing off-reference reaches the SQL on that path.

    Under coverage deltas the bound is a clamp rather than a filter,
    and a clamp needs ``length`` on every record, so the sizes table is
    joined below the grouping as well. Each of a record's two events
    goes at ``LEAST(event, length)``: a record ending past the contig
    closes at its length, and one starting at or past the length puts
    both events there, where they cancel. Filtering the events instead,
    as counts are filtered, loses a contig's zero sum both ways: a
    ``-1`` past the length dropped leaves its record's coverage running
    on to the contig's end, and a ``+1`` at the length netted into the
    row there cancels another record's closing event. Only an alignment whose header declares a
    contig longer than the reference holds such a record. The join
    below collects the two-row ``VALUES`` crossed with the sizes table,
    named first for the reason above, so the collected side is two rows
    per contig and the reads stream through it; ``tests/test_sql.py``
    pins that plan. ``pos >= 0`` is left out there: no event is
    negative, a span taking no shift, and over the clamped position it
    would become a filter on that join. The join above stays, bounding
    at ``pos <= length``, which the clamp already guarantees, so every
    rule's query keeps one shape.

    The ``ORDER BY`` is not cosmetic and is not for the reader, and it
    sits on the outermost query, above the join. It puts each strand,
    and each contig within a strand, in one contiguous run, which is
    what lets `pileup_alignments` and `TrackWriter`
    take every output file as a zero-copy ``slice`` of the result. The
    alternative -- ``filter`` per strand and ``partition_by`` per contig --
    allocates a second and third copy of the whole table, and at ENCODE
    scale those copies are the memory bound rather than an overhead.
    """

    flags = "CAST(b.flags AS BIGINT)"
    forward = "({} & 16) = 0".format(flags)
    tlen = "CAST(b.template_length AS BIGINT)"
    strand = "CASE WHEN {} THEN '+' ELSE '-' END AS strand".format(forward)
    start = "(CAST(b.start AS BIGINT) + {} + ({}))".format(
        BAM_START_OFFSET, aggregate.start_shift
    )
    end = '(CAST(b."end" AS BIGINT) + {} + ({}))'.format(
        BAM_START_OFFSET, aggregate.end_shift
    )

    # The rule's position, per `Aggregate`. A coverage span is not one
    # row per covered base: that multiplies the rows by the read
    # length, and the multiplied `chrom` column is one Arrow string
    # array whose offsets are 32-bit, so it fails once records times
    # read length times contig-name bytes passes 2**31.
    events = ""
    pos = {
        "five_prime": "CASE WHEN {f} THEN {s} ELSE {e} - 1 END",
        "three_prime": "CASE WHEN {f} THEN {e} - 1 ELSE {s} END",
        "midpoint": "CASE WHEN {f} THEN {s} ELSE {e} - {t} END + ({t} - 1) / 2",
        "span": "LEAST(CASE WHEN e.delta = 1 THEN {s} ELSE {e} END, z.length)",
    }[aggregate.position].format(f=forward, s=start, e=end, t=tlen) + " AS pos"
    deltas = aggregate.signal == COVERAGE_DELTAS
    if deltas:
        # The sizes joined per record, for the clamp, with the two small
        # sides named first so they are what the joins collect; see the
        # notes.
        events = "(VALUES (1), (-1)) AS e(delta) CROSS JOIN {} AS z JOIN ".format(
            SIZES_TABLE
        )
        pos += ", e.delta"

    # The record filter, per `Aggregate`. An unmapped record is skipped
    # whatever the mask says: a placed one carries its mate's
    # coordinates, not an alignment of its own.
    filters = ["({} & 4) = 0".format(flags)]
    if aggregate.exclude_flags:
        filters.append("({} & {}) = 0".format(flags, aggregate.exclude_flags))
    if aggregate.require_flags:
        filters.append("({} & {r}) = {r}".format(flags, r=aggregate.require_flags))
    if aggregate.min_mapq:
        filters.append(
            "CAST(b.mapping_quality AS BIGINT) >= {}".format(aggregate.min_mapq)
        )
    if aggregate.min_template_length:
        filters.append("abs({}) >= {}".format(tlen, aggregate.min_template_length))
    if aggregate.max_template_length is not None:
        filters.append("abs({}) <= {}".format(tlen, aggregate.max_template_length))
    if aggregate.position == "midpoint":
        filters.append("{} > 0".format(tlen))

    branches = []
    for table in bam_tables:
        where = " AND ".join(filters)
        if contigs is not None:
            names = list(contigs.get(table, ()))
            if not names:
                # This table has nothing in this group. Emitting a
                # branch anyway would ask its provider for a region it
                # has never heard of, which is an error rather than an
                # empty result.
                continue
            where += " AND b.chrom IN ({})".format(
                ", ".join(sql_string(name) for name in names)
            )
        source = "{} AS b".format(table)
        if deltas:
            source += " ON z.chrom = b.chrom"
        branches.append(
            "SELECT b.chrom, {pos}, {strand} FROM {events}{source} WHERE {where}".format(
                pos=pos, strand=strand, events=events, source=source, where=where
            )
        )

    if not branches:
        return None

    unstranded = aggregate.unstranded
    keys = ["chrom", "pos"] if unstranded else ["chrom", "pos", "strand"]
    group = ", ".join("c." + key for key in keys)
    select = ", ".join("g." + key for key in keys)
    order = ", ".join(
        "g." + key for key in (keys if unstranded else ["strand", "chrom", "pos"])
    )
    if deltas:
        # A start and an end on one position cancel, and a zero is a
        # row that changes no prefix sum. Unscaled, for the reason
        # `quadrat.stores.coverage_levels` gives for scaling last. The
        # bounds are the clamp's; see the notes.
        total, having = "CAST(SUM(c.delta) AS DOUBLE)", " HAVING SUM(c.delta) <> 0"
        lower, bound = "", "g.pos <= s.length"
    else:
        total, having = (
            "CAST(COUNT(*) AS DOUBLE) * {}".format(aggregate.scale_factor),
            "",
        )
        lower, bound = " WHERE c.pos >= 0", "g.pos < s.length"
    return (
        "WITH points AS ({branches}), "
        "totals AS (SELECT {group}, {total} "
        "AS value FROM points AS c{lower} GROUP BY {group}{having}) "
        "SELECT {select}, g.value "
        "FROM {sizes} AS s JOIN totals AS g ON s.chrom = g.chrom "
        "WHERE {bound} "
        "ORDER BY {order}"
    ).format(
        branches=" UNION ALL ".join(branches),
        bound=bound,
        group=group,
        total=total,
        lower=lower,
        having=having,
        select=select,
        order=order,
        sizes=SIZES_TABLE,
    )
