"""Tests for the shape of the track aggregate's query plan.

.. rubric:: Why this test is white-box

It reaches a private module deliberately, because there is no public
route to what it pins: `pileup_sql` builds a query string, nothing
public hands back that string or the plan DataFusion makes of it, and
the sizes frame it joins against is registered under a name
`quadrat._engine` holds.

The behavioural consequence -- that a pileup over off-reference contigs
still counts correctly -- is covered in ``tests/test_track.py`` by the
bam2bw oracle. What that cannot distinguish is a plan that joins
*above* the aggregate from one that joins below, because the two differ
only in whether the query survives at 291,063,148 records, which no
fixture small enough for CI can reach. So the choice is between a
white-box test and no regression guard on a plan that fails outright
at scale, and the guard is kept. Which side of a join is collected is
the same kind of fact, and is pinned here for the same reason. So is
the memory shape of the spliced decomposition: where the CIGAR is
read, and that each skipping record becomes one narrow row per CIGAR
operation rather than a copy of itself or of its CIGAR per operation.
A plan that made the copies would write the same coverage, so at a
size CI can run only the plan tells the two apart.
"""

import re

import pytest

from quadrat import Aggregate, _engine, alignments, reference
from quadrat.pileup import sql
from tests import TWO_CONTIGS


def plan_ancestor(plan, parent, child):
    """Return whether every ``child`` node sits under a ``parent`` node.

    DataFusion prints a physical plan depth first, parents before
    children and two spaces of indent per level, so one node is an
    ancestor of another exactly when it appears earlier and is less
    indented.
    """

    def nodes(marker):
        return [
            (i, len(line) - len(line.lstrip(" ")))
            for i, line in enumerate(plan.splitlines())
            if marker in line
        ]

    parents, children = nodes(parent), nodes(child)
    if not parents or not children:
        return False
    return all(any(pi < ci and pd < cd for pi, pd in parents) for ci, cd in children)


def collected_sides(plan):
    """Return the subtree each join in a physical plan collects, as text.

    A DataFusion join collects its first child, the one printed first
    beneath it, and streams its second through it. A node's children
    are the lines indented two spaces deeper than it, up to the next
    line indented no deeper than it.
    """

    lines = plan.splitlines()
    depth = [len(line) - len(line.lstrip(" ")) for line in lines]
    sides = []
    for i, line in enumerate(lines):
        if "JoinExec" not in line:
            continue
        end = next((j for j in range(i + 1, len(lines)) if depth[j] <= depth[i]), None)
        children = [
            j for j in range(i + 1, end or len(lines)) if depth[j] == depth[i] + 2
        ]
        last = children[1] if len(children) > 1 else end or len(lines)
        sides.append("\n".join(lines[children[0] : last]))
    return sides


def subtrees(plan, marker):
    """Return the subtree under each node of a physical plan naming ``marker``.

    Each subtree is the node's descendants as lines, in printed order:
    every line after it up to the next one indented no deeper than it.
    """

    lines = plan.splitlines()
    depth = [len(line) - len(line.lstrip(" ")) for line in lines]
    found = []
    for i, line in enumerate(lines):
        if marker not in line:
            continue
        end = next((j for j in range(i + 1, len(lines)) if depth[j] <= depth[i]), None)
        found.append(lines[i + 1 : end])
    return found


def explain(corpus_factory, aggregate):
    """Return the physical plan polars-bio makes of a pileup query.

    The query is the one `pileup_sql` builds over a registered
    two-alignment corpus, with the sizes frame registered under the
    name the query joins.
    """

    pb = pytest.importorskip("polars_bio")
    polars = pytest.importorskip("polars")
    corpus = corpus_factory("sql", TWO_CONTIGS, n_bams=2, index_fasta=True)
    registered = alignments.register_alignments(corpus.bams)
    lengths = reference.chrom_lengths(corpus.fasta)
    pb.from_polars(
        _engine.SIZES_TABLE,
        polars.DataFrame(
            {"chrom": list(lengths.keys()), "length": list(lengths.values())},
            schema={"chrom": polars.Utf8, "length": polars.Int64},
        ),
    )
    explained = pb.sql("EXPLAIN " + sql.pileup_sql(registered, aggregate)).collect()
    return next(
        row["plan"]
        for row in explained.iter_rows(named=True)
        if row["plan_type"] == "physical_plan"
    )


def test_pileup_sql_should_bound_positions_above_the_aggregate(corpus_factory):
    """Test where the chromosome-sizes join sits in the plan.

    Given:
        The track pileup query over a registered alignment.
    When:
        polars-bio explains it.
    Then:
        It should take the grouped result as the join's input rather
        than one row per read.
    """

    # Act
    physical = explain(corpus_factory, Aggregate())

    # Assert
    assert "JoinExec" in physical and "AggregateExec" in physical

    # Above, not below: the join probes the grouped rows rather than one
    # per read. Below is what overflowed Arrow's 32-bit string offsets at
    # 291,063,148 reads, because the join collects one side whole and
    # that side was the reads.
    assert plan_ancestor(physical, "JoinExec", "AggregateExec")
    assert not plan_ancestor(physical, "AggregateExec", "JoinExec")


@pytest.mark.parametrize(
    "aggregate, small_sides, sizes_sides",
    [
        (Aggregate(), 0, 0),
        (Aggregate(position="span"), 2, 0),
        (Aggregate(position="span", spliced=True), 4, 2),
    ],
    ids=["five_prime", "span", "spliced"],
)
def test_pileup_sql_should_never_collect_the_reads_side_of_a_join(
    corpus_factory, aggregate, small_sides, sizes_sides
):
    """Test which side of every join in the plan is collected.

    Given:
        The track pileup query over two alignments under a 5' rule,
        which joins the sizes above the grouping; under ``"span"``,
        which also joins them and the two boundary events below it, to
        each record; and under a spliced ``"span"``, which joins those
        to each skipped run as well, and the sizes alone to the records
        it selects the skipping ones from.
    When:
        polars-bio explains it.
    Then:
        No join should collect the side the alignment is scanned on.
        The joins collecting the two-row ``VALUES`` crossed with the
        sizes, whose rows are per contig rather than per record, should
        number one per alignment under ``"span"`` and two when spliced,
        and the spliced branch's join to the records should collect the
        sizes table alone, one per alignment.
    """

    # Act
    physical = explain(corpus_factory, aggregate)

    # Assert
    sides = collected_sides(physical)
    assert sides and not any("BamExec" in side for side in sides)
    small = [
        side
        for side in sides
        if "DataSourceExec" in side and "StreamingTableExec" in side
    ]
    assert len(small) == small_sides
    # A join straight over a scan: the alignment below it with no other
    # join and no grouping between, so not the bound above the grouping.
    over_scan = [
        side.strip()
        for side, below in zip(sides, subtrees(physical, "JoinExec"))
        if "BamExec" in "\n".join(below)
        and not any("HashJoinExec" in line or "AggregateExec" in line for line in below)
    ]
    sizes_alone = [side for side in over_scan if "DataSourceExec" not in side]
    assert len(sizes_alone) == sizes_sides
    assert all(
        side.startswith("StreamingTableExec") and "\n" not in side
        for side in sizes_alone
    )


def test_pileup_sql_should_take_skipped_runs_from_a_sorted_window_over_numbered_records(
    corpus_factory,
):
    """Test the shape of the spliced branch's plan.

    Given:
        The track pileup query over two alignments under a spliced
        ``"span"``.
    When:
        polars-bio explains it.
    Then:
        Each alignment should have one window over the unnested CIGAR
        operations, fed by a sort, with neither the CIGAR nor its array
        of operations carried past the unnest; below it, the records
        numbered once grouped into one row per distinct record, and the
        spliced-record filter applied to the scan below that grouping.
    """

    # Act
    physical = explain(corpus_factory, Aggregate(position="span", spliced=True))

    # Assert
    windows = [
        below
        for below in subtrees(physical, "WindowAggExec")
        if any("UnnestExec" in line for line in below)
    ]
    assert len(windows) == 2
    for below in windows:
        assert "SortExec" in below[0]
        text = "\n".join(below)
        unnest = text.index("UnnestExec")
        assert "cigar" not in text[:unnest]
        assert not re.search(r"\bops@", text[:unnest])
        numbered = text.index("row_number", unnest)
        collapse = text.index("AggregateExec", numbered)
        spliced_only = text.index("LIKE %N%", collapse)
        assert spliced_only < text.index("BamExec", spliced_only)


def test_pileup_sql_should_test_for_a_skip_below_the_events_join_under_unspliced_span(
    corpus_factory,
):
    """Test where the unspliced span query reads the CIGAR.

    Given:
        The track pileup query over two alignments under ``"span"``
        without ``spliced``.
    When:
        polars-bio explains it.
    Then:
        The CIGAR should be read below each join with the two boundary
        events, on that join's scan side, and named on no other line of
        the plan, so neither projected through the join nor read
        anywhere above it.
    """

    # Act
    physical = explain(corpus_factory, Aggregate(position="span"))

    # Assert
    joins = [line for line in physical.splitlines() if "JoinExec" in line]
    events = [
        (line, below)
        for line, side, below in zip(
            joins, collected_sides(physical), subtrees(physical, "JoinExec")
        )
        if "DataSourceExec" in side and "StreamingTableExec" in side
    ]
    assert len(events) == 2
    for line, below in events:
        assert "cigar" in "\n".join(below)
        assert "projection=" in line and "cigar" not in line
    naming = [line for line in physical.splitlines() if "cigar" in line]
    below_events = [line for _, below in events for line in below if "cigar" in line]
    assert naming == below_events


@pytest.mark.parametrize("position", ["five_prime", "three_prime", "midpoint"])
def test_pileup_sql_should_not_read_the_cigar_under_a_counts_rule(position):
    """Test that a counts query is untouched by what a span does with splicing.

    Given:
        Each rule that names one position per record.
    When:
        Its query is built.
    Then:
        It should not mention the CIGAR at all.
    """

    # Act
    query = sql.pileup_sql(["quadrat_bam_0"], Aggregate(position=position))

    # Assert
    assert "cigar" not in query
