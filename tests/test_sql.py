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
the same kind of fact, and is pinned here for the same reason.
"""

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
    "aggregate",
    [Aggregate(), Aggregate(position="span")],
    ids=["five_prime", "span"],
)
def test_pileup_sql_should_never_collect_the_reads_side_of_a_join(
    corpus_factory, aggregate
):
    """Test which side of every join in the plan is collected.

    Given:
        The track pileup query under a 5' rule, which joins the sizes
        above the grouping, and under ``"span"``, which also joins
        them and the two boundary events below it, to each record.
    When:
        polars-bio explains it.
    Then:
        No join should collect the side the alignment is scanned on,
        and under ``"span"`` the join that meets each record should
        collect the two-row ``VALUES`` crossed with the sizes, whose
        rows are per contig rather than per record.
    """

    # Act
    physical = explain(corpus_factory, aggregate)

    # Assert
    sides = collected_sides(physical)
    assert sides and not any("BamExec" in side for side in sides)
    if aggregate.position == "span":
        assert any(
            "DataSourceExec" in side and "StreamingTableExec" in side for side in sides
        )
