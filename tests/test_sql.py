"""Tests for the shape of the track aggregate's query plan.

Relocated from the ported ``tests/test_pileup_io.py``, assertions
unchanged.

.. rubric:: Why this one test is white-box

It is the only test in the suite that reaches a private symbol, and
that is deliberate rather than unconverted. §8 of the test guide
forbids referencing private names, and the honest position is that
there is no public route to what this pins: `pileup_sql` builds a query
string, nothing public hands back that string or the plan DataFusion
makes of it, and the sizes frame it joins against is registered through
`quadrat._engine`.

The behavioural consequence -- that a pileup over off-reference contigs
still counts correctly -- is already covered in ``tests/test_track.py``
by the bam2bw oracle. What that cannot distinguish is a plan that joins
*above* the aggregate from one that joins below, because the two differ
only in whether the query survives at 291,063,148 records, which no
fixture small enough for CI can reach. So the choice is between a
white-box test and no regression guard on a defect that killed the
query outright.

The guard is kept. If a public accessor for the plan ever exists, this
is the test to reseat onto it.
"""

import pytest

from quadrat import _engine, alignments, reference
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

    # Arrange
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

    # Act
    explained = pb.sql("EXPLAIN " + sql.pileup_sql(registered)).collect()
    physical = next(
        row["plan"]
        for row in explained.iter_rows(named=True)
        if row["plan_type"] == "physical_plan"
    )

    # Assert
    assert "JoinExec" in physical and "AggregateExec" in physical

    # Above, not below: the join probes the grouped rows rather than one
    # per read. Below is what overflowed Arrow's 32-bit string offsets at
    # 291,063,148 reads, because the join collects one side whole and
    # that side was the reads.
    assert plan_ancestor(physical, "JoinExec", "AggregateExec")
    assert not plan_ancestor(physical, "AggregateExec", "JoinExec")
