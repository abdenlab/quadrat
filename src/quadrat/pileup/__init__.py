"""The pileup: alignments to tracks, and the fan-out over them.

The implementation lives in sibling modules and is re-exported here, so
``quadrat.pileup`` is the surface a caller reads and nothing has to be
imported by its module path. `track` holds the
single-channel pileup, `run` the backends that run
many of them at once, `plan` the memory model both
size themselves from, and `sql` the aggregate.

Keeping the implementation out of this file is what lets the dependency
run one way: ``run`` imports ``track`` at module scope, which it could
not do while ``track`` was this module's body.
"""

# Two names a caller programs against -- `pileup_alignments` and
# `pileup_many`, the two that carry a `# public` marker -- and two a
# sibling module or a test may read. `plan_pileup` answers what to run
# and `BACKENDS` names the two bundled runners, and neither is
# supported: they are this subpackage's wildcard surface rather than
# the package's, which is what `docs/reference.rst` says under "Not
# documented here" and what the absence of a marker on them means.
# Three signals, one answer.
#
# `MAX_CONTIG_GROUPS` and `default_partitions` do not belong beside
# them: the first is meaningful only as the ceiling the solver stops
# at, which `pileup_alignments` states as a number in its own prose --
# an explicitly passed `contig_groups` is honoured above it -- and the
# second only as what `target_partitions=None` resolves to, which
# `pileup_many` documents. What publishing either would give a caller
# that the number in the prose does not is nothing.
__all__ = [
    "BACKENDS",
    "pileup_alignments",
    "pileup_many",
    "plan_pileup",
]

from quadrat.pileup.plan import plan_pileup
from quadrat.pileup.run import BACKENDS, pileup_many
from quadrat.pileup.track import pileup_alignments
