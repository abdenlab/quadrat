"""Tests for the ``quadrat.pileup`` subpackage's own surface.

What used to live here -- tests for the planner, the fan-out, the track
format and the alignment reader -- now sits beside the module each one
exercises, in ``test_plan.py``, ``test_run.py``, ``test_track.py``,
``test_tables.py``, ``test_alignments.py`` and ``test_reference.py``.
This file keeps only what belongs to ``quadrat/pileup/__init__.py``:
what the subpackage publishes, and that publishing it is wiring rather
than a shim.

The subpackage's ``__all__`` is pinned by nothing else.
``tests/test_quadrat.py`` collects every ``# public`` marker in the
package and compares the set against ``quadrat.__all__`` -- the
top-level list only -- so the two names this subpackage exports
without a marker are invisible to it. That is deliberate and the
reference page says so: ``BACKENDS`` and ``plan_pileup`` are this
subpackage's wildcard surface rather than the package's supported one.
What can be pinned here is that the list and what the subpackage
actually binds agree.
"""

import quadrat
from quadrat import pileup as pileup_pkg


def test_pileup_should_bind_only_the_names_it_declares_when_star_imported():
    """Test that the subpackage's export list is the truth about it.

    Given:
        The ``quadrat.pileup`` subpackage.
    When:
        It is star-imported into a fresh namespace.
    Then:
        It should bind exactly the names its ``__all__`` declares, so a
        list that drifts from what the module binds fails here.
    """

    # Arrange
    namespace = {}

    # Act
    exec("from quadrat.pileup import *", namespace)

    # Assert
    bound = {name for name in namespace if not name.startswith("__")}
    assert bound == set(pileup_pkg.__all__)


def test_pileup_alignments_should_be_the_object_the_package_exports():
    """Test that the re-export is wiring rather than a wrapper.

    Given:
        The package and its ``pileup`` subpackage, each exporting
        `pileup_alignments`.
    When:
        The two are compared by identity.
    Then:
        It should be the same function object both times.

    The subpackage docstring makes this an architectural claim: the
    implementation moved into ``track.py`` so that ``run`` could depend
    on it one way, and the ``__init__`` re-exports both. A regression to
    a lazily-imported shim or a wrapper would satisfy every other test
    in the suite while reintroducing the import cycle the split exists
    to remove.
    """

    # Act & assert
    assert quadrat.pileup_alignments is pileup_pkg.pileup_alignments
    assert quadrat.pileup_many is pileup_pkg.pileup_many
