"""The geometries, paths and readers every test module shares.

Declared here rather than per module so no two test modules can
drift, and the geometries immutably so a corpus built from one cannot
alter the geometry a later corpus is built from.

`builders` writes the suite's synthetic data; what lives here is what
reads it back, locates the project, or runs a child process, none of
which belongs to any one module.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest
from hypothesis import HealthCheck, settings

# Two contigs is the default: the smallest geometry in which a contig
# boundary exists at all. Three is what a non-uniform contig split
# needs, because across two contigs "more on the first than on any
# other" has only one comparison to make.
TWO_CONTIGS = MappingProxyType({"chr1": 1000, "chr2": 500})
THREE_CONTIGS = MappingProxyType({"chr1": 1000, "chr2": 500, "chr3": 400})

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The settings every property test that writes real files runs under.
# Two decisions live here rather than once per test: the health check
# has to be suppressed because `tmp_path` and `pysam` are requested
# per test, and the deadline has to go because a cold first call spends
# a second in pysam's sort and index against Hypothesis's 200ms
# default -- which then reports a filesystem timing as a falsifying
# example. The properties that stay in memory keep the deadline.
FILE_BACKED = settings(
    deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


def load_build_hook(name):
    """Return the build hook module ``name``, loaded from its path.

    The hooks are build-time infrastructure rather than part of the
    package, and ``build-hooks`` is not a legal module name, so they
    are loaded from a path the way hatchling itself loads them.
    """

    path = PROJECT_ROOT / "build-hooks" / "{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_placements(path, pysam, until_eof=False):
    """Return ``(contig, start)`` for every record in a BAM, in file order.

    ``until_eof`` is what makes this a count of records rather than of
    records the index can find: a plain `fetch` drops the unplaced
    ones without saying so.
    """

    with pysam.AlignmentFile(path, "rb") as bam:
        return [
            (read.reference_name, read.reference_start)
            for read in bam.fetch(until_eof=until_eof)
        ]


def run_child(command, failure, cwd=None, environment=None):
    """Run ``command`` to completion and return it, failing with its stderr.

    `subprocess.CalledProcessError` carries the stderr but does not
    print it, so a caller that lets it propagate reports an argv and an
    exit status with no diagnosis -- which in a CI log is the whole of
    what there is to read.
    """

    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail("{}:\n{}".format(failure, error.stderr))
