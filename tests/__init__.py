"""The geometries, paths and readers every test module shares.

Declared here rather than per module so no two test modules can
drift, and the geometries immutably so a corpus built from one cannot
alter the geometry a later corpus is built from.

`builders` writes the suite's synthetic data; what lives here is what
reads it back, states what a reader should answer for it, locates the
project, or runs a child process, none of which belongs to any one
module.
"""

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from types import MappingProxyType

import numpy
import pytest
from hypothesis import HealthCheck, settings

try:
    import polars
except ImportError:
    # Left unbound rather than re-raised, as `builders` leaves `pysam`:
    # the modules that build a window frame import polars themselves,
    # and report its absence in their own way.
    polars = None

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


def bound(pb, name):
    """Return whether a table name is still registered in the shared session."""

    try:
        pb.ctx.table(name)
    except BaseException:
        return False
    return True


def quadrat_tables(pb):
    """Return every table name bound in the shared session under ``quadrat_``.

    The package registers its tables under that prefix and no other, so
    a release can be checked against the prefix rather than against the
    names any one caller happens to use.
    """

    names = pb.sql("SHOW TABLES").collect()["table_name"].to_list()
    return sorted(name for name in names if name.startswith("quadrat_"))


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


def digest_track(table):
    """Return the sha256 of every Parquet file in a track, keyed by name.

    The manifest is excluded: it records absolute input paths, which
    differ between runs into different directories, and it is not what
    the model reads.
    """

    digests = {}
    for name in sorted(os.listdir(table)):
        with open(os.path.join(table, name), "rb") as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def parse_fasta(path):
    """Return contig name to sequence, read back from a FASTA on disk.

    Parsed from the file's own lines rather than through a FASTA
    library, so it is independent of every reader under test.
    """

    sequences = {}
    name = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].strip()
                sequences[name] = []
            else:
                sequences[name].append(line.strip())
    return {name: "".join(parts) for name, parts in sequences.items()}


def window_frame(rows, columns=("sig_start", "sig_end")):
    """Return a window frame of ``(chrom, start, end)`` rows under two bound columns.

    Each field is converted, so a row may carry a numeric contig name
    or numpy integers.
    """

    lo_column, hi_column = columns
    return polars.DataFrame(
        {
            "chrom": [str(row[0]) for row in rows],
            lo_column: [int(row[1]) for row in rows],
            hi_column: [int(row[2]) for row in rows],
        },
        schema={
            "chrom": polars.Utf8,
            lo_column: polars.Int64,
            hi_column: polars.Int64,
        },
    )


def one_hot(residues, alphabet="ACGT"):
    """Return the one-hot matrix a residue string implies, built rather than read.

    An ``int8`` array of one row per residue of ``alphabet`` and one
    column per residue given. Case is ignored, and a residue outside
    the alphabet is an all-zero column.
    """

    out = numpy.zeros((len(alphabet), len(residues)), dtype=numpy.int8)
    for column, base in enumerate(residues.upper()):
        if base in alphabet:
            out[alphabet.index(base), column] = 1
    return out


def budget(plan, ram_bytes, cores):
    """Return the bytes the workers may share on a machine of this size.

    The public restatement of what the planner divides, which is why
    the tests that use it need no private symbol. Here rather than in
    either test module because both restated it and parallel
    near-duplicates drift independently: the two spellings of the
    per-core term already differed -- ``default_partitions(1)`` against
    a pinned core count -- and were equivalent only while
    ``default_partitions(1)`` reduces to ``n_cpus()``.

    The planner module is passed in rather than imported, as ``pysam``
    is to `record_placements` below, and for a stricter reason: this
    package is imported before ``tests/conftest.py`` runs the
    environment block that has to be in place before anything imports
    `quadrat`.
    """

    headroom = int(ram_bytes * plan.MEMORY_HEADROOM)
    machine_cost = min(cores * plan.PEAK_BYTES_PER_CORE, headroom // 2)
    return max(1, headroom - machine_cost)


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


#: How long `run_child` waits. A child that never exits would otherwise
#: hold the whole run until the CI job is cancelled, with the test that
#: hung named nowhere in the log.
CHILD_TIMEOUT = 300


def run_child(command, failure, cwd=None, environment=None):
    """Run ``command`` to completion and return it, failing with its stderr.

    `subprocess.CalledProcessError` carries the stderr but does not
    print it, so a caller that lets it propagate reports an argv and an
    exit status with no diagnosis -- which in a CI log is the whole of
    what there is to read. A child that does not exit within
    `CHILD_TIMEOUT` seconds fails the same way.
    """

    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env=environment,
            timeout=CHILD_TIMEOUT,
        )
    except subprocess.CalledProcessError as error:
        pytest.fail("{}:\n{}".format(failure, error.stderr))
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            "{}: no exit within {} s.\n{}".format(
                failure, CHILD_TIMEOUT, error.stderr or ""
            )
        )
