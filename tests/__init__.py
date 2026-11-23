"""The geometries, paths and readers every test module shares.

Declared here rather than per module so no two test modules can
drift, and the geometries immutably so a corpus built from one cannot
alter the geometry a later corpus is built from.

`builders` writes the suite's synthetic data; what lives here is what
reads it back, states what a reader should answer for it, drains a
loader, locates the project, or runs a child process, none of which
belongs to any one module. The sampler configurations the golden digests are recorded
over are here for the same reason: the plan digests and the batch
digests are pinned in different modules over one set.
"""

import dataclasses
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
from types import MappingProxyType

import numpy
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

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

# The sampler arguments every golden digest is recorded over. The set
# is not arbitrary: a plan skips its permutation without ``shuffle``,
# its jitter draw at a ``max_jitter`` of 0 and its orientation draw
# without ``reverse_complement``, and each skipped call shifts every
# later one, so each branch has a configuration of its own. Every
# configuration but ``grouped_ctl`` is the baseline with the arguments
# of its branch changed. ``grouped_ctl`` changes only the ratio and the
# seed: the golden corpus has no controls and no group of more than
# one channel, and the name is upstream's, kept for provenance.
GOLDEN_BASELINE = MappingProxyType(
    {
        "n_peaks": 24,
        "n_negatives": 8,
        "negative_ratio": 0.5,
        "max_jitter": 2,
        "reverse_complement": True,
        "shuffle": True,
        "random_state": 0,
    }
)
GOLDEN_CONFIGS = MappingProxyType(
    {
        "baseline": GOLDEN_BASELINE,
        "grouped_ctl": MappingProxyType(
            dict(GOLDEN_BASELINE, negative_ratio=0.25, random_state=7)
        ),
        "no_jitter": MappingProxyType(dict(GOLDEN_BASELINE, max_jitter=0)),
        "no_negatives": MappingProxyType(dict(GOLDEN_BASELINE, negative_ratio=0.0)),
        "no_rc": MappingProxyType(dict(GOLDEN_BASELINE, reverse_complement=False)),
        "sequential": MappingProxyType(dict(GOLDEN_BASELINE, shuffle=False)),
    }
)

# The golden configurations in a fixed order, as test ids.
CONFIG_IDS = tuple(sorted(GOLDEN_CONFIGS))


@st.composite
def aggregates(draw):
    """Any record `Aggregate` accepts."""

    # Imported here, not at the top: the conftest imports this module
    # before its environment block, which has to run before anything
    # imports `quadrat`, for the reason its module docstring gives.
    from quadrat import Aggregate

    position = draw(st.sampled_from(("five_prime", "three_prime", "midpoint", "span")))
    exclude = draw(st.integers(0, 0xFFFF))
    require = draw(st.integers(0, 0xFFFF)) & ~(exclude | 0x4)
    floor = draw(st.integers(0, 1000))
    # A midpoint ceiling below one admits no record, so is never drawn.
    lowest = max(floor, 1) if position == "midpoint" else floor
    ceiling = draw(st.one_of(st.none(), st.integers(lowest, 2000)))
    shifts = (
        (0, 0)
        if position in ("midpoint", "span")
        else (draw(st.integers(-50, 50)), draw(st.integers(-50, 50)))
    )
    unstranded = draw(
        st.sampled_from((None, True))
        if position == "midpoint"
        else st.one_of(st.none(), st.booleans())
    )
    spliced = draw(st.booleans()) if position == "span" else False
    return Aggregate(
        exclude_flags=exclude,
        require_flags=require,
        min_mapq=draw(st.integers(0, 255)),
        min_template_length=floor,
        max_template_length=ceiling,
        position=position,
        start_shift=shifts[0],
        end_shift=shifts[1],
        unstranded=unstranded,
        scale_factor=draw(st.floats(-1e6, 1e6, allow_nan=False)),
        spliced=spliced,
    )


def before_92(config):
    """Return a golden configuration with its jitter drawn as upstream drew it before #92.

    A bound ``m`` becomes ``Jitter(-m, m - 1)``, the rule
    jmschrei/cherimoya#92 replaced; a bound of ``0`` stays, being no
    jitter under either.
    """

    # Imported here, not at the top: the conftest imports this module
    # before its environment block, which has to run before anything
    # imports `quadrat`, for the reason its module docstring gives.
    from quadrat.rules import Jitter

    bound = config["max_jitter"]
    return dict(config, max_jitter=Jitter(-bound, bound - 1) if bound else 0)


# Three records' spans on chr1, two overlapping and the third abutting
# the second, and the boundary events a ``"span"`` pileup writes for
# them: ``+1`` at each start and ``-1`` at each end, the abutting pair's
# end and start cancelling on one position.
SPANS = ((10, 20), (15, 30), (30, 34))
DELTA_ROWS = (
    ("chr1", 10, 1.0),
    ("chr1", 15, 1.0),
    ("chr1", 20, -1.0),
    ("chr1", 34, -1.0),
)

# Record fields that make a complete track one no reader can read,
# each with the words of its refusal: a ``signal`` no reader knows, and
# each documented ``signal`` at the other's version. Every reader of
# values is handed each, as `quadrat.tables.track_signal` refuses it.
UNREADABLE_RECORDS = (
    ({"signal": "methylation", "version": 2}, "whose signal is 'methylation'"),
    ({"signal": "counts", "version": 2}, "declares signal 'counts' at manifest"),
    (
        {"signal": "coverage_deltas", "version": 1, "scale": 1.0},
        "declares signal 'coverage_deltas' at manifest",
    ),
)
UNREADABLE_IDS = ("unknown-signal", "counts-at-2", "coverage-deltas-at-1")

# The two kinds of file a track reader refuses by content, as
# `quadrat.tables.track_signal` refuses them: one beginning as a bigWig
# does, and an empty one.
NOT_PARQUET = MappingProxyType({"bigwig": b"\x26\xfc\x8f\x88" + bytes(60), "empty": b""})

# Records placed so a span ends on each contig's last base of
# `TWO_CONTIGS`, one forward and one reverse, beside one that ends
# well inside chr1: ``(name, chrom, pos, cigar, flag)``, as
# ``corpus_factory`` places them.
LAST_BASE_READS = (
    ("inside", "chr1", 50, "20M", 0),
    ("last1", "chr1", 980, "20M", 0),
    ("last2", "chr2", 470, "30M", 16),
)

# A geometry and records for spliced coverage: ``(name, chrom, pos,
# cigar, flag)``, as ``corpus_factory`` places them, under a header that
# declares both contigs longer than the reference. They hold the case
# the CIGAR decomposition is easiest to get wrong in each way: two
# skipped runs of one length in one record, three identical records, a
# deletion on either side of a skip, clips and insertions beside one,
# ``=`` and ``X``, runs closing past a contig and a record starting past
# one, beside unspliced records and a deletion with no skip.
SPLICED_CONTIGS = MappingProxyType({"chr1": 2100, "chr2": 500})
SPLICED_HEADER = MappingProxyType({"chr1": 2200, "chr2": 600})
SPLICED_READS = (
    ("issue", "chr1", 2000, "5M10N5M", 0),
    ("repeat", "chr1", 100, "23M112N20M102N32M102N25M", 16),
    ("same1", "chr1", 600, "10M50N10M", 0),
    ("same2", "chr1", 600, "10M50N10M", 0),
    ("same3", "chr1", 600, "10M50N10M", 0),
    ("del_skip", "chr1", 900, "10M3D2N5M", 0),
    ("skip_del", "chr1", 950, "5M2N3D10M", 16),
    ("clipped", "chr1", 1200, "3S10M2I5M40N8M4S", 0),
    ("matched", "chr1", 1300, "5=1X4=30N10M", 16),
    ("plain", "chr1", 1250, "30M", 0),
    ("plain_r", "chr1", 1260, "30M", 16),
    ("deleted", "chr2", 100, "10M5D10M", 0),
    ("past", "chr1", 2080, "10M30N20M", 0),
    ("past_r", "chr2", 470, "20M20N20M", 16),
    ("beyond", "chr2", 520, "5M5N5M", 0),
)

# How long a loader with workers may wait on one, so a worker that
# never answers fails its test rather than stalling the run.
WORKER_TIMEOUT = 120

#: How long `run_child` waits. A child that never exits would otherwise
#: hold the whole run until the CI job is cancelled, with the test that
#: hung named nowhere in the log.
CHILD_TIMEOUT = 300

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


def coverage(spans, start, end, dtype=numpy.float64):
    """Return the ``chr1`` coverage ``[start, end)`` spans imply, one base at a time."""

    out = numpy.zeros(end - start, dtype=dtype)
    for lo, hi in spans:
        for base in range(max(lo, start), min(hi, end)):
            out[base - start] += 1
    return out


def dense(rows, sizes=TWO_CONTIGS, dtype=numpy.float32):
    """Return the per-contig dense arrays one channel's rows imply.

    Rows landing on one position add, as every reader of a track adds
    them.
    """

    arrays = {chrom: numpy.zeros(size, dtype=dtype) for chrom, size in sizes.items()}
    for chrom, pos, value in rows:
        arrays[chrom][pos] += value
    return arrays


def drain(loader, n_epochs=2):
    """Return every batch a loader yields over ``n_epochs`` passes, in order.

    Each batch is a tuple of its tensors, cloned, so a batch that is a
    view of a fetch a loader goes on to reuse is still what it was when
    it was yielded.
    """

    batches = []
    for _ in range(n_epochs):
        for batch in loader:
            batches.append(tuple(tensor.clone() for tensor in batch))
    return batches


def same_batches(first, second):
    """Return whether two drained streams hold equal tensors, batch for batch.

    Equal means the same number of batches, the same number of tensors
    in each, and every pair of tensors equal in shape, dtype and value.
    """

    if len(first) != len(second):
        return False
    return all(
        len(a) == len(b) and all(p.dtype == q.dtype and p.equal(q) for p, q in zip(a, b))
        for a, b in zip(first, second)
    )


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


def oracle_kept(read, aggregate):
    """Return whether a record passes an aggregate's filter, judged by pysam.

    Read off pysam's own fields for each clause the aggregate names,
    so it agrees with the pileup only where both read a record alike.
    """

    length = abs(read.template_length)
    return not (
        read.is_unmapped
        or read.flag & aggregate.exclude_flags
        or read.flag & aggregate.require_flags != aggregate.require_flags
        or read.mapping_quality < aggregate.min_mapq
        or length < aggregate.min_template_length
        or (
            aggregate.max_template_length is not None
            and length > aggregate.max_template_length
        )
        or (aggregate.position == "midpoint" and read.template_length <= 0)
    )


def oracle_pileup(bam_paths, chrom_sizes, pysam, aggregate):
    """Return the dense track per contig that alignments imply, read with pysam.

    bam2bw's loop, densified and filtered: every record `oracle_kept`
    keeps on a known contig contributes the one position its rule
    names, and a position outside its contig is dropped. The midpoint
    is found from the two mates' 5' ends, paired by name, rather than
    from the template length, so it is a second derivation of the
    fragment and not a restatement of the first. ``"span"`` is
    `oracle_depth`'s. ``pysam`` is passed in, as it is to
    `record_placements`.
    """

    def five_prime(read, shift=(0, 0)):
        if read.is_forward:
            return read.reference_start + shift[0]
        return read.reference_end + shift[1] - 1

    shifts = (aggregate.start_shift, aggregate.end_shift)
    dense = {c: numpy.zeros(n, dtype=numpy.float64) for c, n in chrom_sizes.items()}
    for path in bam_paths:
        with pysam.AlignmentFile(path, "rb") as bam:
            reads = list(bam.fetch(until_eof=True))
        mates = {}
        for read in reads:
            mates.setdefault(read.query_name, []).append(read)
        for read in reads:
            if not oracle_kept(read, aggregate) or read.reference_name not in dense:
                continue
            if aggregate.position == "five_prime":
                pos = five_prime(read, shifts)
            elif aggregate.position == "three_prime":
                if read.is_forward:
                    pos = read.reference_end + aggregate.end_shift - 1
                else:
                    pos = read.reference_start + aggregate.start_shift
            else:
                (mate,) = [
                    other for other in mates[read.query_name] if other is not read
                ]
                ends = sorted((five_prime(read), five_prime(mate)))
                pos = ends[0] + (ends[1] - ends[0]) // 2
            if 0 <= pos < chrom_sizes[read.reference_name]:
                dense[read.reference_name][pos] += 1
    return {c: v * aggregate.scale_factor for c, v in dense.items()}


def oracle_depth(bam_paths, chrom_sizes, pysam, aggregate):
    """Return per-base depth per contig over the records an aggregate keeps.

    What ``samtools depth -a -J`` reports, found by walking each kept
    record's CIGAR as pysam's ``cigartuples`` gives it: a base counts
    where an ``M``, ``D``, ``=`` or ``X`` operation covers it, an ``N``
    passes over the bases it skips without covering them, and ``I``,
    ``S``, ``H`` and ``P`` consume no reference. A base past its contig
    is dropped, as is a record on a contig ``chrom_sizes`` does not
    name. ``pysam`` is passed in, as it is to `record_placements`.
    """

    covering = (pysam.CMATCH, pysam.CDEL, pysam.CEQUAL, pysam.CDIFF)
    dense = {c: numpy.zeros(n, dtype=numpy.float64) for c, n in chrom_sizes.items()}
    for path in bam_paths:
        with pysam.AlignmentFile(path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if not oracle_kept(read, aggregate) or read.reference_name not in dense:
                    continue
                row = dense[read.reference_name]
                pos = read.reference_start
                for op, length in read.cigartuples:
                    if op in covering:
                        row[pos : pos + length] += 1
                    if op in covering or op == pysam.CREF_SKIP:
                        pos += length
    return {c: v * aggregate.scale_factor for c, v in dense.items()}


def strand_channels(oracle, bam_paths, chrom_sizes, pysam, aggregate):
    """Return what each channel of a pileup under ``aggregate`` should hold.

    One mapping per channel, in channel order, as ``oracle`` (`oracle_pileup`
    or `oracle_depth`) answers for that channel's records: every kept
    record under an unstranded aggregate, and otherwise the forward
    records and then the reverse ones. A channel's records are the pooled
    filter with its strand's flag excluded or required, so a mask that
    already excludes ``0x10`` leaves the reverse channel empty. ``pysam``
    is passed in, as it is to `record_placements`.
    """

    pooled = dataclasses.replace(aggregate, unstranded=True)
    if aggregate.unstranded:
        return [oracle(bam_paths, chrom_sizes, pysam, pooled)]
    forward = dataclasses.replace(pooled, exclude_flags=pooled.exclude_flags | 0x10)
    channels = [oracle(bam_paths, chrom_sizes, pysam, forward)]
    if pooled.exclude_flags & 0x10:
        channels.append({c: numpy.zeros(n) for c, n in chrom_sizes.items()})
    else:
        reverse = dataclasses.replace(pooled, require_flags=pooled.require_flags | 0x10)
        channels.append(oracle(bam_paths, chrom_sizes, pysam, reverse))
    return channels


def coverage_from_track(paths, chrom_sizes):
    """Return each channel's dense values per contig, read through the store.

    One array per contig of ``chrom_sizes``, a row per channel in the
    order of ``paths``. A channel of coverage deltas comes back as the
    coverage its prefix sum gives, and a channel of counts as the
    counts.
    """

    # On call, as `pile_up_spans` imports the package: importing it here
    # at module scope would precede the cache variable `tests.conftest`
    # sets.
    stores = pytest.importorskip("quadrat.stores")
    store = stores.TrackStore(paths)
    return {
        chrom: store.window(chrom, 0, size).numpy().astype(numpy.float64)
        for chrom, size in chrom_sizes.items()
    }


def pile_up_spans(corpus, name, pysam, scale_factor=1.0, spliced=False):
    """Pile a corpus's alignments up unstranded under ``"span"``, beside pysam's depth.

    The track is written at ``scale_factor`` and ``spliced`` as
    ``name`` inside the corpus's own directory, and the depth is
    `oracle_depth`'s over the same records at the same scale, per
    contig of the corpus. Skips the requesting test without the
    ``pileup`` extra. ``pysam`` is passed in, as it is to
    `record_placements`.

    :returns: ``(track, depth)``.
    """

    pytest.importorskip("polars_bio")
    pileup = pytest.importorskip("quadrat.pileup")
    aggregate = pileup.Aggregate(
        position="span", unstranded=True, scale_factor=scale_factor, spliced=spliced
    )
    (track,) = pileup.pileup_alignments(
        corpus.bams, corpus.fasta, corpus.path(name), aggregate=aggregate
    )
    return track, oracle_depth(corpus.bams, corpus.chrom_sizes, pysam, aggregate)


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
