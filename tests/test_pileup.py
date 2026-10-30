"""Tests for running many cut-site pileups at once.

Ported from cherimoya's ``tests/test_pileup.py``, which held the whole data path in
one module. The names it reached for as ``giql_io.x`` and
``pileup.x`` are spread over several modules here, so the references
are rewritten and nothing else is: the assertions are the evidence
that the split preserved behaviour, and rewriting them in the same
change that moves them would destroy that evidence.

Style follows cherimoya rather than this project's test conventions --
natural-language names, no Arrange-Act-Assert phases. Converting them
is a separate change, tracked on the epic.
"""

import hashlib
import os
import random
import shutil
import warnings

import pytest

pytest.importorskip("polars_bio")
pysam = pytest.importorskip("pysam")
polars = pytest.importorskip("polars")


from quadrat import alignments, tables
from quadrat import pileup as pileup_pkg
from quadrat.pileup import plan, run

os.environ.setdefault("TQDM_DISABLE", "1")

CHROM_SIZES = {"chr1": 1000, "chr2": 500}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pileup")


@pytest.fixture(scope="module")
def fasta(data_dir):
    rng = random.Random(0)
    path = str(data_dir / "genome.fa")
    with open(path, "w") as f:
        for chrom, size in CHROM_SIZES.items():
            seq = "".join(rng.choice("ACGT") for _ in range(size))
            f.write(">{}\n".format(chrom))
            for i in range(0, size, 60):
                f.write(seq[i : i + 60] + "\n")
    pysam.faidx(path)
    return path


@pytest.fixture(scope="module")
def bams(data_dir):
    """Four small BAMs, each with a different read layout."""

    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 1000}, {"SN": "chr2", "LN": 500}],
        }
    )

    paths = []
    for index in range(4):
        rng = random.Random(index)
        reads = []
        for i in range(40):
            chrom = "chr1" if i % 3 else "chr2"
            start = rng.randrange(0, CHROM_SIZES[chrom] - 40)
            read = pysam.AlignedSegment(header)
            read.query_name = "r{}_{}".format(index, i)
            read.flag = 16 if i % 2 else 0
            read.reference_id = header.get_tid(chrom)
            read.reference_start = start
            read.cigarstring = "20M"
            read.query_sequence = "A" * 20
            read.query_qualities = pysam.qualitystring_to_array("I" * 20)
            read.mapping_quality = 30
            reads.append(read)

        path = str(data_dir / "s{}.bam".format(index))
        unsorted = path + ".unsorted.bam"
        with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
            for read in sorted(reads, key=lambda r: (r.reference_id, r.reference_start)):
                out.write(read)
        pysam.sort("-o", path, unsorted)
        pysam.index(path)
        os.remove(unsorted)
        paths.append(path)
    return paths


def _jobs(bams, fasta, out, prefix, unstranded=True):
    return [
        {
            "alignments": [bam],
            "sequences": fasta,
            "name": os.path.join(str(out), "{}{}".format(prefix, i)),
            "unstranded": unstranded,
            "pos_shift": 4,
            "neg_shift": -4,
        }
        for i, bam in enumerate(bams)
    ]


def _digest(table):
    """sha256 of every Parquet file in a cut-site table, by name.

    The manifest is excluded: it records absolute input paths, which
    differ between runs into different directories, and it is not what
    the model reads."""

    out = {}
    for name in sorted(os.listdir(table)):
        with open(os.path.join(table, name), "rb") as handle:
            out[name] = hashlib.sha256(handle.read()).hexdigest()
    return out


def test_backends_should_write_byte_identical_tables(bams, fasta, tmp_path):
    """GIVEN the same jobs run serially and across a process pool
    WHEN the written tables are hashed
    THEN every Parquet file is byte-for-byte the same.

    This is the whole contract: a backend is a scheduling choice, not a
    numerical one. It holds exactly because the aggregate is an integer
    COUNT(*) scaled once after grouping, so no partition order can
    accumulate floats differently."""

    a, b = tmp_path / "serial", tmp_path / "process"
    a.mkdir()
    b.mkdir()

    serial = run.pileup_many(_jobs(bams, fasta, a, "x"), backend="serial")
    parallel = run.pileup_many(_jobs(bams, fasta, b, "x"), backend="process", workers=2)

    assert [os.path.basename(p) for group in serial for p in group] == [
        os.path.basename(p) for group in parallel for p in group
    ]

    for lhs, rhs in zip(serial, parallel):
        for left, right in zip(lhs, rhs):
            assert _digest(left) == _digest(right)
            assert _digest(left), "table wrote no Parquet files"


def test_every_table_is_verifiable_after_a_parallel_run(bams, fasta, tmp_path):
    """GIVEN jobs run across a process pool
    WHEN each written table is checked for completeness
    THEN all of them pass.

    Each pileup commits its own table atomically, so concurrency must not
    leave any of them unverifiable."""

    jobs = _jobs(bams, fasta, tmp_path, "x")
    for group in run.pileup_many(jobs, backend="process", workers=2):
        for table in group:
            assert tables.require_complete(table) == table
            assert tables.read_manifest(table)["rows"] > 0


def test_stranded_jobs_keep_channel_order(bams, fasta, tmp_path):
    """GIVEN stranded jobs run across a process pool
    WHEN the returned paths are inspected
    THEN each job yields its plus table before its minus table."""

    jobs = _jobs(bams, fasta, tmp_path, "s", unstranded=False)
    groups = run.pileup_many(jobs, backend="process", workers=2)

    for group in groups:
        assert len(group) == 2
        assert group[0].endswith(".+" + tables.CUT_SITE_SUFFIX)
        assert group[1].endswith(".-" + tables.CUT_SITE_SUFFIX)


def test_results_come_back_in_job_order(bams, fasta, tmp_path):
    """GIVEN jobs whose names make their order recoverable
    WHEN they are run concurrently
    THEN the results are in submission order, not completion order."""

    jobs = _jobs(bams, fasta, tmp_path, "ordered")
    groups = run.pileup_many(jobs, backend="process", workers=3)

    assert [os.path.basename(group[0]) for group in groups] == [
        "ordered{}{}".format(i, tables.CUT_SITE_SUFFIX) for i in range(len(jobs))
    ]


def test_target_partitions_is_divided_among_workers():
    """GIVEN a worker count
    WHEN the default partition count is computed
    THEN the machine is divided rather than claimed by each worker.

    The setting is process-global in DataFusion, so eight pileups each
    asking for every core would oversubscribe the thread pool eightfold."""

    cpus = plan._n_cpus()
    assert plan.default_partitions(1) == cpus
    assert plan.default_partitions(cpus) == 1
    assert plan.default_partitions(2) == max(1, cpus // 2)
    # Never zero, however many workers are asked for.
    assert plan.default_partitions(cpus * 100) == 1


def test_the_memory_bound_scales_with_the_input(monkeypatch):
    """GIVEN jobs whose alignments differ in size
    WHEN the default worker count is computed
    THEN the budget tracks the input rather than a fixed guess.

    A pileup peaks at roughly three times the bytes it reads, so what is
    safe for a 300 MB alignment is fatal for a 4 GB one. The constant
    this replaced said 2 GB regardless, which on a 64 GB machine admitted
    32 workers against a real need of about 380 GB.

    The bound is now something to spend rather than only to obey: a
    large input is split into contig batches until it fits, instead of
    the machine simply running fewer pileups."""

    # Pin the machine so the test does not depend on the host.
    monkeypatch.setattr(plan, "_n_cpus", lambda: 64)
    monkeypatch.setattr(
        plan.os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": (64 * 1024**3) // 4096}[
            name
        ],
    )

    big, big_groups = plan.plan_pileup(32, 4 * 1024**3)
    small, small_groups = plan.plan_pileup(32, 256 * 1024**2)

    assert big < small, "a larger input must not permit more workers"
    # A 4 GB input needs 8 GB in one batch, so the budget admits four
    # pileups at once -- which is what this did before contig batches
    # existed. Splitting it five ways brings the peak to 3.2 GB and so
    # buys back most of the machine.
    assert big > 4
    assert big_groups > 1

    # A small input is not split, because what the batches divide is
    # only the part of the peak that follows the input, and on a
    # 256 MB alignment that part is already small beside the fixed
    # cost of running a worker at all. Splitting anyway would buy a
    # few percent more concurrency for a quarter more wall clock.
    assert small_groups == 1

    # Nor does it reach the job count: thirty-two workers would cost
    # 64 GB in fixed per-worker memory alone, which is the whole
    # machine. The bound that binds here is the one the single-multiple
    # model had no way to express.
    assert small < 32
    assert small * plan.PEAK_BYTES_PER_WORKER <= plan._memory_budget()


def test_the_memory_bound_is_conservative_without_sizes(monkeypatch):
    """GIVEN jobs whose alignments cannot be sized
    WHEN the default worker count is computed
    THEN it falls back to the measured per-worker figure.

    Better to under-subscribe a machine than to guess low and have the
    run die partway."""

    monkeypatch.setattr(plan, "_n_cpus", lambda: 64)
    monkeypatch.setattr(
        plan.os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": (64 * 1024**3) // 4096}[
            name
        ],
    )

    workers, groups = plan.plan_pileup(32)
    # Twelve GB per worker ungrouped, so four at once on a 64 GB
    # machine; batches bring that down without ever guessing the peak
    # is smaller than it is.
    assert plan.projected_peak(None) == plan.UNKNOWN_INPUT_PEAK_BYTES
    assert 4 <= workers < 32
    assert groups > 1


def test_the_core_count_is_the_allocation_not_the_machine(monkeypatch):
    """GIVEN a process restricted to part of the machine
    WHEN the usable core count is read
    THEN it is the restriction, not what the machine has.

    A pileup is usually launched under something that confines it --
    SLURM, a cgroup, a container quota, `taskset` -- and `cpu_count`
    reports the whole node whatever the allocation. Believing it sets
    `target_partitions` to a parallelism the job cannot have and lets
    the worker cap oversubscribe by the restriction's factor. Measured
    on js031 under `taskset`, the harness went on recording 256 cores
    while confined to 32."""

    monkeypatch.setattr(
        plan.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    monkeypatch.setattr(plan.os, "cpu_count", lambda: 256)
    assert plan._n_cpus() == 8

    # Where the platform has no affinity call, the machine is the only
    # answer there is.
    monkeypatch.delattr(plan.os, "sched_getaffinity", raising=False)
    assert plan._n_cpus() == 256


def _machine(monkeypatch, cores, ram_gb):
    monkeypatch.setattr(plan, "_n_cpus", lambda: cores)
    monkeypatch.setattr(
        plan.os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": (ram_gb * 1024**3) // 4096}[
            name
        ],
    )


def test_the_peak_keeps_a_part_the_batches_cannot_divide():
    """GIVEN a projected peak at several batch counts
    WHEN the batch count rises
    THEN only the part that follows the input falls, and it falls in
    proportion rather than as a square root.

    The single multiple this replaced had nowhere to put the engine,
    the interpreter and the buffers a worker holds whatever it reads,
    so it believed eight batches made all of it nearly three times
    smaller. Measured across the twelve arms in
    `scripts/results/ladder-after-fix/`, that part does not move."""

    fixed = plan.PEAK_BYTES_PER_WORKER
    ratio = plan.PEAK_BYTES_PER_INPUT_BYTE
    job = 8 * 1024**3

    assert plan.projected_peak(job, 1) == fixed + int(job * ratio)

    # In proportion: four batches scan a quarter of the bytes and hold
    # a quarter of the distinct positions.
    quarter = plan.projected_peak(job, 4) - fixed
    assert quarter == pytest.approx(int(job * ratio) // 4, rel=1e-6)

    # And the floor is the fixed part, however far it is split.
    assert plan.projected_peak(job, 1024) > fixed
    assert plan.projected_peak(job, 1024) < fixed * 1.01


def test_the_machine_cost_comes_off_the_budget_not_each_worker(monkeypatch):
    """GIVEN two machines with the same memory and different core counts
    WHEN the budget the workers share is computed
    THEN the larger one reserves more, because what the engine holds
    per core is spent once rather than once per worker.

    Measured by confining the process tree with `taskset` and changing
    nothing else: at twelve workers and eight partitions the peak goes
    50.2, 62.4, 70.0 and 73.8 GB at 32, 64, 128 and 256 cores. Holding
    the cores and sweeping `target_partitions` 4 to 32 instead moves it
    72.0 to 76.8 GB, inside the replicate spread -- so the cost follows
    the cores and not the partitions, and not the fan-out either."""

    _machine(monkeypatch, 16, 256)
    small = plan._memory_budget()
    _machine(monkeypatch, 256, 256)
    large = plan._memory_budget()

    assert large < small
    assert small - large == 240 * plan.PEAK_BYTES_PER_CORE

    # Never negative, however many cores sit on a small machine.
    _machine(monkeypatch, 4096, 8)
    assert plan._memory_budget() >= 1


def test_batches_are_not_bought_for_a_marginal_gain(monkeypatch):
    """GIVEN an input small enough that the fixed cost dominates
    WHEN the batch count is planned
    THEN it stays at one, because splitting buys a few percent of
    concurrency for a quarter more wall clock.

    The part of the peak the batches divide can always be driven to
    nearly nothing, so without this the planner takes every batch it
    is allowed on inputs where none of them help."""

    _machine(monkeypatch, 64, 64)

    workers, groups = plan.plan_pileup(32, 256 * 1024**2)
    assert groups == 1

    # Not because splitting was unavailable -- it would have admitted
    # more workers, just not enough more to be worth the wall clock.
    assert plan.plan_pileup(32, 256 * 1024**2) == (workers, 1)
    more = 0
    for g in range(2, plan.MAX_CONTIG_GROUPS + 1):
        fits = plan._memory_budget() // plan.projected_peak(256 * 1024**2, g)
        more = max(more, min(32, fits))
    assert workers < more < workers * plan.CONTIG_GROUP_GAIN

    # Where the gain is real it is still taken: a 4 GB input is split,
    # because one batch of it admits far fewer workers than several.
    big, big_groups = plan.plan_pileup(32, 4 * 1024**3)
    whole = plan._memory_budget() // plan.projected_peak(4 * 1024**3, 1)
    assert big_groups > 1
    assert big >= whole * plan.CONTIG_GROUP_GAIN


def _roomy(monkeypatch):
    # js031, where the overflow was found: 256 cores and 1007 GB, which
    # is enough memory that the planner never had to split anything.
    monkeypatch.setattr(plan, "_n_cpus", lambda: 256)
    monkeypatch.setattr(
        plan.os,
        "sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": (1007 * 1024**3) // 4096}[
            name
        ],
    )


def test_the_scan_cap_counts_batches_from_the_bytes(monkeypatch):
    """GIVEN an input size
    WHEN the batch floor is computed
    THEN it is the fewest batches keeping one scan under the cap."""

    cap = plan.MAX_GROUP_INPUT_BYTES
    assert plan._group_floor(None) == 1
    assert plan._group_floor(0) == 1
    assert plan._group_floor(cap) == 1
    assert plan._group_floor(cap + 1) == 2

    # The ceiling is real and the floor does not pretend otherwise: past
    # `MAX_CONTIG_GROUPS` batches it asks for no more than it can get.
    assert plan._group_floor(1024**4) == plan.MAX_CONTIG_GROUPS


def test_a_large_job_is_split_however_large_the_machine_is(monkeypatch):
    """GIVEN a machine with memory to spare and one very large input
    WHEN the pileup is planned
    THEN the genome is split anyway, because the bound that matters is
    how much one query scans rather than how much memory is free.

    Solving from headroom alone answers one batch here, which is how a
    21.2 GB alignment came to be read as a single aggregate over
    291,063,148 records."""

    _roomy(monkeypatch)

    assert plan.plan_pileup(1, int(21.2 * 1024**3)) == (1, 3)

    # And not otherwise: batches cost 25% of the pileup wall clock at
    # the sizes the ladder measured, so the cap must stay clear of them.
    assert plan.plan_pileup(1, 2 * 1024**3)[1] == 1


def test_a_lopsided_job_set_splits_only_the_large_jobs(
    bams, fasta, tmp_path, monkeypatch
):
    """GIVEN a job set whose mean size is small and one job enormous
    WHEN the fan-out is planned
    THEN the large job is split and the small ones are left whole.

    The planner is handed the mean, which on a set like this is small:
    the rung-3 benchmark set averaged 3.28 GB across 68 alignments and
    held one of 21.2 GB, and it was that one that died. A cap read
    from the mean alone would have missed it. A job whose caller
    pinned a count keeps it."""

    _roomy(monkeypatch)

    # Five small jobs against two of 24 GB keeps the mean under the cap,
    # so nothing but the large jobs own bytes can be what splits them.
    small = []
    for i in range(5):
        path = str(tmp_path / "small{}.bam".format(i))
        shutil.copy(bams[i % len(bams)], path)
        small.append(path)
    huge = str(tmp_path / "huge.bam")
    shutil.copy(bams[0], huge)
    pinned = str(tmp_path / "pinned.bam")
    shutil.copy(bams[0], pinned)

    real = run.os.path.getsize
    monkeypatch.setattr(
        run.os.path,
        "getsize",
        lambda path: 24 * 1024**3 if path in (huge, pinned) else real(path),
    )

    seen = []
    # `_run_process` rather than `_run_one`: the real one dispatches
    # through loky, so a patched callable never reaches the children.
    monkeypatch.setattr(
        run, "_run_process", lambda jobs, *a, **k: seen.extend(jobs) or []
    )

    jobs = [
        {
            "name": str(tmp_path / "s{}".format(i)),
            "alignments": [path],
            "sequences": fasta,
        }
        for i, path in enumerate(small)
    ]
    jobs.append(
        {"name": str(tmp_path / "huge"), "alignments": [huge], "sequences": fasta}
    )
    jobs.append(
        {
            "name": str(tmp_path / "pinned"),
            "alignments": [pinned],
            "sequences": fasta,
            "contig_groups": 1,
        }
    )
    run.pileup_many(jobs, backend="process")

    groups = {os.path.basename(job["name"]): job["contig_groups"] for job in seen}
    assert groups["huge"] == 3, "the 24 GB job must be split"
    assert all(groups["s{}".format(i)] == 1 for i in range(5)), (
        "a small job must not be split for its neighbour's sake"
    )
    assert groups["pinned"] == 1, "an explicit count must survive the cap"


def test_job_sizes_reach_the_worker_count(bams, fasta, tmp_path, monkeypatch):
    """GIVEN a job list whose alignments are on disk
    WHEN pileup_many plans the fan-out
    THEN it sizes the budget from those files.

    The sizes are available at the call site, so a caller who does not
    name a worker count should still get a bound fitted to their data."""

    seen = {}
    real = plan.plan_pileup

    def spy(n_jobs, bytes_per_job=None):
        seen["bytes_per_job"] = bytes_per_job
        return real(n_jobs, bytes_per_job)

    monkeypatch.setattr(run, "plan_pileup", spy)
    jobs = _jobs(bams, fasta, tmp_path, "sized")
    run.pileup_many(jobs, backend="process")

    assert seen["bytes_per_job"] is not None and seen["bytes_per_job"] > 0


def test_unsizable_alignments_fall_back(tmp_path):
    """GIVEN a job naming an alignment that is not on disk
    WHEN the per-job size is taken
    THEN it is None rather than an error or a wrong number."""

    assert run._bytes_per_job([{"alignments": [str(tmp_path / "absent.bam")]}]) is None
    assert run._bytes_per_job([{"alignments": []}]) is None


def test_default_workers_never_exceeds_the_job_count():
    """GIVEN fewer jobs than cores
    WHEN the default worker count is computed
    THEN it is capped at the number of jobs."""

    assert plan.default_workers(0) == 0
    assert plan.default_workers(1) == 1
    assert plan.default_workers(3) <= 3


def test_unknown_backend_is_rejected_by_name():
    """GIVEN an unrecognized backend
    WHEN pileup_many is called
    THEN it names the backends that do exist."""

    with pytest.raises(ValueError, match="serial, process"):
        run.pileup_many([], backend="dask")


def test_malformed_jobs_are_rejected_before_any_work(bams, fasta, tmp_path):
    """GIVEN a job missing a required argument
    WHEN pileup_many is called
    THEN it fails naming the job and the argument, before dispatching.

    Validating up front matters more for the concurrent backends: a
    KeyError raised inside a worker after fifteen minutes of other jobs
    is a far worse way to learn the job list was wrong."""

    jobs = _jobs(bams, fasta, tmp_path, "x")
    del jobs[2]["sequences"]
    with pytest.raises(KeyError, match="job 2"):
        run.pileup_many(jobs, backend="process", workers=2)

    with pytest.raises(TypeError, match="job 0"):
        run.pileup_many(["not-a-dict"], backend="serial")

    assert not os.listdir(tmp_path)


def test_empty_job_list_is_a_no_op():
    """GIVEN no jobs
    WHEN pileup_many is called
    THEN it returns nothing without starting a pool."""

    for backend in run.BACKENDS:
        assert run.pileup_many([], backend=backend) == []


###
# Contig grouping
#
# Splitting the genome into batches is a memory optimisation and must be
# nothing else, so the tests here are about equality with the ungrouped
# form rather than about the peak -- a byte comparison is checkable in
# CI, where a memory figure is not.
###


def _one_contig_bam(path, chrom="chr1", size=1000, reads=20):
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": chrom, "LN": size}],
        }
    )
    rng = random.Random(99)
    out = []
    for i in range(reads):
        read = pysam.AlignedSegment(header)
        read.query_name = "solo{}".format(i)
        read.flag = 16 if i % 2 else 0
        read.reference_id = 0
        read.reference_start = rng.randrange(0, size - 40)
        read.cigarstring = "20M"
        read.query_sequence = "A" * 20
        read.query_qualities = pysam.qualitystring_to_array("I" * 20)
        read.mapping_quality = 30
        out.append(read)

    unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as handle:
        for read in sorted(out, key=lambda r: r.reference_start):
            handle.write(read)
    pysam.sort("-o", path, unsorted)
    pysam.index(path)
    os.remove(unsorted)
    return path


@pytest.mark.parametrize("groups", [2, 4])
def test_contig_groups_write_byte_identical_tables(bams, fasta, tmp_path, groups):
    """GIVEN a pileup run as one aggregate and as several contig batches
    WHEN the written tables are hashed
    THEN every Parquet file is byte-for-byte the same.

    Grouping must be a memory optimisation and nothing else. It holds
    because batches partition the contigs, and a cut site belongs to
    exactly one contig, so no count can straddle a batch boundary."""

    whole = str(tmp_path / "whole")
    split = str(tmp_path / "split")

    one = pileup_pkg.pileup_alignments(
        [bams[0]],
        fasta,
        whole,
        unstranded=True,
        pos_shift=4,
        neg_shift=-4,
        contig_groups=1,
    )
    many = pileup_pkg.pileup_alignments(
        [bams[0]],
        fasta,
        split,
        unstranded=True,
        pos_shift=4,
        neg_shift=-4,
        contig_groups=groups,
    )

    for left, right in zip(one, many):
        assert _digest(left) == _digest(right)
        assert _digest(left), "table wrote no Parquet files"
        assert tables.read_manifest(left)["rows"] == tables.read_manifest(right)["rows"]


def test_grouping_tolerates_a_reference_contig_the_bam_never_mentions(
    bams, data_dir, tmp_path
):
    """GIVEN a reference carrying a contig absent from the BAM header
    WHEN the pileup is split into contig batches
    THEN it completes and matches the ungrouped run.

    polars-bio raises rather than returning nothing when asked for a
    region its header does not name, so batches have to be built from
    the intersection. A reference with more contigs than the alignment
    is the normal case, not an edge case -- hg38 has 455 where an ATAC
    BAM has 195."""

    rng = random.Random(7)
    path = str(tmp_path / "extra.fa")
    sizes = dict(CHROM_SIZES)
    sizes["chrUnplaced"] = 300
    with open(path, "w") as handle:
        for chrom, size in sizes.items():
            handle.write(">{}\n".format(chrom))
            seq = "".join(rng.choice("ACGT") for _ in range(size))
            for i in range(0, size, 60):
                handle.write(seq[i : i + 60] + "\n")
    pysam.faidx(path)

    whole = pileup_pkg.pileup_alignments(
        [bams[0]], path, str(tmp_path / "w"), unstranded=True, contig_groups=1
    )
    split = pileup_pkg.pileup_alignments(
        [bams[0]], path, str(tmp_path / "s"), unstranded=True, contig_groups=3
    )

    assert _digest(whole[0]) == _digest(split[0])
    assert "chrUnplaced.parquet" not in _digest(split[0])


def test_pooled_alignments_with_unlike_headers_keep_every_record(bams, fasta, tmp_path):
    """GIVEN one channel pooling a two-contig BAM and a one-contig BAM
    WHEN the pileup is split into contig batches
    THEN the table matches the ungrouped run.

    Each branch is restricted to the contigs its own table has. Taking
    the intersection across the pooled alignments instead would drop
    chr2 entirely and still look like a successful run."""

    solo = _one_contig_bam(str(tmp_path / "solo.bam"))
    pooled = [bams[0], solo]

    whole = pileup_pkg.pileup_alignments(
        pooled, fasta, str(tmp_path / "w"), unstranded=True, contig_groups=1
    )
    split = pileup_pkg.pileup_alignments(
        pooled, fasta, str(tmp_path / "s"), unstranded=True, contig_groups=2
    )

    assert _digest(whole[0]) == _digest(split[0])
    # chr2 exists only in the first alignment; a naive intersection
    # across the pool would have discarded it.
    assert "chr2.parquet" in _digest(split[0])


def test_bam_contigs_matches_the_header_pysam_reads(bams):
    """GIVEN a BAM
    WHEN its header is read with the stdlib rather than pysam
    THEN the reference sequences and lengths agree."""

    with pysam.AlignmentFile(bams[0], "rb") as handle:
        expected = dict(zip(handle.references, handle.lengths))

    assert alignments.bam_contigs(bams[0]) == expected


def test_bam_contigs_declines_anything_that_is_not_a_bam(tmp_path):
    """GIVEN a SAM path
    WHEN its contigs are asked for
    THEN None comes back, because SAM carries no index to restrict."""

    assert alignments.bam_contigs(str(tmp_path / "reads.sam")) is None


def test_bam_contigs_declines_a_file_it_cannot_read(tmp_path):
    """GIVEN a path that is missing, and one that is not really a BAM
    WHEN their contigs are asked for
    THEN None comes back rather than an exception.

    Reading the header is an optimisation, not a validation step. The
    engine opens the same file moments later and reports the problem
    better than a gzip traceback would, and a header that cannot be
    read only means the pileup is not split."""

    junk = tmp_path / "junk.bam"
    junk.write_bytes(b"not actually a bam")

    assert alignments.bam_contigs(str(tmp_path / "missing.bam")) is None
    assert alignments.bam_contigs(str(junk)) is None


###
# Contig naming and the BAM index
#
# Both of these were found against a corpus of deliberately defective
# files rather than against the synthetic BAMs above, which are all
# indexed and all named to match their reference. Each failure used to
# be silent, and the tests here are about the noise.
###


@pytest.mark.parametrize("groups", [1, 4])
def test_a_reference_sharing_no_contig_with_the_alignment_is_refused(
    fasta, tmp_path, groups
):
    """GIVEN an Ensembl-headered alignment and a UCSC-named reference
    WHEN a pileup is requested
    THEN it raises rather than writing an empty table.

    Every record would be dropped by the join against the chrom sizes,
    leaving a table with no rows that `require_complete` passes, since
    a manifest listing no files is one whose files all exist. Training
    then reads a genome with no signal in it. Refused at one batch as
    well as four, because the empty table predates grouping."""

    solo = _one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")
    name = str(tmp_path / "out")

    with pytest.raises(ValueError, match="would write an empty table"):
        pileup_pkg.pileup_alignments(
            [solo], fasta, name, unstranded=True, contig_groups=groups
        )

    assert not os.path.exists(name + tables.CUT_SITE_SUFFIX)


def test_the_refusal_names_the_alignment_and_both_conventions(fasta, tmp_path):
    """GIVEN a contig naming mismatch
    WHEN the pileup refuses
    THEN the message names the file and a contig from either side.

    The cause is almost never a wrong genome, it is a convention, and
    the two are told apart by seeing `1` beside `chr1`."""

    solo = _one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")

    with pytest.raises(ValueError) as error:
        pileup_pkg.pileup_alignments(
            [solo], fasta, str(tmp_path / "out"), unstranded=True
        )

    assert "ensembl.bam" in str(error.value)
    assert "chr1" in str(error.value)


def test_a_pooled_alignment_sharing_no_contig_is_refused(bams, fasta, tmp_path):
    """GIVEN a channel pooling a matching alignment with a mismatched one
    WHEN a pileup is requested
    THEN it raises rather than counting only the matching one.

    This is the quiet case the check exists for. The other branch keeps
    the batches non-empty, so nothing looks wrong: the run succeeds, the
    table verifies, and one alignment's records are simply absent."""

    solo = _one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")

    with pytest.raises(ValueError, match="ensembl.bam"):
        pileup_pkg.pileup_alignments(
            [bams[0], solo],
            fasta,
            str(tmp_path / "out"),
            unstranded=True,
            contig_groups=2,
        )


def test_an_unreadable_header_is_exempt_from_the_contig_check(fasta, tmp_path):
    """GIVEN an alignment that does not exist
    WHEN a pileup is requested
    THEN the engine reports the open failure, not the contig check.

    Diagnosing an unreadable file belongs to the component that opens
    it. Claiming a naming mismatch for a missing BAM would be worse
    than saying nothing, and it is what pins `bam_contigs` returning
    None rather than raising."""

    with pytest.raises(Exception) as error:
        pileup_pkg.pileup_alignments(
            [str(tmp_path / "missing.bam")],
            fasta,
            str(tmp_path / "out"),
            unstranded=True,
        )

    assert "would write an empty table" not in str(error.value)


def test_splitting_an_unindexed_alignment_warns_and_still_matches(bams, fasta, tmp_path):
    """GIVEN an alignment with no .bai beside it
    WHEN it is piled up in several contig batches
    THEN a warning says what the scans cost, and the table is unchanged.

    Without an index a contig predicate is still an ordinary filter, so
    the rows are identical; it is only no longer free. The warning must
    report that and change nothing else."""

    bare = str(tmp_path / "bare.bam")
    shutil.copy(bams[0], bare)

    indexed = pileup_pkg.pileup_alignments(
        [bams[0]], fasta, str(tmp_path / "i"), unstranded=True, contig_groups=2
    )

    with pytest.warns(UserWarning, match="samtools index"):
        split = pileup_pkg.pileup_alignments(
            [bare], fasta, str(tmp_path / "b"), unstranded=True, contig_groups=2
        )

    assert _digest(indexed[0]) == _digest(split[0])


def test_an_indexed_alignment_splits_without_warning(bams, fasta, tmp_path):
    """GIVEN an alignment with its index beside it
    WHEN it is piled up in several contig batches
    THEN nothing is warned about, because the batches are nearly free."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pileup_pkg.pileup_alignments(
            [bams[0]], fasta, str(tmp_path / "ok"), unstranded=True, contig_groups=2
        )


def test_an_unindexed_alignment_is_quiet_when_it_is_not_split(bams, fasta, tmp_path):
    """GIVEN an alignment with no index
    WHEN it is piled up as a single batch
    THEN nothing is warned about.

    One batch is one scan whether or not an index could have restricted
    it, so there is nothing to report and no reason to make the common
    case noisy."""

    bare = str(tmp_path / "bare.bam")
    shutil.copy(bams[0], bare)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pileup_pkg.pileup_alignments(
            [bare], fasta, str(tmp_path / "one"), unstranded=True, contig_groups=1
        )


def test_the_index_check_accepts_both_htslib_conventions(bams, tmp_path):
    """GIVEN the two index names htslib accepts, and a .csi
    WHEN each is tested
    THEN both .bai spellings count and the .csi does not.

    polars-bio cannot open a CSI-indexed BAM at all, so treating one as
    indexed would promise a pushdown that does not exist."""

    suffixed = str(tmp_path / "suffixed.bam")
    replaced = str(tmp_path / "replaced.bam")
    csi_only = str(tmp_path / "csi.bam")
    for path in (suffixed, replaced, csi_only):
        shutil.copy(bams[0], path)

    shutil.copy(bams[0] + ".bai", suffixed + ".bai")
    shutil.copy(bams[0] + ".bai", replaced[: -len(".bam")] + ".bai")
    shutil.copy(bams[0] + ".bai", csi_only + ".csi")

    assert alignments._has_bam_index(suffixed)
    assert alignments._has_bam_index(replaced)
    assert not alignments._has_bam_index(csi_only)
