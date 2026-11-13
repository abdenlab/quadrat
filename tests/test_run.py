"""Tests for running many pileups at once, over either of two backends.

Relocated from the ported ``tests/test_pileup.py``. The assertions are
the port's evidence and are carried over unchanged; the docstring form,
the names, the phase comments and the seams are what changed.

Two seams rather than the private functions the ported tests reached
for. The machine is pinned at the OS boundary by the ``machine``
fixture, and what the fan-out resolved is read where it crosses into
the process pool, by the ``dispatched`` fixture below -- `joblib` is a
third-party library that spawns OS processes, which is the kind of
boundary §11 of the test guide permits standing in for. Reading the
arguments that cross it names no private symbol, where the ported tests
patched ``run._run_process``, ``run._run_one`` and ``run.plan_pileup``.

Only the tests that run a real pileup need the ``pileup`` extra, and
they take it through the ``corpus`` fixture. The guards, the refusals
and the planning tests run on a base install.
"""

import collections
import logging
import os
import pathlib
import re
import shutil
import types

import joblib
import pytest

from quadrat import Aggregate, tables
from quadrat.pileup import plan, run
from tests import TWO_CONTIGS, budget, builders, digest_track


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Four small alignments over two contigs, with an indexed reference."""

    pytest.importorskip("polars_bio")
    return corpus_factory("run", TWO_CONTIGS, n_bams=4, index_fasta=True)


@pytest.fixture(scope="module")
def paired(tmp_path_factory, pysam):
    """One indexed alignment of forward-reverse mate pairs over two contigs.

    Each mate carries the pair's template length, positive on the
    forward one, so a midpoint pileup over it counts every pair.
    """

    header = builders.alignment_header(TWO_CONTIGS)
    reads = []
    for i, (chrom, left) in enumerate([("chr1", 100), ("chr1", 400), ("chr2", 50)]):
        forward = builders.aligned_read(
            header, "p{}".format(i), chrom, left, "20M", 0x41
        )
        reverse = builders.aligned_read(
            header, "p{}".format(i), chrom, left + 80, "20M", 0x91
        )
        forward.template_length, reverse.template_length = 100, -100
        reads += [forward, reverse]
    root = tmp_path_factory.mktemp("run-paired")
    return builders.write_bam(str(root / "paired.bam"), reads, header)


@pytest.fixture
def dispatched(mocker):
    """Record what reaches the process pool, without starting one.

    `joblib.delayed` builds a ``(callable, args, kwargs)`` triple, and
    the job dict the fan-out resolved is the whole of ``args``: a
    runner is ``map``-shaped, so everything the fan-out decided for a
    job -- its contig batches, its partition count -- is in the job
    rather than passed alongside it.
    """

    record = {"n_jobs": None, "backend": None, "jobs": []}

    class Recorder:
        def __init__(self, n_jobs=None, backend=None, **rest):
            record["n_jobs"] = n_jobs
            record["backend"] = backend

        def __call__(self, tasks):
            for _, args, _ in tasks:
                record["jobs"].append(args[0])
            return [[job["name"]] for job in record["jobs"]]

    mocker.patch.object(joblib, "Parallel", Recorder)
    return record


def jobs_for(corpus, out, prefix, unstranded=True):
    """One job per alignment in ``corpus``, writing under ``out``."""

    return [
        {
            "alignments": [bam],
            "sequences": corpus.fasta,
            "name": os.path.join(str(out), "{}{}".format(prefix, i)),
            "aggregate": Aggregate(start_shift=4, end_shift=-4, unstranded=unstranded),
        }
        for i, bam in enumerate(corpus.bams)
    ]


def plain_jobs(corpus, out, names):
    """One job per name, carrying only the three required keys.

    Each job gets its own copy of an alignment, so a test that patches
    `os.path.getsize` by path reaches exactly the jobs it means to --
    sharing one file between a small job and a large one would size
    both of them alike.
    """

    jobs = []
    for i, name in enumerate(names):
        alignment = os.path.join(str(out), "{}.bam".format(name))
        shutil.copy(corpus.bams[i % len(corpus.bams)], alignment)
        jobs.append(
            {
                "name": os.path.join(str(out), name),
                "alignments": [alignment],
                "sequences": corpus.fasta,
            }
        )
    return jobs


def numbers(text):
    """The integers in one line of progress output, in order."""

    return [int(found) for found in re.findall(r"\d+", text)]


def batches_for(input_bytes, workers, ram_bytes, cores):
    """Fewest batches that keep ``workers`` of this job inside the budget.

    `memory_floor` is the function the fan-out itself calls for a
    caller-named worker count, and it is named in ``plan.__all__`` for
    reads like this one. `plan_pileup` is the wrong oracle for that
    path: it maximises the worker count rather than taking one, so it
    answers a different question and agreeing with it is a
    coincidence.

    What this helper still contributes is the budget, restated from the
    published constants rather than read from the machine the suite
    happens to run on -- so a test pins a figure the model implies
    instead of one the solver computed. Restating the *search* as well
    is what made a wrong batch-count bound wrong identically in the
    code and in the oracle, where no test could fail on it.
    """

    free = budget(plan, ram_bytes, cores)
    return plan.memory_floor(input_bytes, workers, free).groups


def test_pileup_many_should_raise_when_the_backend_is_unknown():
    """Test the refusal of a backend that does not exist.

    Given:
        A backend name that is not one of the two, and a job list that
        is also malformed.
    When:
        The pileups are run.
    Then:
        It should name the backends that do exist, before the job list
        is inspected at all.
    """

    # Act & assert
    with pytest.raises(ValueError, match="serial, process"):
        run.pileup_many(["not-a-dict"], backend="dask")


@pytest.mark.parametrize("backend", run.BACKENDS)
def test_pileup_many_should_return_empty_when_the_job_list_is_empty(backend):
    """Test the early return for no work.

    Given:
        No jobs, for each backend the package declares.
    When:
        The pileups are run.
    Then:
        It should return nothing without starting a pool.
    """

    # Act & assert
    assert run.pileup_many([], backend=backend) == []


def test_pileup_many_should_dispatch_through_a_runner_of_the_callers_own(
    corpus, tmp_path, machine
):
    """Test the seam a caller with its own executor reaches for.

    Given:
        A job list, a named worker count, and a ``map``-shaped runner
        the caller supplies.
    When:
        The pileups are run.
    Then:
        It should map the jobs through that runner and return the same
        paths the bundled serial runner returns.

    A caller who already has a cluster, a pool or a warm executor could
    otherwise only reach this by forking the API: calling
    `pileup_alignments` itself and re-deriving the plan, the partition
    division and the per-job batch count, none of which the subpackage
    exports. The seam is one line of the code that was already here.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    mapped = []

    def runner(function, items):
        mapped.extend(items)
        return [function(item) for item in items]

    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])
    serial = run.pileup_many(plain_jobs(corpus, tmp_path, ["a", "b"]))
    digests = [digest_track(paths[0]) for paths in serial]

    # Act
    written = run.pileup_many(jobs, runner=runner, workers=2)

    # Assert
    assert len(mapped) == 2
    assert all("contig_groups" in job for job in mapped)
    assert written == serial
    assert [digest_track(paths[0]) for paths in written] == digests


def test_pileup_many_should_refuse_a_backend_and_a_runner_together(corpus, tmp_path):
    """Test the one combination that has no answer.

    Given:
        Both a bundled runner's name and a runner of the caller's own.
    When:
        The pileups are run.
    Then:
        It should refuse, because there is no answer to which of the
        two maps the jobs.
    """

    # Act & assert
    with pytest.raises(ValueError, match="backend or runner, not both"):
        run.pileup_many(
            plain_jobs(corpus, tmp_path, ["a"]), backend="process", runner=map
        )


def test_pileup_many_should_raise_when_a_job_is_not_a_mapping():
    """Test the refusal of a job that is not keyword arguments.

    Given:
        A job list whose first member is a string.
    When:
        The pileups are run.
    Then:
        It should name that job's index and the type it got.
    """

    # Act & assert
    with pytest.raises(TypeError, match="job 0"):
        run.pileup_many(["not-a-dict"], backend="serial")


@pytest.mark.parametrize("wrap", [types.MappingProxyType, collections.ChainMap])
def test_pileup_many_should_accept_any_mapping_as_a_job(
    corpus, tmp_path, dispatched, wrap
):
    """Test the job shape the published contract names.

    Given:
        A job list whose members are mappings that are not dicts.
    When:
        The pileups are run.
    Then:
        It should run them, because the contract says one mapping per
        pileup.

    A read-only proxy is a natural thing to hand a fan-out that should
    not mutate the caller's state, which is the concern the copy taken
    here exists to address in the first place. The guard tested for
    `dict` and refused all three with a `TypeError` written against
    the documentation it contradicted.
    """

    # Arrange
    jobs = [wrap(job) for job in plain_jobs(corpus, tmp_path, ["a", "b"])]

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    assert len(dispatched["jobs"]) == 2
    assert all(isinstance(job, dict) for job in dispatched["jobs"])


def test_pileup_many_should_accept_one_alignment_given_as_a_bare_path(
    corpus, tmp_path, dispatched
):
    """Test the single-alignment job written without a list.

    Given:
        A job whose ``alignments`` is one path rather than a list of
        one.
    When:
        The pileups are run.
    Then:
        It should run that one alignment.

    The bare string passed the truthiness check and was then iterated
    character-wise: `os.path.getsize` was handed ``"x"``, every size
    came back unsizable, and the *whole* fan-out's plan collapsed to
    the unknown-input path before the job failed inside a worker with a
    message about a single letter.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a"])
    jobs[0]["alignments"] = jobs[0]["alignments"][0]

    bare = jobs[0]["alignments"]

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    assert dispatched["jobs"][0]["alignments"] == [bare]


def test_pileup_many_should_normalize_bytes_paths_in_a_job(corpus, tmp_path, dispatched):
    """Test that a job spelled in bytes reaches a worker as strings.

    Given:
        A job whose ``alignments``, ``sequences`` and ``name`` are all
        ``bytes`` paths.
    When:
        The pileups are run.
    Then:
        Every one of the three should reach the worker as the
        equivalent ``str``.

    Normalizing in the worker is too late for all three. ``name`` keys
    the collision check here, so a ``bytes`` spelling and an equal
    ``str`` were distinct keys and two jobs writing one table went
    undetected; ``alignments`` is sized here by `os.path.getsize`, so
    the plan and the pileup could be made from different figures; and
    ``sequences`` reaches the manifest record through
    `os.path.abspath`, which preserves ``bytes``, so the write failed
    inside the worker with the tables already on disk.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a"])
    expected = dict(jobs[0])
    jobs[0]["alignments"] = [os.fsencode(jobs[0]["alignments"][0])]
    jobs[0]["sequences"] = os.fsencode(str(jobs[0]["sequences"]))
    jobs[0]["name"] = os.fsencode(jobs[0]["name"])

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    job = dispatched["jobs"][0]
    assert job["alignments"] == expected["alignments"]
    assert job["sequences"] == str(expected["sequences"])
    assert job["name"] == expected["name"]


def test_pileup_many_should_refuse_a_job_name_that_is_not_a_path(
    corpus, tmp_path, dispatched
):
    """Test that a wrong-typed name is refused before any job is sent.

    Given:
        A job whose ``name`` is an integer.
    When:
        The pileups are run.
    Then:
        It should raise `TypeError` naming the job by index, and
        dispatch nothing.

    The message is matched by index rather than by the word ``name``,
    because `pileup_alignments` refuses the same argument at its own
    boundary and a looser assertion would be satisfied by that one --
    which is one refusal too late for a fan-out. `_normalize` exists to
    turn a failure a worker would hit into a refusal the caller gets,
    and the name is the argument where that matters most: it keys the
    collision check, so an unnormalized one makes two jobs writing one
    table undetectable, and the later writer reaps the earlier one's
    staging sibling mid-write.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a"])
    jobs[0]["name"] = 42

    # Act & assert
    with pytest.raises(TypeError, match="job 0"):
        run.pileup_many(jobs, backend="process")
    assert dispatched["jobs"] == []


def test_pileup_many_should_detect_two_jobs_writing_one_table_as_path_and_string(
    corpus, tmp_path
):
    """Test the collision check against two spellings of one name.

    Given:
        Two jobs whose ``name`` is the same path, one as a string and
        one as a `pathlib.Path`.
    When:
        The pileups are run.
    Then:
        It should refuse, naming both jobs.

    The check keys on the name, so the two spellings were distinct keys
    and the collision went undetected -- and the loser of that race
    does not merely lose a rename: the later writer reaps every staging
    sibling of the path, destroying the earlier table mid-write.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a", "a"])
    jobs[1]["name"] = pathlib.Path(jobs[1]["name"])

    # Act & assert
    with pytest.raises(ValueError, match="both write"):
        run.pileup_many(jobs, backend="serial")


@pytest.mark.parametrize("key", ["alignments", "sequences", "name"])
def test_pileup_many_should_raise_when_a_job_omits_a_required_key(corpus, tmp_path, key):
    """Test the refusal of a job missing an argument it must carry.

    Given:
        A job list in which one job omits one of the three required
        keys.
    When:
        The pileups are run.
    Then:
        It should fail naming that job's index, before dispatching.

    Validating up front matters most for the concurrent backends: a
    `KeyError` raised inside a worker after fifteen minutes of other
    jobs is a far worse way to learn the job list was wrong.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "x")
    del jobs[2][key]

    # Act & assert
    with pytest.raises(KeyError, match="job 2"):
        run.pileup_many(jobs, backend="process", workers=2)

    assert not os.listdir(tmp_path)


@pytest.mark.parametrize("key", ["alignments", "sequences", "name"])
def test_pileup_many_should_raise_when_a_required_key_is_none(corpus, tmp_path, key):
    """Test the refusal of a required key spelled out as ``None``.

    Given:
        A job list in which one job sets one of the three required keys
        to ``None``.
    When:
        The pileups are run.
    Then:
        It should say that key is set to ``None``, distinctly from the
        error for a key that is simply absent.

    A missing key and a key whose value is meaningless are different
    mistakes, and the second is the one a caller building jobs
    programmatically makes -- a lookup that returned nothing, written
    straight into the job.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "x")
    jobs[1][key] = None

    # Act & assert
    with pytest.raises(ValueError, match="job 1 has '{}' set to None".format(key)):
        run.pileup_many(jobs, backend="process", workers=2)

    assert not os.listdir(tmp_path)


def test_pileup_many_should_raise_when_a_job_names_an_unknown_keyword(corpus, tmp_path):
    """Test the refusal of a misspelled keyword.

    Given:
        A job carrying a keyword `pileup_alignments` does not take.
    When:
        The pileups are run.
    Then:
        It should name the job index and the key, before dispatching.

    A key the pileup does not take is a `TypeError` wherever it lands.
    Landing it here names the job index and costs nothing; landing it in
    a loky child costs however many jobs ran first.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "typo")
    jobs[1]["pos_shfit"] = 4

    # Act & assert
    with pytest.raises(TypeError, match="job 1.*'pos_shfit'"):
        run.pileup_many(jobs, backend="process", workers=2)

    assert not os.listdir(tmp_path)


def test_pileup_many_should_raise_when_a_job_names_unknown_keywords(corpus, tmp_path):
    """Test that two bad keywords are reported as two.

    Given:
        A job carrying two keyword names `pileup_alignments` does not
        take.
    When:
        The pileups are run.
    Then:
        It should name both of them in the plural, so a caller fixing
        one does not have to run again to find the other.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "typos")
    jobs[0]["pos_shfit"] = 4
    jobs[0]["neg_shfit"] = -4

    # Act & assert
    with pytest.raises(TypeError, match="keywords.*'neg_shfit'.*'pos_shfit'"):
        run.pileup_many(jobs, backend="serial")


def test_pileup_many_should_raise_when_two_jobs_share_a_name(corpus, tmp_path):
    """Test the refusal of two jobs writing one table.

    Given:
        Two jobs in a list sharing one output name.
    When:
        The pileups are run.
    Then:
        It should name both job indices and refuse.

    The race is not a lost rename. A writer sweeps every staging sibling
    of its own path before it starts, so the second job to begin deletes
    the first one's staging directory from under it and the first fails
    partway through. Under the process backend the two are concurrent by
    construction.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "clash")
    jobs[3]["name"] = jobs[1]["name"]

    # Act & assert
    with pytest.raises(ValueError, match="jobs 1 and 3"):
        run.pileup_many(jobs, backend="process", workers=4)

    assert not os.listdir(tmp_path)


def test_pileup_many_should_raise_when_two_names_write_one_channel(corpus, tmp_path):
    """Test the refusal of two jobs whose names differ and whose tables do not.

    Given:
        A stranded job named ``a`` and an unstranded one named ``a.+``,
        both of which write ``a.+.track.parquet``.
    When:
        The pileups are run.
    Then:
        It should name both job indices and the table they share, and
        refuse before writing anything.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "a")[:2]
    jobs[0]["aggregate"] = Aggregate()
    jobs[0]["name"] = str(tmp_path / "a")
    jobs[1]["name"] = str(tmp_path / "a.+")

    # Act & assert
    with pytest.raises(ValueError, match=r"jobs 0 and 1 both write .*a\.\+\.track"):
        run.pileup_many(jobs, backend="serial")
    assert not os.listdir(tmp_path)


def test_pileup_many_should_raise_when_a_job_names_no_alignments(corpus, tmp_path):
    """Test the refusal of an empty alignment pool.

    Given:
        A job whose alignment list is empty.
    When:
        The pileups are run.
    Then:
        It should refuse rather than write a table with nothing in it.

    An empty pool reaches none of the guards: no alignment means no
    header to check against the reference, no batch to pile up, and a
    manifest listing no files, which `require_complete` passes because
    every file it lists does exist. Training reads the result as a
    genome with no signal.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "empty")
    jobs[2]["alignments"] = []

    # Act & assert
    with pytest.raises(ValueError, match="job 2 names no alignments"):
        run.pileup_many(jobs, backend="serial")

    assert not os.listdir(tmp_path)


def test_pileup_many_should_start_no_pool_when_a_job_is_malformed(
    corpus, tmp_path, dispatched
):
    """Test that a bad job list is refused before any process starts.

    Given:
        A job list bound for the process backend, one of whose members
        omits a required key.
    When:
        The pileups are run.
    Then:
        It should raise without ever constructing the pool, which is
        stronger than writing no table: a pool that started and found
        nothing to do has already paid for however many workers.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "nopool")
    del jobs[1]["sequences"]

    # Act & assert
    with pytest.raises(KeyError):
        run.pileup_many(jobs, backend="process", workers=2)

    assert dispatched["n_jobs"] is None
    assert dispatched["jobs"] == []


def test_pileup_many_should_leave_the_callers_job_dicts_unchanged(
    corpus, tmp_path, dispatched
):
    """Test that the fan-out does not write through to its argument.

    Given:
        A valid job list bound for the process backend.
    When:
        The fan-out fills in the defaults it computes.
    Then:
        It should leave the caller's own dicts exactly as they were.

    `_normalize` copies for this reason, and nothing pinned it. A caller
    who reuses a job list for a second run would otherwise find the
    first run's resolved batch count already in it, and silently get a
    plan fitted to the wrong concurrency.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "pure")
    before = [dict(job) for job in jobs]

    # Act
    run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    assert jobs == before
    assert "contig_groups" not in jobs[0]


def test_pileup_many_should_clamp_the_pool_to_the_job_count_when_workers_exceeds_it(
    corpus, tmp_path, dispatched
):
    """Test the upper clamp on an explicit worker count.

    Given:
        Two jobs and a worker count far larger than that.
    When:
        The pileups are run on the process backend.
    Then:
        It should size the pool to the job count, there being nothing
        for the rest of the workers to do.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])

    # Act
    run.pileup_many(jobs, backend="process", workers=99)

    # Assert
    assert dispatched["n_jobs"] == 2


def test_pileup_many_should_clamp_the_pool_to_one_worker_when_workers_is_zero(
    corpus, tmp_path, dispatched
):
    """Test the lower clamp on an explicit worker count.

    Given:
        Two jobs and a worker count of zero.
    When:
        The pileups are run on the process backend.
    Then:
        It should size the pool to one worker rather than none, since a
        pool of none would never run the jobs it was given.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])

    # Act
    run.pileup_many(jobs, backend="process", workers=0)

    # Assert
    assert dispatched["n_jobs"] == 1


def test_pileup_many_should_size_the_contig_batches_for_the_named_worker_count(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that naming a worker count re-plans the batches for it.

    Given:
        One job set run twice on a 64-core, 64 GB machine with 16 GB
        alignments, once letting the fan-out choose the worker count and
        once naming a much smaller one.
    When:
        The batches each run resolved are compared.
    Then:
        It should buy fewer batches for the smaller pool, because the
        peak has to fit the concurrency the caller asked for and no
        more.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    size = 16 * 1024**3
    sized(size)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    run.pileup_many(jobs, backend="process")
    chosen = {job["contig_groups"] for job in dispatched["jobs"]}
    dispatched["jobs"].clear()
    run.pileup_many(jobs, backend="process", workers=1)
    named = {job["contig_groups"] for job in dispatched["jobs"]}

    # Assert
    assert chosen == {plan.plan_pileup(4, size)[1]}
    assert named == {batches_for(size, 1, 64 * 1024**3, 64)}
    assert max(named) < max(chosen)


def test_pileup_many_should_forward_target_partitions_when_the_caller_names_it(
    corpus, tmp_path, dispatched
):
    """Test that an explicit partition count reaches every job.

    Given:
        A job list and a ``target_partitions`` the caller named.
    When:
        The pileups are dispatched.
    Then:
        It should hand that exact count to every job, rather than the
        count it would have computed for the pool.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])

    # Act
    run.pileup_many(jobs, backend="process", workers=2, target_partitions=5)

    # Assert
    assert {job["target_partitions"] for job in dispatched["jobs"]} == {5}


def test_pileup_many_should_divide_the_partitions_among_workers_when_omitted(
    corpus, tmp_path, dispatched, machine
):
    """Test the partition count the fan-out computes for itself.

    Given:
        A job list with ``target_partitions`` left out, an explicit
        worker count, and a machine pinned to a known core count.
    When:
        The pileups are dispatched.
    Then:
        It should divide the cores among the workers, which is what
        keeps K concurrent pileups from each claiming the whole
        process-global thread pool.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    run.pileup_many(jobs, backend="process", workers=4)

    # Assert
    assert {job["target_partitions"] for job in dispatched["jobs"]} == {
        plan.default_partitions(4)
    }


def test_pileup_many_should_plan_as_though_absent_when_a_defaulted_key_is_none(
    corpus, tmp_path, dispatched, machine
):
    """Test that a documented ``None`` default means what omission means.

    Given:
        Two job lists, one omitting ``target_partitions`` and
        ``contig_groups`` and one spelling both out as ``None``.
    When:
        The pileups are dispatched.
    Then:
        It should resolve both to the same values, and to the partition
        count computed for the concurrency.

    ``None`` is what `pileup_alignments` documents as the default of
    both, and ``jobs`` is documented as its keyword arguments, so
    writing the default out loud must mean what omitting it means. Left
    in place it survives `setdefault`, reaches the pileup, and resolves
    to every core on the machine -- for every worker at once, which is
    the K-fold oversubscription dividing the partitions exists to
    prevent.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act
    run.pileup_many(jobs_for(corpus, tmp_path, "absent"), backend="process", workers=4)
    absent = [
        (job["target_partitions"], job["contig_groups"]) for job in dispatched["jobs"]
    ]
    dispatched["jobs"].clear()
    spelled = [
        dict(job, target_partitions=None, contig_groups=None)
        for job in jobs_for(corpus, tmp_path, "spelled")
    ]
    run.pileup_many(spelled, backend="process", workers=4)

    # Assert
    assert [
        (job["target_partitions"], job["contig_groups"]) for job in dispatched["jobs"]
    ] == absent
    assert {partitions for partitions, _ in absent} == {plan.default_partitions(4)}


def test_pileup_many_should_split_only_the_large_jobs_when_the_job_set_is_lopsided(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that each job is raised to what its own bytes ask for.

    Given:
        A job set whose mean size is small, two members of 24 GB, and
        one of those pinning its own batch count.
    When:
        The fan-out is planned.
    Then:
        It should split the large job, leave the small ones whole, and
        keep the pinned count.

    The planner is handed the mean, which on a set like this is small:
    the rung-3 benchmark set averaged 3.28 GB across 68 alignments and
    held one of 21.2 GB, and it was that one that died. A cap read from
    the mean alone would have missed it.
    """

    # Arrange
    # js031, where the overflow was found: 256 cores and 1007 GB, which
    # is enough memory that the planner never had to split anything.
    machine(256, 1007 * 1024**3)
    names = ["s0", "s1", "s2", "s3", "s4", "huge", "pinned"]
    jobs = plain_jobs(corpus, tmp_path, names)
    jobs[-1]["contig_groups"] = 1
    large = {job["alignments"][0] for job in jobs[-2:]}

    sized({path: 24 * 1024**3 for path in large})

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    groups = {
        os.path.basename(job["name"]): job["contig_groups"] for job in dispatched["jobs"]
    }
    assert groups["huge"] == 3, "the 24 GB job must be split"
    assert all(groups["s{}".format(i)] == 1 for i in range(5)), (
        "a small job must not be split for its neighbour's sake"
    )
    assert groups["pinned"] == 1, "an explicit count must survive the cap"


def test_pileup_many_should_not_split_a_small_job_for_the_means_sake(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that the mean's scan floor is not imposed on a small job.

    Given:
        One 80 GiB alignment beside one of 7 GiB, so the pair's mean of
        43.5 GiB is itself over the scan cap, on a machine with enough
        memory that neither job is memory-bound.
    When:
        The fan-out is planned at one worker.
    Then:
        The large job should be split to the batch ceiling and the
        small one should run in a single batch, which is what its own
        bytes ask for.

    This is the companion to the lopsided test above, on the other side
    of the cap. There the mean stays under `MAX_GROUP_INPUT_BYTES`, so
    the mean's own scan floor is one batch and taking the larger of the
    two answers changes nothing; here the mean's floor is six, and the
    7 GiB job was being dispatched at six genome-wide aggregates where
    one is required. Criterion #3's second clause is exactly this case
    -- "one large alignment among many small ones is split *and the
    small ones are not*" -- and no test reached the regime it names.

    The memory axis is deliberately slack. It is the only way to see
    the scan term on its own: on a machine where the budget binds,
    every count below the mean's floor is refused for memory anyway and
    a floor taken from the wrong figure is unobservable.
    """

    # Arrange
    machine(8, 512 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["small", "huge"])
    sized(
        {
            jobs[0]["alignments"][0]: 7 * 1024**3,
            jobs[1]["alignments"][0]: 80 * 1024**3,
        }
    )

    # Act
    run.pileup_many(jobs, backend="process", workers=1)

    # Assert
    groups = {
        os.path.basename(job["name"]): job["contig_groups"] for job in dispatched["jobs"]
    }
    assert groups["huge"] == plan.MAX_CONTIG_GROUPS, "the 80 GiB job must be split"
    assert groups["small"] == 1, (
        "a job under the scan cap must not be split because its "
        "neighbour pulled the mean over it"
    )


def test_pileup_many_should_lower_a_concurrency_it_chose_itself(
    corpus, tmp_path, dispatched, machine, sized, recwarn
):
    """Test that an auto-planned worker count is clamped, not warned about.

    Given:
        A 64-core, 64 GB machine and a hundred jobs of 100 MB beside
        one of 7.9 GB, which the mean hides.
    When:
        The fan-out is planned with no worker count.
    Then:
        It should dispatch at a concurrency that brings every job
        inside the budget, and emit no over-budget warning.

    The worked example is the module's own: 13.85 GB projected for the
    7.9 GB job, a worst concurrent total of 50.1 GB, and a 39.2 GB
    budget -- so the run went ahead over budget and said so, where the
    code held every job's exact bytes and had chosen the concurrency
    itself.

    `A8` declined to lower `workers` as "a behavior change the issue
    did not ask for", which is right for a caller who named the count
    and is the opposite of right for one the planner inferred. This
    pins the inferred half only; the named half still warns and is
    tested above.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    names = ["s{}".format(index) for index in range(100)] + ["big"]
    jobs = plain_jobs(corpus, tmp_path, names)
    sizes = {job["alignments"][0]: 100 * 1024**2 for job in jobs[:-1]}
    sizes[jobs[-1]["alignments"][0]] = 7900 * 1024**2
    sized(sizes)

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    free = budget(plan, 64 * 1024**3, 64)
    workers = dispatched["n_jobs"]
    for job in dispatched["jobs"]:
        peak = plan.projected_peak(
            100 * 1024**2 if os.path.basename(job["name"]) != "big" else 7900 * 1024**2,
            job["contig_groups"],
        )
        assert workers * peak <= free, os.path.basename(job["name"])
    assert not [
        warning
        for warning in recwarn
        if "does not fit the memory budget" in str(warning.message)
    ]


def test_pileup_many_should_size_the_fan_out_from_the_files_on_disk(
    corpus, tmp_path, dispatched, machine, sized, caplog
):
    """Test that the worker count is fitted to the input bytes.

    Given:
        A job list whose alignments are on disk, on a pinned machine.
    When:
        The fan-out is planned without a worker count.
    Then:
        It should report the worker count the planner gives for those
        bytes, rather than one fitted to no data at all.

    The sizes are available at the call site, so a caller who does not
    name a worker count should still get a bound fitted to their data.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    sized(4 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    with caplog.at_level(logging.INFO, logger="quadrat.pileup"):
        run.pileup_many(jobs, backend="process")

    # Assert
    expected, _ = plan.plan_pileup(4, 4 * 1024**3)
    assert dispatched["n_jobs"] == expected
    assert numbers(" ".join(caplog.messages))[1] == expected


def test_pileup_many_should_plan_without_sizes_when_the_alignments_cannot_be_sized(
    corpus, tmp_path, dispatched, machine
):
    """Test the fallback where no input can be measured.

    Given:
        A job list naming alignments that are not on disk.
    When:
        The fan-out is planned.
    Then:
        It should still plan a pool, falling back to the measured
        unknown-input figure rather than raising or guessing a size.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    jobs = [
        {
            "name": os.path.join(str(tmp_path), name),
            "alignments": [str(tmp_path / "absent.bam")],
            "sequences": corpus.fasta,
        }
        for name in ("a", "b")
    ]

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    assert dispatched["n_jobs"] == plan.plan_pileup(2, None)[0]


def test_pileup_many_should_report_a_serial_run(corpus, tmp_path, caplog, mocker):
    """Test the progress record for a serial run.

    Given:
        A job list on the serial runner.
    When:
        The pileups are run.
    Then:
        It should report how many datasets are being piled up, and say
        that the run is serial, there being no worker or partition count
        to report.

    Emitted as a record whatever the caller has configured, which is
    what makes routing it the caller's choice rather than this
    package's -- `pileup_alignments` still carries the ``verbose``
    convenience for someone who has configured nothing.
    """

    # Arrange
    mocker.patch.object(run, "_run_one", lambda job: [job["name"]])
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c"])

    # Act
    with caplog.at_level(logging.INFO, logger="quadrat.pileup"):
        run.pileup_many(jobs, backend="serial")

    # Assert
    reported = " ".join(caplog.messages)
    assert numbers(reported) == [3]
    assert "serially" in reported


def test_pileup_many_should_report_the_resolved_plan(
    corpus, tmp_path, dispatched, machine, caplog
):
    """Test the progress line for a process run.

    Given:
        A job list on the process runner with a named worker count.
    When:
        The pileups are run.
    Then:
        It should report the dataset, worker, partition and contig-group
        counts it resolved, those four numbers being the only public
        account of what the fan-out decided.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    with caplog.at_level(logging.INFO, logger="quadrat.pileup"):
        run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    reported = " ".join(caplog.messages)
    assert numbers(reported) == [4, 2, plan.default_partitions(2), 1]
    assert "process worker(s)" in reported


def test_pileup_many_should_report_a_contig_group_range_when_the_jobs_differ(
    corpus, tmp_path, dispatched, machine, sized, caplog
):
    """Test the progress line where the jobs resolve to different batches.

    Given:
        Six jobs of which one alignment is far larger than the rest, so
        the batch counts differ.
    When:
        The pileups are run.
    Then:
        It should report the batch count as a low-high range rather than
        a single number that is true of no job.
    """

    # Arrange
    machine(256, 1007 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["s0", "s1", "s2", "s3", "s4", "huge"])
    huge = jobs[-1]["alignments"][0]
    sized({huge: 24 * 1024**3})

    # Act
    with caplog.at_level(logging.INFO, logger="quadrat.pileup"):
        run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    assert "1-3 contig group(s)" in " ".join(caplog.messages)


def test_pileup_many_should_write_byte_identical_tables_across_backends(
    corpus, tmp_path
):
    """Test that a backend is a scheduling choice and not a numerical one.

    Given:
        The same jobs run serially and across a process pool.
    When:
        The written tables are hashed.
    Then:
        It should write byte-for-byte identical Parquet files, and
        return the same paths in the same order.

    This is the whole contract. It holds exactly because the aggregate
    is an integer ``COUNT(*)`` scaled once after grouping, so no
    partition order can accumulate floats differently.
    """

    # Arrange
    a, b = tmp_path / "serial", tmp_path / "process"
    a.mkdir()
    b.mkdir()

    # Act
    serial = run.pileup_many(jobs_for(corpus, a, "x"), backend="serial")
    parallel = run.pileup_many(jobs_for(corpus, b, "x"), backend="process", workers=2)

    # Assert
    assert [os.path.basename(p) for group in serial for p in group] == [
        os.path.basename(p) for group in parallel for p in group
    ]

    for lhs, rhs in zip(serial, parallel):
        for left, right in zip(lhs, rhs):
            assert digest_track(left) == digest_track(right)
            assert digest_track(left), "table wrote no Parquet files"


@pytest.mark.workers
def test_pileup_many_should_carry_an_aggregate_across_the_process_boundary(
    corpus, paired, tmp_path
):
    """Test that the record a job carries is the one its worker counts by.

    Given:
        Jobs carrying aggregates that between them set every field away
        from its default, one per job, stranded and unstranded, one of
        them ``"span"`` and the midpoint one over mate pairs.
    When:
        They are run serially and across a process pool.
    Then:
        Each pair of runs should write byte-identical tables, none of
        them empty, carrying the same manifest signal and the same
        recorded aggregate, so the record crossed into each worker
        whole.
    """

    # Arrange
    aggregates = [
        Aggregate(
            exclude_flags=0x404,
            require_flags=0,
            min_mapq=1,
            position="three_prime",
            start_shift=3,
            end_shift=-2,
            scale_factor=0.25,
        ),
        Aggregate(position="span", unstranded=True, scale_factor=2.0),
        Aggregate(
            position="midpoint",
            unstranded=True,
            min_template_length=1,
            max_template_length=500,
        ),
        Aggregate(require_flags=0x10, unstranded=True),
    ]
    a, b = tmp_path / "serial", tmp_path / "process"
    a.mkdir()
    b.mkdir()

    def jobs(out):
        listed = jobs_for(corpus, out, "x")
        for job, aggregate in zip(listed, aggregates):
            job["aggregate"] = aggregate
        # The corpus carries no template lengths, under which the
        # midpoint table would be empty on both sides.
        listed[2]["alignments"] = [paired]
        return listed

    # Act
    serial = run.pileup_many(jobs(a), backend="serial")
    parallel = run.pileup_many(jobs(b), backend="process", workers=2)

    # Assert
    for aggregate, lhs, rhs in zip(aggregates, serial, parallel):
        assert len(lhs) == len(rhs) == (1 if aggregate.unstranded else 2)
        for left, right in zip(lhs, rhs):
            assert digest_track(left), "table wrote no Parquet files"
            assert digest_track(left) == digest_track(right)
            record = tables.read_manifest(right)
            assert record["signal"] == aggregate.signal
            assert (
                record["parameters"]["aggregate"]
                == (tables.read_manifest(left)["parameters"]["aggregate"])
            )


@pytest.mark.parametrize("aggregate", [None, {"position": "span"}])
def test_pileup_many_should_refuse_a_job_whose_aggregate_is_not_one(
    corpus, tmp_path, dispatched, aggregate
):
    """Test that a job's aggregate is vetted before any worker starts.

    Given:
        A job whose ``aggregate`` is ``None`` or a mapping of the
        record's fields.
    When:
        The fan-out is run.
    Then:
        It should raise `TypeError` naming the job, and dispatch
        nothing.
    """

    # Arrange
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])
    jobs[1]["aggregate"] = aggregate

    # Act & assert
    with pytest.raises(TypeError, match="job 1 has aggregate set to"):
        run.pileup_many(jobs, backend="process", workers=2)
    assert dispatched["jobs"] == []


def test_pileup_many_should_size_each_jobs_batches_by_its_rows_per_record(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that the scan cap reaches the fan-out per job.

    Given:
        Two jobs reading 6 GB each on a machine with memory to spare,
        one counting 5' ends and one writing span coverage.
    When:
        The fan-out is planned.
    Then:
        The 5' job should run in one batch, its bytes under the scan
        cap, and the span job in two, its two rows per record taking
        it over.
    """

    # Arrange
    machine(64, 1024 * 1024**3)
    sized(6 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["points", "span"])
    jobs[1]["aggregate"] = Aggregate(position="span")

    # Act
    run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    assert [job["contig_groups"] for job in dispatched["jobs"]] == [1, 2]


def test_pileup_many_should_leave_every_table_verifiable_after_a_parallel_run(
    corpus, tmp_path
):
    """Test that concurrency leaves every table complete.

    Given:
        Jobs run across a process pool.
    When:
        Each written table is checked for completeness.
    Then:
        It should leave all of them verifiable with a positive row
        count, each pileup committing its own table atomically.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "x")

    # Act
    groups = run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    for group in groups:
        for table in group:
            assert tables.require_complete(table) == table
            assert tables.read_manifest(table)["rows"] > 0


def test_pileup_many_should_return_the_plus_channel_before_the_minus_channel(
    corpus, tmp_path
):
    """Test the order of the channels within one job's result.

    Given:
        Stranded jobs run across a process pool.
    When:
        The returned paths are inspected.
    Then:
        It should give each job its plus table before its minus table.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "s", unstranded=False)

    # Act
    groups = run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    for group in groups:
        assert len(group) == 2
        assert group[0].endswith(".+" + tables.TRACK_SUFFIX)
        assert group[1].endswith(".-" + tables.TRACK_SUFFIX)


def test_pileup_many_should_return_results_in_job_order(corpus, tmp_path):
    """Test that the results follow submission rather than completion.

    Given:
        Jobs whose names make their submission order recoverable, run
        concurrently.
    When:
        The returned groups are inspected.
    Then:
        It should return them in submission order.
    """

    # Arrange
    jobs = jobs_for(corpus, tmp_path, "ordered")

    # Act
    groups = run.pileup_many(jobs, backend="process", workers=3)

    # Assert
    assert [os.path.basename(group[0]) for group in groups] == [
        "ordered{}{}".format(i, tables.TRACK_SUFFIX) for i in range(len(jobs))
    ]


def test_pileup_many_should_clamp_the_named_worker_count_before_sizing_the_batches(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that batches are sized for the pool that runs, not the one asked for.

    Given:
        Two jobs on an 8-core, 64 GB machine, with a named worker count
        of 32 that the job count will clamp to 2.
    When:
        The fan-out is planned.
    Then:
        It should size the batches for 2 workers, which is the
        concurrency that will actually run.

    The clamp used to happen after the planning, so the batch count was
    solved for a concurrency that never existed. Batches are not free --
    four of them cost 25% more pileup wall clock at 56 GB of input --
    so paying for headroom no worker will use is a measurable loss for
    nothing.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    size = 4 * 1024**3
    sized(size)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])

    # Act
    run.pileup_many(jobs, backend="process", workers=32)

    # Assert
    clamped = batches_for(size, 2, 64 * 1024**3, 8)
    assert {job["contig_groups"] for job in dispatched["jobs"]} == {clamped}
    # The unclamped count is what the old ordering planned for, and it
    # has to differ or this passes for the wrong reason.
    assert batches_for(size, 8, 64 * 1024**3, 8) != clamped


def test_pileup_many_should_size_every_alignment_exactly_once(
    corpus, tmp_path, dispatched, machine, mocker
):
    """Test that the corpus is walked once rather than twice.

    Given:
        Four jobs on a pinned machine.
    When:
        The fan-out is planned.
    Then:
        It should stat each alignment exactly once.

    The mean was folded over a walk whose per-job figures were thrown
    away, and the per-job pass below it then walked the same corpus
    again. These stats are the one part of the planning path that
    touches a filesystem, and on the Lustre or NFS mounts these
    corpora live on a cold one is milliseconds -- so a 1,000-job set
    spent tens of seconds reading sizes it already had.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    counted = collections.Counter()
    real = os.path.getsize

    def counting(path):
        counted[str(path)] += 1
        return real(path)

    mocker.patch.object(os.path, "getsize", counting)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    alignments = {job["alignments"][0] for job in jobs}
    assert set(counted) >= {str(path) for path in alignments}
    assert set(counted.values()) == {1}


def test_pileup_many_should_read_the_memory_budget_exactly_once(
    corpus, tmp_path, dispatched, machine, mocker
):
    """Test that one fan-out reads the machine's memory once.

    Given:
        Four jobs on a pinned machine, with the budget read counted.
    When:
        The fan-out is planned with no worker count.
    Then:
        It should read the budget exactly once.

    The sibling of the stat test above, on the other OS boundary. The
    read opens two cgroup paths, reads the SLURM environment, calls
    ``sysconf`` twice and queries the affinity mask, and the comment
    above it claims a single read -- which held for the per-job loop
    and not for `plan_pileup`, whose own call was a second full read.
    Two reads also need not agree, since the budget is mutable OS
    state, and the worker count was solved against one of them while
    every per-job floor was checked against the other.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    reads = []
    real = plan.memory_budget

    def counting():
        reads.append(None)
        return real()

    # Both bindings, because `run` imported the function by name and so
    # holds its own reference: patching only one of the two counts one
    # of the two callers and reads as a single read however many there
    # are.
    mocker.patch.object(run, "memory_budget", counting)
    mocker.patch.object(plan, "memory_budget", counting)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    assert len(reads) == 1


def test_pileup_many_should_warn_when_an_auto_planned_job_cannot_fit(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test the warning left once an auto-planned concurrency is lowered.

    Given:
        An 8-core, 8 GB machine and four jobs, one of 32 GB that fits
        at no batch count while the mean the planner saw does.
    When:
        The fan-out is planned with no worker count.
    Then:
        It should lower the concurrency to one worker, still warn
        naming the job that does not fit, and say that the job needs a
        bigger machine rather than advising a ``workers`` the caller
        never passed.

    The check ran only when the caller named ``workers``, and it was
    evaluated against the mean -- so the lopsided set criterion #3
    exists for was clamped to eight batches and dispatched with nothing
    said. The issue's Motivation is that under-estimating ends a run
    rather than slowing it.

    What is left here is the residual case, and it is why the warning
    survived the clamp: lowering the concurrency cannot rescue a job
    that does not fit at one worker and eight batches. Telling that
    caller to pass ``workers`` was advice they could not act on, since
    the plan had already taken the count as low as it goes.
    """

    # Arrange
    machine(8, 8 * 1024**3)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "huge"])
    huge = jobs[-1]["alignments"][0]
    sized({huge: 32 * 1024**3})

    # Act
    with pytest.warns(UserWarning, match="does not fit the memory budget") as caught:
        run.pileup_many(jobs, backend="process")

    # Assert
    message = str(caught[0].message)
    assert message.startswith("1 worker(s)")
    assert "job 3 at {} batches".format(plan.MAX_CONTIG_GROUPS) in message
    assert "job 0" not in message
    assert "bigger machine" in message
    assert "pass workers outright" not in message
    assert len(dispatched["jobs"]) == 4


def test_pileup_many_should_warn_when_the_named_worker_count_cannot_fit(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that an unfittable concurrency is honoured out loud.

    Given:
        Four jobs of 4 GB on an 8-core, 8 GB machine and a named worker
        count of 4, which no batch count can bring inside the budget.
    When:
        The fan-out is planned.
    Then:
        It should warn, naming the worker count and the batch count it
        settled for, and run anyway.

    Honoured rather than refused because the caller named it; said out
    loud because the batch count has run out of room to buy it back and
    the alternative is learning it as an OOM kill.
    """

    # Arrange
    machine(8, 8 * 1024**3)
    size = 4 * 1024**3
    sized(size)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b", "c", "d"])

    # Act
    with pytest.warns(UserWarning, match="does not fit the memory budget") as caught:
        run.pileup_many(jobs, backend="process", workers=4)

    # Assert
    message = str(caught[0].message)
    assert "4 worker(s) over 4 job(s)" in message
    assert "{} batches".format(plan.MAX_CONTIG_GROUPS) in message
    assert "job 0" in message and "job 3" in message
    assert "Pass a smaller workers" in message
    assert len(dispatched["jobs"]) == 4


def test_pileup_many_should_not_warn_when_the_named_worker_count_fits(
    corpus, tmp_path, dispatched, machine, sized, recwarn
):
    """Test that a fan-out inside its budget says nothing.

    Given:
        Two jobs of 4 GB on an 8-core, 64 GB machine and a named worker
        count of 2.
    When:
        The fan-out is planned.
    Then:
        It should emit no budget warning.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    size = 4 * 1024**3
    sized(size)
    jobs = plain_jobs(corpus, tmp_path, ["a", "b"])

    # Act
    run.pileup_many(jobs, backend="process", workers=2)

    # Assert
    assert not [
        caught
        for caught in recwarn.list
        if "does not fit the memory budget" in str(caught.message)
    ]


def test_pileup_many_should_split_a_job_the_mean_hides_from_the_memory_bound(
    corpus, tmp_path, dispatched, machine, sized
):
    """Test that the per-job raise reaches the memory axis too.

    Given:
        Six jobs on a 64-core, 64 GB machine, five of them tiny and one
        of 7.9 GB -- under the scan cap, so the scan floor asks for
        nothing, and far above the mean the planner saw.
    When:
        The fan-out is planned.
    Then:
        It should split the large job and leave the small ones whole.

    This is criterion #3's lopsided set on the memory axis rather than
    the scan axis. The per-job raise used to consult only the scan cap,
    so a job many times the mean but under 8 GB was planned with the
    mean's peak: 13.85 GB projected for a worst concurrent total of
    50.1 GB against a 39.2 GB budget.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    names = ["s0", "s1", "s2", "s3", "s4", "huge"]
    jobs = plain_jobs(corpus, tmp_path, names)
    large = jobs[-1]["alignments"][0]
    size = 7900 * 1024**2

    sized({large: size})

    # Act
    run.pileup_many(jobs, backend="process")

    # Assert
    groups = {
        os.path.basename(job["name"]): job["contig_groups"] for job in dispatched["jobs"]
    }
    # The scan floor asks for one batch at this size, so anything above
    # one came from the memory bound.
    assert plan.plan_pileup(0, size)[1] == 1
    assert groups["huge"] > 1, "the job above the mean must be split"
    assert all(groups["s{}".format(i)] == 1 for i in range(5)), (
        "a small job must not be split for its neighbour's sake"
    )
