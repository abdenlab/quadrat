"""Running many pileups at once, over either of two backends.

Concurrency is worth roughly fourfold on a large machine and nothing on
a small one, because a pileup is already internally multithreaded past
its own partition count -- one burns about two cores at a single
partition and about five at fourteen, so the machine is largely spoken
for before a second pileup starts. The curve rises steeply to about a
dozen workers and then flattens.

The backends are verified byte-identical rather than merely
equivalent. The aggregate is an exact integer count with the scale
applied once afterwards, so there is no partition-order-dependent float
accumulation to diverge.

A job is a mapping of `quadrat.pileup.pileup_alignments` keyword
arguments, copied into a plain dict here so that it serialises by
value and needs no class definition on the far side of a process
boundary. What this module reads out of one is bounded:
``alignments`` to size it, ``name`` to catch two jobs writing one
table, and ``contig_groups`` and ``target_partitions`` to fill in the
defaults the fan-out computes. Everything else is passed through
untouched.

How the jobs are mapped is the caller's: `pileup_many` takes a
``map``-shaped ``runner`` and bundles two of them under `BACKENDS`,
rather than closing the set of executors a deployment may use. What
the session constrains is what an executor may *be* -- one pileup per
process, started by spawn rather than fork -- and not whether the
executor is the seam.

Two names the issue's Description assigns to this module are
deliberately absent. ``default_workers`` is not ported because
`quadrat.pileup.plan.plan_pileup` answers the worker count as part of
a plan rather than on its own, which is the same question asked once
instead of twice. A distributed backend is not ported because
equivalence is scoped to the two runners below, and a third could not
be verified byte-identical from here without a cluster to verify it
against -- the ``runner`` seam is what a caller who has one reaches
for in the meantime.
"""

import collections.abc
import inspect
import os
import warnings

from quadrat._engine import fspath_or_none, import_on_first_use
from quadrat.pileup.plan import (
    MAX_CONTIG_GROUPS,
    default_partitions,
    memory_budget,
    memory_floor,
    plan_pileup,
    projected_peak,
)
from quadrat.pileup.track import log, pileup_alignments

#: The runners this package bundles, by the name ``backend`` selects
#: them with, both verified byte-identical to each other and to any
#: other runner: what a runner changes is the wall clock and not what
#: is counted. A caller with an executor of its own passes ``runner``
#: instead; see `pileup_many`.
BACKENDS = ("serial", "process")


#: Keys a job dict may carry, read off the signature it forwards to
#: rather than restated here, so adding or removing a parameter of
#: `pileup_alignments` cannot leave this behind.
_JOB_KEYS = frozenset(inspect.signature(pileup_alignments).parameters)


#: Job keys this module fills in a default for. A ``None`` against one
#: of these is treated as absent, because ``None`` is the documented
#: default of the parameter being forwarded -- so a caller who spells
#: the default out loud must get the same plan as one who omits it,
#: rather than a worker that re-plans for the whole machine.
_DEFAULTED_KEYS = ("contig_groups", "target_partitions")


# public
def pileup_many(
    jobs,
    *,
    backend=None,
    runner=None,
    workers=None,
    target_partitions=None,
):
    """Run independent pileups, serially unless a runner says otherwise.

    One pileup runs per process at a time. The polars-bio session is
    process-global and every pileup registers its alignments under the
    same table names, so the ``serial`` backend runs them strictly one
    after another and the ``process`` backend gives each worker a
    process with a session of its own. A caller who wants concurrency
    wants ``backend="process"``, not threads: calling
    `pileup_alignments` from two threads raises.

    Every backend returns the same paths in the same order, and the
    tables they write are byte-identical; only the wall clock differs.
    Exceptions propagate from the first job that raises, and the tables
    of jobs that had already finished are left in place -- each one is
    committed atomically on its own, so a partial run leaves complete
    tables and absent ones, never a half-written table.

    :param jobs: Keyword arguments for `pileup_alignments`, one
        `collections.abc.Mapping` per pileup, each copied before
        anything is filled in. ``alignments``, ``sequences`` and
        ``name`` are required. Jobs must be independent: two jobs
        writing the same ``name`` race, and nothing here detects it.
    :param backend: One of ``"serial"`` or ``"process"``, naming one of
        the two bundled runners. ``None`` with no ``runner`` means
        ``"serial"``.
    :param runner: A ``map``-shaped callable the jobs are dispatched
        through -- ``runner(function, jobs)``, returning the results in
        job order, as `multiprocessing.pool.Pool.map` and
        `concurrent.futures.Executor.map` both are. This is the seam
        for a caller who already has a cluster, a pool or a warm
        executor; pass ``workers`` alongside it to say what its
        concurrency is, so the contig batches are sized for the
        concurrency that will actually run.

        What the runner must satisfy is a precondition rather than a
        preference, and it is the session's and not this function's:
        one pileup per process at a time, in processes started by
        spawn rather than fork. `quadrat.reference` caches a
        polars-bio FASTA registration in a process-global that a
        forked child would inherit as a handle it does not own, and
        two pileups sharing one session read each other's reads --
        see `quadrat.pileup.pileup_alignments`, which refuses the
        second. A thread pool is therefore not a runner.
    :param workers: How many pileups run at once, clamped to the job
        count. ``None`` solves the worker count and the batch count
        together with `plan_pileup`, which a bundled runner then
        honours and a caller's own runner is responsible for honouring.
        A named count is honoured, and the contig batches are then
        solved for that concurrency rather than maximised -- with a
        warning when no batch count in range brings a job inside the
        memory budget, since the count is honoured either way. Ignored
        by the ``serial`` runner, which runs one at a time by
        definition.
    :param target_partitions: Passed to each pileup, overriding what it
        would choose. ``None`` uses `default_partitions` for a
        concurrent runner and leaves the serial one at its
        whole-machine default.
    :returns: Each job's written table paths, in channel order, in job
        order.
    """

    if backend is not None and runner is not None:
        raise ValueError(
            "pileup_many takes backend or runner, not both: backend "
            "names one of the two bundled runners ({}) and runner is "
            "one of your own, so passing both leaves no answer to "
            "which maps the jobs. Got backend={!r}.".format(", ".join(BACKENDS), backend)
        )
    if runner is None:
        backend = "serial" if backend is None else backend
        if backend not in BACKENDS:
            raise ValueError(
                "backend must be one of {}; got {!r}. A runner outside "
                "that set is passed as runner rather than "
                "named here.".format(", ".join(BACKENDS), backend)
            )

    jobs = _normalize(jobs)
    if not jobs:
        return []

    named_workers = workers is not None

    if runner is None and backend == "serial":
        # One at a time, so there is no concurrency to size and nothing
        # to plan. The partition count is the caller's if they named
        # one and the whole machine otherwise, which is right for a
        # pileup that has the machine to itself.
        if target_partitions is not None:
            for job in jobs:
                job.setdefault("target_partitions", target_partitions)
        log.info("Piling up %s dataset(s) serially", len(jobs))
        return [_run_one(job) for job in jobs]

    # Sized once. `_bytes_per_job` folds this list rather than walking
    # the corpus again, and the per-job pass below indexes it: every
    # alignment was otherwise `stat`ed exactly twice, and on the Lustre
    # or NFS mounts these corpora live on a cold `stat` is milliseconds.
    sizes = [_job_bytes(job) for job in jobs]
    bytes_per_job = _bytes_per_job(sizes)
    # Read once for the same reason. `memory_budget` opens two cgroup
    # paths, reads the SLURM environment, calls `sysconf` twice and
    # queries the affinity mask, none of which can change during the
    # planning below.
    budget = memory_budget()
    if workers is None:
        plan = plan_pileup(len(jobs), bytes_per_job, budget)
        workers, groups = plan.workers, plan.contig_groups
        # Then lowered, if the mean hid a job the chosen concurrency
        # cannot afford. The plan solves from `bytes_per_job`, and a
        # set holding one alignment many times the size of the rest is
        # the shape that figure says nothing about -- so this is the
        # one path where the library both knows every job's exact
        # bytes and chose the concurrency itself, and warning about a
        # number it picked would be reporting its own decision back to
        # the caller.
        #
        # Solved directly rather than by searching. A job fits at some
        # batch count exactly when it fits at `MAX_CONTIG_GROUPS`,
        # because the projected peak falls monotonically with the
        # count, so the largest affordable concurrency is the budget
        # over the biggest of those per-job peaks.
        #
        # A named `workers` is left alone deliberately: the count is
        # the caller's to name, they may know something about the
        # machine this does not, and silently running fewer pileups
        # than asked for is the behaviour change this declines to
        # make. That branch still warns.
        if budget is not None:
            ceiling = max(
                projected_peak(job_bytes, MAX_CONTIG_GROUPS) for job_bytes in sizes
            )
            workers = max(1, min(workers, budget // ceiling))
    else:
        # A caller who names the worker count gets batches sized for
        # the concurrency that will actually run, which is two steps.
        #
        # Clamped to the job count first. Clamping after the planning
        # solves the batch count for a concurrency that never runs:
        # two jobs with `workers=32` get five batches where one fits,
        # and four batches cost 25% more pileup wall clock at 56 GB of
        # input, so the headroom is paid for and no worker uses it.
        #
        # Then solved directly rather than through `plan_pileup`,
        # which maximises the worker count and only takes `n_jobs` as
        # a cap -- so above the core count it stops responding
        # altogether, and the gain gate can refuse the batches a
        # request needs. At 8 cores and 64 GB with 4 GB per job it
        # answered 2 batches for `workers` of 16, 32 and 64 alike,
        # projecting 80, 160 and 320 GB against a 49.7 GB budget.
        workers = max(1, min(int(workers), len(jobs)))
        groups = memory_floor(bytes_per_job, workers, budget).groups
    if target_partitions is None:
        target_partitions = default_partitions(workers)
    # The planner saw the mean, which says nothing about a set holding
    # one alignment many times the size of the rest. So each job is
    # sized from its own bytes here, and the plan's count is not a
    # floor under that answer: batches cost wall clock on small inputs,
    # the output is byte-identical at any count, and a job that reads a
    # megabyte beside one that reads 80 GB has nothing to gain from the
    # pair's average.
    #
    # `memory_floor` answers both axes at once for the bytes it is
    # handed -- it starts its search at that job's own `group_floor`,
    # so the scan cap is satisfied, and it tests the budget at the
    # concurrency that will run, so the memory bound is too. There is
    # nothing left for a second term to add.
    #
    # Taking the larger of the two was wrong in exactly one regime and
    # badly: once the mean itself crossed `MAX_GROUP_INPUT_BYTES`,
    # `plan_pileup` seeded its search at the mean's scan floor and
    # every job in the set was raised to it. An 80 GB alignment beside
    # a 1 MB one made the small job run six genome-wide aggregates
    # where one was required -- about 3.5x one ungrouped pass on an
    # unindexed BAM, bought for nothing. Both tests that look closest
    # use sets whose mean stays under the cap, where the larger-of-two
    # is a no-op.
    #
    # Unsized jobs are the exception and keep the plan's count, because
    # there are no own bytes to size from: `memory_floor(None, ...)`
    # scores the unknown-input peak, which is a guess, and the mean is
    # the better-informed guess of the two whenever the rest of the set
    # could be measured.
    over = []
    for index, (job, job_bytes) in enumerate(zip(jobs, sizes)):
        floor = memory_floor(job_bytes, workers, budget)
        own = floor.groups if job_bytes is not None else max(groups, floor.groups)
        resolved = int(job.setdefault("contig_groups", own))
        job.setdefault("target_partitions", target_partitions)
        # `fits` is false only when no count in range brings this job
        # inside the budget, and the count this job runs at is then
        # over it whatever it is -- the peak falls with the batch
        # count, so a count at or above the floor fits whenever the
        # floor does. What this deliberately does not cover is a
        # caller who pinned a count *below* the floor on a job that
        # would otherwise have fitted: an explicit count is honoured as
        # given, and the batch count it was weighed against is the scan
        # cap rather than the budget.
        if not floor.fits:
            over.append((index, resolved))

    if over:
        # Honoured anyway: a named concurrency is the caller's to name,
        # and an auto-planned one has already spent every batch it has.
        # Said out loud because the alternative is learning it as an
        # OOM kill, and because nothing else in the plan will mention
        # it. Once per fan-out rather than once per job, naming the
        # jobs, because a lopsided set is the shape this finds and the
        # outlier is the thing to act on.
        listed = ", ".join(
            "job {} at {} batch{}".format(index, count, "" if count == 1 else "es")
            for index, count in over[:5]
        )
        if len(over) > 5:
            listed += " and {} more".format(len(over) - 5)
        warnings.warn(
            "{} worker(s) over {} job(s) does not fit the memory budget "
            "at any contig batch count up to {}: {}. Those jobs are "
            "running over budget. The tables are unaffected -- batch "
            "count changes the peak and the wall clock, not what is "
            "counted -- but a worker may be killed before it writes "
            "one. {}".format(
                workers,
                len(jobs),
                MAX_CONTIG_GROUPS,
                listed,
                "Pass a smaller workers, or leave it unset and let the plan choose it."
                if named_workers
                else "The plan already lowered the concurrency as far as "
                "it goes, so this is one pileup of this job not fitting "
                "rather than a fan-out sized wrongly; the job needs a "
                "bigger machine or a smaller input.",
            ),
            UserWarning,
            stacklevel=2,
        )

    counts = sorted({int(job["contig_groups"]) for job in jobs})
    log.info(
        "Piling up %s dataset(s) across %s %s worker(s), %s "
        "partition(s) each, %s contig group(s)",
        len(jobs),
        workers,
        backend or "mapped",
        target_partitions,
        counts[0] if len(counts) == 1 else "{}-{}".format(counts[0], counts[-1]),
    )

    if runner is None:
        runner = _process_runner(workers)
    return list(runner(_run_one, jobs))


def _run_one(job):
    """Run one pileup in this process. The unit every runner maps over.

    One argument, because that is what a ``map``-shaped runner passes:
    everything the fan-out resolved is already in the job, which is
    also what has to cross the process boundary.
    """

    job = dict(job)
    alignments = job.pop("alignments")
    sequences = job.pop("sequences")
    name = job.pop("name")
    return pileup_alignments(alignments, sequences, name, **job)


def _normalize(jobs):
    """Copy and vet the job list, refusing a bad one before any work.

    Every refusal here is one a worker would otherwise reach minutes
    in, with however many jobs already written -- which is the whole
    reason the check is at the boundary rather than at the call.
    """

    # Copied, not just listed: `pileup_many` fills in defaults with
    # setdefault and must not write through to the caller's mapping.
    #
    # Any `Mapping` rather than a `dict`, because that is what the
    # published contract says -- "one mapping per pileup" -- and the
    # body only ever reads keys and writes to this copy. A
    # `MappingProxyType`, a `ChainMap` of defaults and a `Mapping`
    # subclass are all plausible ways a caller assembles a fan-out, and
    # all three were refused by a `TypeError` written against the
    # documentation it contradicted. The copy is what the process
    # boundary needs, and it is taken either way.
    jobs = [
        dict(job) if isinstance(job, collections.abc.Mapping) else job for job in jobs
    ]
    owners = {}
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(
                "job {} must be a mapping of pileup_alignments keyword "
                "arguments; got {!r}".format(i, type(job).__name__)
            )

        unknown = sorted(set(job) - _JOB_KEYS)
        if unknown:
            raise TypeError(
                "job {} names {} that pileup_alignments does not take "
                "({}); a key that reaches a worker unrecognised fails "
                "there rather than here".format(
                    i,
                    "a keyword" if len(unknown) == 1 else "keywords",
                    ", ".join(repr(key) for key in unknown),
                )
            )

        for required in ("alignments", "sequences", "name"):
            if required not in job:
                raise KeyError("job {} is missing {!r}".format(i, required))
            if job[required] is None:
                raise ValueError(
                    "job {} has {!r} set to None, which is not a value it "
                    "can take".format(i, required)
                )

        # One alignment given as a bare path rather than as a list of
        # one, which is the mistake the plural name invites. Accepted
        # as the trivial case rather than refused: the alternative is
        # that `os.path.getsize` is handed single characters, every
        # size comes back unsizable, and the *whole* fan-out is
        # re-planned against `UNKNOWN_INPUT_PEAK_BYTES` before the job
        # finally fails inside a worker with a message about the letter
        # `x`. Normalized here and in `pileup_alignments`, so the
        # direct call behaves the same way, and through the one
        # `fspath_or_none` both of them use so the two cannot disagree
        # about which shapes count as a path.
        if fspath_or_none(job["alignments"]) is not None:
            job["alignments"] = [job["alignments"]]

        # `name` is normalized here rather than only in
        # `pileup_alignments`, because the collision check below keys on
        # it: a `pathlib.Path` and an equal `str` are distinct keys, so
        # two jobs writing one table would go undetected and the later
        # writer would reap the earlier one's staging sibling mid-write.
        # A `bytes` name is the same hazard, which is the other half of
        # why this goes through `fspath_or_none` and not an
        # `os.PathLike` probe.
        name = fspath_or_none(job["name"])
        if name is None:
            raise TypeError(
                "job {} has name set to {}, which is not a path. It must "
                "be a str, bytes or os.PathLike -- a name that is not "
                "one reaches the path builder inside a worker, after "
                "that worker has registered its alignments.".format(
                    i, type(job["name"]).__name__
                )
            )
        job["name"] = name

        # Elementwise too, because `_job_bytes` stats these before any
        # worker sees them: a `bytes` path reached `os.path.getsize`,
        # which takes one, but `_bytes_per_job` folds the result and a
        # path that normalizes differently in two places is a plan made
        # against one set of sizes and a pileup run against another. A
        # non-path element is left alone for `register_alignments` to
        # refuse in its own words.
        pool = []
        for alignment in job["alignments"]:
            converted = fspath_or_none(alignment)
            pool.append(alignment if converted is None else converted)
        job["alignments"] = pool

        # `sequences` for the reason `pileup_alignments` records: it
        # reaches the manifest record through `os.path.abspath`, which
        # preserves `bytes`, and the write then fails inside a worker
        # with the tables already on disk.
        converted = fspath_or_none(job["sequences"])
        if converted is not None:
            job["sequences"] = converted

        # The empty pool is the quiet one: no alignment means no batch
        # to pile up, and the writer still commits a manifest listing
        # no files, which `require_complete` then passes.
        if not job["alignments"]:
            raise ValueError(
                "job {} names no alignments; a pileup over none of them "
                "writes an empty table that reads as a genome with no "
                "signal".format(i)
            )

        # A `None` spelled out loud is the parameter's own default, so
        # it must mean the same thing as leaving the key out.
        for key in _DEFAULTED_KEYS:
            if key in job and job[key] is None:
                del job[key]

        owner = owners.setdefault(job["name"], i)
        if owner != i:
            raise ValueError(
                "jobs {} and {} both write {!r}. Two pileups on one name "
                "race, and the loser does not merely lose a rename: the "
                "later writer reaps every staging sibling of that path, "
                "so it destroys the earlier one mid-write.".format(owner, i, job["name"])
            )
    return jobs


def _job_bytes(job):
    """Alignment bytes one job reads, or None if they cannot be sized.

    The one walk over a job's alignments. `pileup_many` calls this once
    per job and both folds the list and indexes it, so the unsizable
    edge is decided in a single place and no alignment is `stat`ed
    twice.
    """

    total = 0
    for path in job.get("alignments", ()):
        try:
            total += os.path.getsize(path)
        except OSError:
            return None
    return total or None


def _bytes_per_job(sizes):
    """Mean of ``sizes``, or None if any job could not be sized.

    The mean rather than the max: workers are long-lived and the sizes in
    one set are usually within a factor of two, so sizing every worker to
    the largest file would leave most of the machine idle. A set that is
    genuinely lopsided is better served by passing ``workers`` outright.

    Folds a list the caller already has rather than walking the corpus
    itself, because the per-job pass needs the same figures and a
    second walk means `stat`ing every alignment twice.
    """

    if not sizes or any(size is None for size in sizes):
        return None
    return int(sum(sizes) / len(sizes))


def _process_runner(workers):
    """A ``map``-shaped runner over a local pool of ``workers`` processes.

    ``loky`` rather than the default: it starts workers with ``spawn``,
    and `quadrat.reference` caches a polars-bio FASTA registration in a
    process-global that a forked child would inherit as a handle it does
    not own.

    .. rubric:: Implementation notes

    The pool outlives the fan-out, deliberately and not quite
    invisibly. ``loky``'s backend does not terminate its workers on
    completion -- "we want to reuse them in later calls" -- and the
    executor's ``idle_worker_timeout`` defaults to 300 seconds, so
    ``workers`` child processes stay alive for five minutes of
    idleness after this returns, each holding an initialized
    polars-bio/DataFusion runtime, its thread pool, its allocator
    arenas and a Python interpreter. That is most of what
    `PEAK_BYTES_PER_WORKER` accounts for, less the Arrow buffers an
    idle worker is not holding.

    Reuse is why a second `pileup_many` in the same process skips the
    spawn and the polars-bio import, which is the trade; what a caller
    sizing a *downstream* phase needs to know is that the memory is
    still occupied and that `memory_budget` reads the machine rather
    than this process tree, so it does not see it.
    """

    joblib = import_on_first_use("joblib")

    def run(function, items):
        return joblib.Parallel(n_jobs=workers, backend="loky")(
            joblib.delayed(function)(item) for item in items
        )

    return run
