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

A job is a plain dict of :func:`quadrat.pileup.pileup_alignments`
keyword arguments. Nothing here interprets it beyond
``target_partitions``, so it serialises by value and needs no class
definition on the far side of a process boundary.
"""

import os

from quadrat.pileup.plan import (
    _group_floor,
    default_partitions,
    plan_pileup,
)

#: Backends this package can dispatch to, all of them verified
#: byte-identical to each other.
BACKENDS = ("serial", "process")


def _run_one(job, target_partitions=None, verbose=False):
    """Run one pileup in this process. The unit every backend dispatches.

    Imported inside so that a worker pays for polars-bio and torch once,
    when it first runs a job, rather than at module import.
    """

    from quadrat.pileup import pileup_alignments

    job = dict(job)
    alignments = job.pop("alignments")
    sequences = job.pop("sequences")
    name = job.pop("name")
    job.setdefault("verbose", verbose)
    if target_partitions is not None:
        job.setdefault("target_partitions", target_partitions)
    return pileup_alignments(alignments, sequences, name, **job)


def _normalize(jobs):
    # Copied, not just listed: `pileup_many` fills in defaults with
    # setdefault and must not write through to the caller's dicts.
    jobs = [dict(job) if isinstance(job, dict) else job for job in jobs]
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(
                "job {} must be a dict of pileup_alignments keyword "
                "arguments; got {!r}".format(i, type(job).__name__)
            )
        for required in ("alignments", "sequences", "name"):
            if required not in job:
                raise KeyError("job {} is missing {!r}".format(i, required))
    return jobs


# public
def pileup_many(
    jobs,
    backend="serial",
    workers=None,
    target_partitions=None,
    verbose=False,
):
    """Run independent cut-site pileups concurrently.

    Parameters
    ----------
    jobs: list of dict
        Keyword arguments for :func:`~quadrat.pileup_alignments`.
        ``alignments``, ``sequences`` and ``name`` are required. Jobs must
        be independent: two jobs writing the same ``name`` race, and
        nothing here detects it.

    backend: str
        One of :data:`BACKENDS`.

    workers: int or None
        How many pileups to run at once. ``None`` uses
        :func:`default_workers`. Ignored by the ``serial`` backend.

    target_partitions: int or None
        Passed to each pileup, overriding what it would choose. ``None``
        uses :func:`default_partitions` for the concurrent backends and
        leaves the serial backend at its whole-machine default.

    verbose: bool
        Report progress, and pass through to each pileup.

    Returns
    -------
    paths: list of list of str
        Each job's written table paths, in channel order, in job order.

    Notes
    -----
    Every backend returns the same paths in the same order, and the
    tables they write are byte-identical; only the wall clock differs.
    Exceptions propagate from the first job that raises, and the tables
    of jobs that had already finished are left in place -- each one is
    committed atomically on its own, so a partial run leaves complete
    tables and absent ones, never a half-written table.
    """

    if backend not in BACKENDS:
        raise ValueError(
            "backend must be one of {}; got {!r}".format(", ".join(BACKENDS), backend)
        )

    jobs = _normalize(jobs)
    if not jobs:
        return []

    if backend == "serial":
        if verbose:
            print("Piling up {} dataset(s) serially".format(len(jobs)))
        return [_run_one(job, target_partitions, verbose) for job in jobs]

    planned, groups = plan_pileup(len(jobs), _bytes_per_job(jobs))
    if workers is None:
        workers = planned
    else:
        # A caller who names the worker count still gets batches sized
        # for it, since the peak has to fit whatever concurrency they
        # asked for.
        groups = plan_pileup(int(workers), _bytes_per_job(jobs))[1]
    workers = max(1, min(int(workers), len(jobs)))
    if target_partitions is None:
        target_partitions = default_partitions(workers)
    # The planner saw the mean, which says nothing about a set holding
    # one alignment many times the size of the rest -- and that is the
    # one at risk. Each job is raised to what its own bytes ask for,
    # rather than every job being sized for the largest: batches cost
    # wall clock on small inputs, and the output is byte-identical at
    # any count, so there is nothing to lose by letting them differ.
    for job in jobs:
        job.setdefault("contig_groups", max(groups, _group_floor(_job_bytes(job))))

    if verbose:
        counts = sorted({int(job["contig_groups"]) for job in jobs})
        print(
            "Piling up {} dataset(s) across {} {} worker(s), {} "
            "partition(s) each, {} contig group(s)".format(
                len(jobs),
                workers,
                backend,
                target_partitions,
                counts[0] if len(counts) == 1 else "{}-{}".format(counts[0], counts[-1]),
            )
        )

    if backend == "process":
        return _run_process(jobs, workers, target_partitions, verbose)

    # Unreachable while every member of `BACKENDS` has a branch above,
    # which is what makes this the guard against adding one without.
    raise NotImplementedError(
        "no dispatch for backend {!r}, which BACKENDS declares".format(backend)
    )


def _job_bytes(job):
    """Alignment bytes one job reads, or None if they cannot be sized.

    The per-job counterpart of :func:`_bytes_per_job`, which answers
    for the set. Kept separate rather than factored out of it: that
    one divides by the job count and returns None for a set with no
    alignments at all, and both edges are pinned by tests.
    """

    total = 0
    for path in job.get("alignments", ()):
        try:
            total += os.path.getsize(path)
        except OSError:
            return None
    return total or None


def _bytes_per_job(jobs):
    """Mean alignment bytes a job reads, or None if they cannot be sized.

    The mean rather than the max: workers are long-lived and the sizes in
    one set are usually within a factor of two, so sizing every worker to
    the largest file would leave most of the machine idle. A set that is
    genuinely lopsided is better served by passing ``workers`` outright.
    """

    sizes = []
    for job in jobs:
        for path in job.get("alignments", ()):
            try:
                sizes.append(os.path.getsize(path))
            except OSError:
                return None
    if not sizes:
        return None
    return int(sum(sizes) / len(jobs))


def _run_process(jobs, workers, target_partitions, verbose):
    """Fan out over a local process pool.

    ``loky`` rather than the default: it starts workers with ``spawn``,
    and ``giql_io`` caches a polars-bio FASTA registration in a
    process-global that a forked child would inherit as a handle it does
    not own.
    """

    from joblib import Parallel, delayed

    return list(
        Parallel(n_jobs=workers, backend="loky")(
            delayed(_run_one)(job, target_partitions, verbose) for job in jobs
        )
    )
