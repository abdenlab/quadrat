"""How many pileups to run at once, and in how many contig batches.

Getting the projection too low ends a run; getting it too high only
costs concurrency. So what these constants describe is an *envelope*
over measured peaks rather than a decomposition into independent
costs -- the surface is concave in both the worker count and the core
count, and the two interact.

Three terms, because only one of them follows the input::

    total peak  =  workers * PEAK_BYTES_PER_WORKER
                +  cores   * PEAK_BYTES_PER_CORE
                +  workers * bytes_per_job / groups * PEAK_BYTES_PER_INPUT_BYTE

Verified over 25 configurations -- 4 to 60 workers, 32 to 256 cores,
1.5 to 22.3 GB per job, 1 to 4 contig batches -- covering every one at
worst 1.06x and mean 1.29x. Outside that region this is extrapolation.

This module is a leaf. Nothing here imports the engine, the SQL or the
runner, which is what lets the runner and the alignment reader both
depend on it without a cycle.
"""

import os

#: Contigs are grouped so one pileup holds only part of the genome at a
#: time. Past about eight groups the peak stops falling -- the engine's
#: own floor dominates -- while the per-group overhead keeps accruing,
#: so solving for K is clamped here rather than allowed to run away.
#: Measured on a 3.9 GB ENCODE alignment: K=1 12.65 GB, K=2 6.15 GB,
#: K=4 5.95 GB, K=8 3.06 GB, K=16 2.56 GB, K=24 2.48 GB. That curve
#: predates the aggregate fix, and nothing since has been measured past
#: five batches, so the clamp rests on where the old peak flattened.
MAX_CONTIG_GROUPS = 8


def _n_cpus():
    """Cores this process may actually run on.

    ``sched_getaffinity`` rather than ``cpu_count`` because a pileup
    is usually launched under something that restricts it -- SLURM,
    a cgroup, a container quota, ``taskset`` -- and ``cpu_count``
    reports the whole machine regardless. Reading the machine instead
    of the allocation sets ``target_partitions`` to a parallelism the
    job cannot have and lets the worker cap oversubscribe by whatever
    factor the restriction was.

    Not available off Linux, where ``cpu_count`` is the only answer
    there is.
    """

    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


#: Peak resident memory one pileup needs for the alignment bytes it
#: reads, as a multiple of those bytes -- after the contig batches have
#: divided them, so this multiplies `bytes_per_job / contig_groups`.
#:
#: Re-derived from the twelve configurations in
#: `scripts/results/ladder-after-fix/`, five replicates each, spanning
#: 1.5-22.3 GB per job and 10-68 concurrent workers. It replaces 3.2,
#: which was measured under the pre-aggregate-join query and is wrong
#: twice over now.
#:
#: Wrong in magnitude, because dropping the sizes join from the probe
#: side removed the part of the peak that scaled with read count. What
#: is left is the grouped aggregate's state, one entry per distinct
#: `(chrom, pos)`, which saturates at genome coverage rather than at
#: reads. At four batches the pooled arms read 5.57, 10.87 and
#: 22.28 GB per job -- four-fold -- for 7.04, 8.99 and 11.61 GB per
#: worker, which is 1.65-fold.
#:
#: And wrong in shape, which matters more. A single multiple of the
#: input had to absorb two costs that do not scale with the input at
#: all -- see :data:`PEAK_BYTES_PER_WORKER` and
#: :data:`PEAK_BYTES_PER_CORE` -- and absorbing them is what made 3.2
#: *under*-estimate the per-BAM fan-out while over-estimating the
#: large pooled jobs threefold. Ten workers over 1.55 GB inputs
#: reached 78.4 GB against the 49.5 GB it projected, and over 1.81 GB
#: inputs 85.0 GB against 58.0 GB. Under-estimating is the direction
#: that ends a run, and no value of a single constant fixes it: the
#: smallest one that covers every arm measured is 5.0, and it then
#: over-provisions the large pooled jobs close to fivefold.
PEAK_BYTES_PER_INPUT_BYTE = 1.5


#: Peak resident memory one pileup holds whatever it reads: the engine,
#: its thread pool, the Python interpreter the worker runs in, and the
#: Arrow buffers in flight between them. Contig batches do not reduce
#: it, which is the whole reason it is separate -- a planner that folds
#: it into a multiple of the input believes that splitting the genome
#: eight ways makes it nearly three times smaller.
PEAK_BYTES_PER_WORKER = 2 * 1024**3


#: Peak resident memory the engine holds per core, across all the
#: pileups running at once rather than per worker. Subtracted from the
#: budget once; see :func:`_memory_budget`.
#:
#: Measured by confining the process tree with `taskset` and changing
#: nothing else -- same twelve workers, same eight partitions, same
#: input (`scripts/results/planner-sweep/`). The part of the peak that
#: is not per-worker comes to 9.1, 21.3, 28.9 and 32.6 GB at 32, 64,
#: 128 and 256 cores. So it does follow the cores, and a flat
#: machine-wide term is ruled out: an eightfold change in cores moves
#: the peak from 50.2 to 73.8 GB with everything else held.
#:
#: What it is *not* is DataFusion's record batches in flight. That was
#: the mechanism this comment used to assert, on the reasoning that
#: `default_partitions` keeps `workers * partitions` near the core
#: count. The partition sweep refutes it: at twelve workers, taking
#: `target_partitions` from 4 to 32 moves the peak 72.0 to 76.8 GB --
#: inside the 4-6% replicate spread, and not even monotonic -- while
#: the wall clock moves 59.7 s to 38.3 s, so the setting plainly
#: reached the engine. Something sized once from the machine's
#: parallelism fits: per-thread allocator arenas and runtime thread
#: state do, batches in flight do not.
#:
#: The relationship saturates rather than being proportional: each
#: doubling of the cores adds about half what the previous one did, so
#: a linear term over-reserves on a large machine (1.4x at 256 cores)
#: and is nearly tight in the middle (1.06x at 64). Linear anyway,
#: because a saturating curve here would be two more parameters fitted
#: to four points from one machine, and because the error it makes on
#: a large machine is 6% of that machine's budget while the error a
#: curve might make in the middle is not.
PEAK_BYTES_PER_CORE = 192 * 1024**2


# These three terms are an envelope, not a decomposition.
#
# The surface they bound is concave in both directions. The marginal
# cost of a worker falls as the fan-out widens -- 5.70, 5.12, 3.88
# and 2.17 GB going 4 to 6 to 12 to 24 to 36 workers -- and the cost
# of the cores saturates as above. The two also interact: 64 cores to
# 256 costs 6.7 GB at four workers and 11.4 GB at twelve. So no term
# here is an independent cost that could be measured on its own;
# together they are a plane sitting above a concave surface, which is
# why they stay safe between the points they touch and why they are
# loosest at the extremes (2.1x at four workers on 256 cores).
#
# Being linear in the worker count is the conservative error in the
# direction that matters: the marginal worker costs less than the
# model charges, and most at the wide fan-outs where running out of
# memory ends a run rather than slowing it.
#
# Verified over 25 configurations -- 4 to 60 workers, 32 to 256
# cores, 1.5 to 22.3 GB per job, 1 to 4 contig batches -- in
# `scripts/results/ladder-after-fix/` and
# `scripts/results/planner-sweep/`. Every one is covered, worst
# 1.06x, mean 1.29x. Outside that region this is extrapolation.

#: Per-worker budget when the input size is not known at all -- the
#: whole figure, not merely the part that scales with the input. The
#: measured figure for a typical ATAC alignment, used so a caller that
#: does not pass sizes is still bounded somewhere defensible. It stays
#: comfortably above what the terms above would give such a job
#: (2 GB plus 1.5 times 3.9 GB, so about 7.9 GB), which is the
#: direction to be wrong in when nothing is known.
UNKNOWN_INPUT_PEAK_BYTES = 12 * 1024**3


#: Fraction of total memory to plan against. A pileup is not the only
#: thing on the machine, and the peak is a sample rather than a
#: guarantee, so the budget leaves a fifth of memory alone.
MEMORY_HEADROOM = 0.8


#: How much concurrency a larger batch count must buy before it is
#: worth taking. Batches are not free: four of them cost 25% more
#: pileup wall clock at 56 GB of total input (`scripts/results/ladder/`),
#: so a batch count that buys a few percent more workers is a loss.
#:
#: It needs saying because the shape of :func:`projected_peak` makes
#: splitting look free on paper. The part that falls with the batch
#: count falls in proportion, so on a small input it can always be
#: driven to nearly nothing -- and the planner would then take eight
#: batches to turn sixteen workers into nineteen. The threshold is the
#: measured cost of four batches, which is the count that gain would
#: typically be bought at.
CONTIG_GROUP_GAIN = 1.25


#: Most alignment bytes one contig batch should have to scan. A ceiling
#: on the size of a single query rather than on the machine, and the
#: one bound memory headroom cannot express: a batch that fits in
#: memory can still be too large to compute, because an aggregate over
#: enough records runs into limits that have nothing to do with how
#: much is free. A 21.2 GB RNA-seq alignment read in one batch is one
#: aggregate over 291,063,148 records, and the engine has died on it
#: where 2 and 4 batches succeeded.
#:
#: Solving from headroom alone gets this exactly backwards -- the more
#: memory a machine has, the more certainly it answers one batch -- so
#: the biggest machines were the ones that fell over.
#:
#: Eight GB rather than less because batches are not free below about
#: 100 GB of total input: four batches cost 25% more pileup wall clock
#: at 56 GB of input and 2% less at 109 GB, for 37-54% lower peak at
#: both, with byte-identical output (`scripts/results/ladder/`). At
#: 8 GB the cap cannot fire on inputs that size -- the largest single
#: file in either set is 3.3 GB.
#:
#: Two honest limits. Bytes are a proxy for records and an imperfect
#: one: a 35.2 GB starr-seq alignment in the same set piles up fine in
#: one batch, because read density is nothing like uniform across
#: assays. And `MAX_CONTIG_GROUPS` is a hard ceiling, so this
#: saturates at 64 GB of input per job and asks for no more.
MAX_GROUP_INPUT_BYTES = 8 * 1024**3


def _memory_budget():
    """Bytes the workers may share, or None if unknowable.

    What is left of memory once the headroom and the engine's own
    machine-wide cost are set aside, so that what remains can be
    divided by the per-worker figure. :data:`PEAK_BYTES_PER_CORE` is
    subtracted here rather than charged to each worker because it
    follows the cores and not the fan-out -- measured by confining the
    tree with `taskset` and leaving the worker count alone.
    """

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    return max(1, int(total * MEMORY_HEADROOM) - _n_cpus() * PEAK_BYTES_PER_CORE)


def projected_peak(bytes_per_job, groups=1):
    """Peak bytes one pileup is expected to reach, machine cost aside.

    A fixed part and a part that scales with what this job reads:
    :data:`PEAK_BYTES_PER_WORKER` plus
    :data:`PEAK_BYTES_PER_INPUT_BYTE` times the bytes one contig batch
    scans. The machine-wide term is not here; :func:`_memory_budget`
    takes it off the top instead.

    ``groups`` divides the second part and leaves the first alone,
    which is what the batches actually do. The division is in
    proportion and not a square root: one batch in K holds about one
    Kth of the distinct positions, and the aggregate's state is one
    entry per position. An earlier form raised `groups` to the 0.5
    only because, with nowhere to put the fixed part, that was the
    exponent which kept the single multiple from collapsing on a
    machine that had to split. Measured against
    `scripts/results/ladder-after-fix/`, a proportional fall with the
    fixed part broken out fits the corpus better than any exponent
    fits without it.
    """

    if not bytes_per_job:
        # Nothing to scale, but the batches still divide whatever is
        # there, so the unknown figure is split the same way: the part
        # of it above the fixed cost behaves like the input part.
        implied = max(0, UNKNOWN_INPUT_PEAK_BYTES - PEAK_BYTES_PER_WORKER)
        return int(PEAK_BYTES_PER_WORKER + implied / float(max(1, groups)))
    scanned = bytes_per_job / float(max(1, groups))
    return int(PEAK_BYTES_PER_WORKER + max(1, int(scanned * PEAK_BYTES_PER_INPUT_BYTE)))


def _group_floor(input_bytes):
    """Fewest contig batches that keep one query's scan under the cap.

    Independent of memory by design: this is how large a single
    aggregate may get, not how much of the machine is free. See
    :data:`MAX_GROUP_INPUT_BYTES`.
    """

    if not input_bytes:
        return 1
    groups = -(-int(input_bytes) // MAX_GROUP_INPUT_BYTES)
    return max(1, min(MAX_CONTIG_GROUPS, groups))


def plan_pileup(n_jobs, bytes_per_job=None):
    """Choose ``(workers, contig_groups)`` together for a fan-out.

    These used to be one decision: memory bounded the worker count, and
    a machine too small simply ran fewer pileups at once. Splitting the
    genome into contig batches makes the per-worker peak something we
    choose rather than something we are handed, so the two are solved
    together -- batches are spent to buy concurrency, up to
    :data:`~quadrat.MAX_CONTIG_GROUPS`.

    On a 64-core, 64 GB machine with 4 GB inputs that is the difference
    between four pileups at 8.0 GB each and twelve at 3.2 GB.

    Only where the batches are worth their wall clock, though; see
    :data:`CONTIG_GROUP_GAIN`.

    The smallest batch count that reaches the best worker count wins,
    because batches are not free: two or four are faster than one, but
    eight is slower, and there is no reason to pay for memory headroom
    nothing will use.

    Never fewer than :data:`MAX_GROUP_INPUT_BYTES` asks for, though.
    That floor is a bound on how much one query scans rather than on
    how much memory is free, and it exists because solving from
    headroom alone answers one batch on exactly the machines with the
    most room -- which is where the largest inputs get run.
    """

    cap = min(n_jobs, _n_cpus())
    floor = _group_floor(bytes_per_job)
    budget = _memory_budget()
    if budget is None:
        # Nothing to plan against. Four is what this has always
        # guessed, and the scan cap is the one thing still knowable
        # without a memory figure.
        return max(1, min(cap, 4)), floor

    # `best[0]` starts at zero so the first candidate always wins,
    # which is what keeps the answer at or above the floor.
    best = (0, floor)
    for groups in range(floor, MAX_CONTIG_GROUPS + 1):
        fits = budget // projected_peak(bytes_per_job, groups)
        workers = max(1, min(cap, fits))
        # Strictly better for the first candidate, and materially
        # better for every one after it: see `CONTIG_GROUP_GAIN`.
        enough = best[0] * CONTIG_GROUP_GAIN if best[0] else 0
        if workers > best[0] and workers >= enough:
            best = (workers, groups)
    return best


def default_workers(n_jobs, backend="process", bytes_per_job=None):
    """How many pileups to run at once, by default.

    One per job, bounded by the cores and by memory. The measured speed
    curve flattens well before the cap (see the module docstring: 16 and
    20 workers are indistinguishable on 128 cores), so erring high costs
    little speed -- what matters is not landing on the steep part, at two
    or four.

    Memory is the bound that actually bites, and it bites hard: ten
    pileups over 3.9 GB inputs peaked at 120 GB together. Exceeding this
    budget does not make a run slow, it makes it die, so the bound is
    deliberately conservative where the speed cap is not -- an envelope
    over every arm measured rather than a fit through them. What the
    peak is made of is :func:`projected_peak`.

    This is the worker half of :func:`plan_pileup`, which chooses the
    contig batch count alongside it; the two are one decision now, since
    batches are what buy the concurrency.

    Parameters
    ----------
    n_jobs: int
        How many pileups there are to run.

    backend: str
        Accepted for symmetry with :func:`pileup_many`; the budget does
        not depend on it.

    bytes_per_job: int or None
        Alignment bytes one job will read. When given, the per-worker
        budget scales with it, which is what makes the bound right for
        small inputs as well as large. ``None`` falls back to
        :data:`UNKNOWN_INPUT_PEAK_BYTES`.
    """

    if n_jobs <= 0:
        return 0
    return plan_pileup(n_jobs, bytes_per_job)[0]


def default_partitions(workers):
    """DataFusion ``target_partitions`` for one pileup among ``workers``.

    ``pileup_alignments`` otherwise sets this to every core, and the
    setting is process-global, so K concurrent pileups would each ask
    for the whole machine. Dividing it costs little -- one pileup gets
    only about 2.1x from 14 partitions -- and avoids oversubscribing the
    thread pool K-fold.
    """

    return max(1, _n_cpus() // max(1, workers))
