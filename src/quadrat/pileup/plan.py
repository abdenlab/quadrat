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
runner, which is what lets its three dependents -- the runner, the
single-channel pileup and the subpackage's own re-export surface --
depend on it without a cycle. The alignment reader is not among them
and never was: it imports only `quadrat._engine`.

PROVENANCE. Every fitted constant below cites a measurement, and the
measurements were taken in cherimoya -- https://github.com/conradbzura/cherimoya
-- under `scripts/`, which this repository does not carry. The corpus
is not vendored because it is ENCODE alignments in the tens of
gigabytes; what is reproducible from here is the method each constant
states, not the runs. Citations name the arm of that corpus rather
than a path, since a path that is not here cannot be followed. The one
measurement re-derived in this repository is the track row-group size,
whose method is stated inline beside it.
"""

# Recorded rather than left to the absence of a leading underscore.
# Everything here is package-internal: no name carries a `# public`
# marker and none is re-exported from `quadrat`. What this list says is
# which of them a sibling module or a test may read, as against which
# belong to the body of this one.
#
# The constants are in it because they are the model: a test that
# restates the budget rather than patching a private helper has to
# reach them, and the two `#:` blocks on each one are written for that
# reader. A test that pins a figure by patching the function that
# computes it pins the patch; one that restates it from the published
# constants fails when the model changes, which is the point.
# Underscoring them would make that restatement the illegal option and
# contradict the docstrings that name them.
__all__ = [
    "CONTIG_GROUP_GAIN",
    "MAX_CONTIG_GROUPS",
    "MAX_GROUP_INPUT_BYTES",
    "MEMORY_HEADROOM",
    "PEAK_BYTES_PER_CORE",
    "PEAK_BYTES_PER_INPUT_BYTE",
    "PEAK_BYTES_PER_WORKER",
    "UNKNOWN_INPUT_PEAK_BYTES",
    "MemoryFloor",
    "PileupPlan",
    "default_partitions",
    "group_floor",
    "memory_budget",
    "memory_floor",
    "n_cpus",
    "plan_pileup",
    "projected_peak",
]

import os
import types
import typing

#: Contigs are grouped so one pileup holds only part of the genome at a
#: time. Past about eight groups the peak stops falling -- the engine's
#: own floor dominates -- while the per-group overhead keeps accruing,
#: so solving for K is clamped here rather than allowed to run away.
#: Measured on a 3.9 GB ENCODE alignment: K=1 12.65 GB, K=2 6.15 GB,
#: K=4 5.95 GB, K=8 3.06 GB, K=16 2.56 GB, K=24 2.48 GB. That curve
#: predates the grouping fix, and nothing since has been measured past
#: five batches, so the clamp rests on where the old peak flattened.
MAX_CONTIG_GROUPS = 8


#: Peak resident memory one pileup needs for the alignment bytes it
#: reads, as a multiple of those bytes -- after the contig batches have
#: divided them, so this multiplies `bytes_per_job / contig_groups`.
#:
#: Re-derived from the twelve configurations in
#: the post-grouping-fix ladder of cherimoya's benchmark corpus
#: (`scripts/results/ladder-after-fix/` there, not vendored here --
#: see PROVENANCE below), five replicates each, spanning
#: 1.5-22.3 GB per job and 10-68 concurrent workers.
#:
#: A figure near 3.2 belongs to a query that joins contig sizes on the
#: probe side, where part of the peak scales with read count. The
#: query of counts does not: the join sits above the grouping, so what
#: the input bytes buy is the grouping's state, one entry per distinct
#: `(chrom, pos)`, which saturates at genome coverage rather than at
#: reads. The query of coverage deltas also joins the sizes to every
#: record below the grouping, to clip each span to its contig, with the
#: per-contig side collected and the records streamed through it; the
#: figures here predate that join and were not measured against it.
#: At four batches the pooled arms read 5.57, 10.87 and
#: 22.28 GB per job -- four-fold -- for 7.04, 8.99 and 11.61 GB per
#: worker, which is 1.65-fold. So the multiple is smaller than the
#: read-scaled one, and smaller by more than the batch count explains.
#:
#: The shape matters more than the magnitude. Making this the *only*
#: term, with no per-worker and no per-core cost beside it, forces one
#: multiple of the input to absorb two costs that do not scale with
#: the input at all -- see `PEAK_BYTES_PER_WORKER` and
#: `PEAK_BYTES_PER_CORE`. A single term of 3.2 therefore
#: *under*-estimates the per-BAM fan-out while over-estimating the
#: large pooled jobs threefold: ten workers over 1.55 GB inputs reach
#: 78.4 GB against the 49.5 GB it projects, and over 1.81 GB inputs
#: 85.0 GB against 58.0 GB. Under-estimating is the direction that
#: ends a run, and no value of a single constant fixes it -- the
#: smallest one covering every arm measured is 5.0, which then
#: over-provisions the large pooled jobs close to fivefold. Three
#: terms rather than a better-fitted one is the conclusion.
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
#: budget once; see `memory_budget`.
#:
#: Measured by confining the process tree with `taskset` and changing
#: nothing else -- same twelve workers, same eight partitions, same
#: input (the planner sweep of cherimoya's corpus). The part of the peak that
#: is not per-worker comes to 9.1, 21.3, 28.9 and 32.6 GB at 32, 64,
#: 128 and 256 cores. So it does follow the cores, and a flat
#: machine-wide term is ruled out: an eightfold change in cores moves
#: the peak from 50.2 to 73.8 GB with everything else held.
#:
#: What it is *not* is DataFusion's record batches in flight, which is
#: the obvious candidate: `default_partitions` keeps
#: `workers * partitions` near the core count, so batches in flight
#: would follow the cores for exactly the reason this term does. The
#: partition sweep refutes it: at twelve workers, taking
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
#: pileup wall clock at 56 GB of total input (cherimoya's ladder),
#: so a batch count that buys a few percent more workers is a loss.
#:
#: It needs saying because the shape of `projected_peak` makes
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
#: memory can still be too large to compute, because a grouping over
#: enough records runs into limits that have nothing to do with how
#: much is free. A 21.2 GB RNA-seq alignment read in one batch is one
#: grouping over 291,063,148 records, and the engine has died on it
#: where 2 and 4 batches succeeded.
#:
#: Solving from headroom alone gets this exactly backwards -- the more
#: memory a machine has, the more certainly it answers one batch -- so
#: the biggest machines were the ones that fell over.
#:
#: Eight GB rather than less because batches are not free below about
#: 100 GB of total input: four batches cost 25% more pileup wall clock
#: at 56 GB of input and 2% less at 109 GB, for 37-54% lower peak at
#: both, with byte-identical output (cherimoya's ladder). At
#: 8 GB the cap cannot fire on inputs that size -- the largest single
#: file in either set is 3.3 GB.
#:
#: The figure is for a pileup whose records contribute one row each to
#: the grouping, and the rows are what the bytes stand in for: one
#: whose records contribute ``m`` rows each scans at most this divided
#: by ``m`` -- see `group_floor`. The divisor is the record's own count
#: of rows per record, as `quadrat.pileup.aggregate.Aggregate` answers
#: it, rather than a fitted multiple.
#:
#: Two honest limits. Bytes are a proxy for records and an imperfect
#: one: a 35.2 GB starr-seq alignment in the same set piles up fine in
#: one batch, because read density is nothing like uniform across
#: assays. And `MAX_CONTIG_GROUPS` is a hard ceiling, so this
#: saturates at 64 GB of input per job, divided by ``m``, and asks for
#: no more.
MAX_GROUP_INPUT_BYTES = 8 * 1024**3


#: Where the kernel publishes this process's memory allowance, cgroup
#: v2 first and v1 second. Module constants rather than literals
#: because they are the only seam a test has onto this: the files are
#: absent on the platforms the suite runs on, so a test that needs the
#: branch has to point these somewhere it can write.
#:
#: A v2 limit reads as the literal `max` when there is none.
_CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max"


_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


#: SLURM's two ways of stating a job's memory, in MB unless a unit
#: suffix says otherwise. Per-node is the allocation outright; per-CPU
#: has to be multiplied by the cores this process may run on, which is
#: what `n_cpus` answers.
_SLURM_MEM_VARS = ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")


# `MappingProxyType` below so the binding and the referent are both
# immutable, which is what the SCREAMING casing claims: a bare dict
# under this name reads as a true constant and is not one.
# `_SLURM_MEM_VARS` above is a tuple and needs no wrapper, and the
# pair reads alike this way rather than by lower-casing one of them.
#: Multipliers for the unit suffix SLURM allows on those two figures,
#: relative to the MB the bare number is in. `K` is therefore a
#: fraction, which is why the product below is taken in bytes rather
#: than in megabytes: integer megabytes would floor a sub-MB
#: allocation to zero, and the `limit > 0` filter would then drop it as
#: though nothing had been stated -- reading a small allocation as no
#: allocation, which is the one direction this must not be wrong in.
_SLURM_MEM_UNITS = types.MappingProxyType(
    {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0**2}
)


class PileupPlan(typing.NamedTuple):
    """What `plan_pileup` decided, by name.

    A `NamedTuple` rather than a bare pair, and not only for
    readability: the two elements are read at three call sites and
    `plan_pileup` had to spell the shape out in its own summary, which
    is the signal that positional indexing had become the published
    read protocol. Which element is which also carries more weight than
    usual, since the planner's bugs have been about exactly that.

    Still a tuple, so ``== (1, 3)`` and unpacking both keep working.
    """

    #: How many pileups to run at once.
    workers: int

    #: How many contig batches to run each of them in.
    contig_groups: int


class MemoryFloor(typing.NamedTuple):
    """What `memory_floor` decided, by name.

    A pair rather than a bare count, because the count alone cannot
    say whether it is a solution or the best available failure --
    ``MAX_CONTIG_GROUPS`` is both the answer when the largest batch
    count is what the concurrency needs and the answer when no count
    fits at all. A caller that warns on the second has to be able to
    tell them apart.

    Still a tuple, so unpacking keeps working.
    """

    #: How many contig batches this job needs at that concurrency.
    groups: int

    #: Whether that count brings it inside the budget. ``True`` when
    #: the budget is unknowable, since an unknown bound is not a bound
    #: that was exceeded.
    fits: bool


def projected_peak(bytes_per_job, groups=1, points_per_record=1):
    """Peak bytes one pileup is expected to reach, machine cost aside.

    A fixed part and a part that scales with what this job reads:
    `PEAK_BYTES_PER_WORKER` plus
    `PEAK_BYTES_PER_INPUT_BYTE` times the bytes one contig batch
    scans, times ``points_per_record``. The machine-wide term is not
    here; `memory_budget` takes it off the top instead.

    ``points_per_record`` multiplies the second part for the reason it
    divides the scan cap in `group_floor`: what the input bytes buy is
    the grouping's state, one entry per distinct key, and a pileup
    contributing two rows per record keys about twice as many entries
    as one contributing one. A job of ``b`` bytes at two rows per
    record is therefore projected as one of ``2 * b`` bytes at one.
    The fit behind `PEAK_BYTES_PER_INPUT_BYTE` is of one-row
    pileups, so this is the model's extrapolation and not a
    measurement. An unsized job's figure is not multiplied: it is a
    guess at a whole peak rather than a multiple of known bytes.

    ``groups`` divides the second part and leaves the first alone,
    which is what the batches actually do. The division is in
    proportion and not a square root: one batch in K holds about one
    Kth of the distinct positions, and the grouping's state is one
    entry per position. A fractional exponent -- ``groups`` to the 0.5
    -- fits better than proportion only when the fixed part has
    nowhere else to go, because it is then standing in for the part of
    the peak the batches do not divide. With that part broken out as
    its own term there is nothing left for the exponent to absorb, and
    measured against cherimoya's post-fix ladder a proportional fall
    fits the corpus better than any exponent fits without it.
    """

    if not bytes_per_job:
        # Nothing to scale, but the batches still divide whatever is
        # there, so the unknown figure is split the same way: the part
        # of it above the fixed cost behaves like the input part.
        implied = max(0, UNKNOWN_INPUT_PEAK_BYTES - PEAK_BYTES_PER_WORKER)
        return int(PEAK_BYTES_PER_WORKER + implied / float(max(1, groups)))
    scanned = bytes_per_job * points_per_record / float(max(1, groups))
    return int(PEAK_BYTES_PER_WORKER + max(1, int(scanned * PEAK_BYTES_PER_INPUT_BYTE)))


def plan_pileup(n_jobs, bytes_per_job=None, budget=None, points_per_record=1):
    """Choose the worker count and the batch count together for a fan-out.

    The two are one decision rather than two, because splitting the
    genome into contig batches makes the per-worker peak something the
    planner chooses rather than something it is handed: batches are
    spent to buy concurrency, up to `MAX_CONTIG_GROUPS`, and only
    where they are worth their wall clock -- see `CONTIG_GROUP_GAIN`.
    On a 64-core, 64 GB machine with 4 GB inputs that is the
    difference between four pileups at 8.0 GB each and twelve at
    3.2 GB.

    :param n_jobs: How many pileups there are to run. Zero or fewer
        plans nothing and answers zero workers, so a caller asking what
        to run for an empty set is told to run nothing rather than one.
    :param bytes_per_job: Alignment bytes one pileup reads, or ``None``
        when they cannot be sized.
    :param budget: Bytes the workers may share, from `memory_budget`.
        ``None`` reads it here. The same parameter `memory_floor`
        takes, and for the same reason: a caller that solves both
        halves of one plan passes the budget it already holds, so the
        worker count and the per-job batch counts are solved against
        one figure. The budget comes from mutable OS state and two
        reads are not guaranteed to agree.
    :param points_per_record: Rows one record contributes to the
        grouping, as `group_floor` takes it.
    :returns: a `PileupPlan`, whose two elements are read by name.
        Unpacking and ``== (1, 3)`` both keep working, and a caller who
        wants to name or annotate the type reaches it here rather than
        through `quadrat.pileup` -- neither this function nor its
        return type is part of the supported surface.

    .. rubric:: Implementation notes

    Feasibility comes first and concurrency second. A batch count that
    fits nothing in the budget scores zero rather than one, so the
    search keeps looking; only once a count admits a worker does the
    gain gate decide whether a larger one is worth its wall clock. The
    smallest batch count that reaches the best worker count then wins,
    because batches are not free: two or four are faster than one, but
    eight is slower, and there is no reason to pay for memory headroom
    nothing will use.

    One worker is a floor on the answer rather than on the search, and
    that distinction is the whole of it. Clamping inside the loop made
    "one worker fits" and "not even one fits" score the same, so a
    single pileup on a machine too small for it stayed at the scan
    floor and projected over its own budget -- 1.18x on 32 cores with
    16 GB reading 4 GB, and 2.83x on 8 cores with 8 GB reading 7.9 GB,
    with larger counts available that fit in both cases.



    Never fewer than `MAX_GROUP_INPUT_BYTES` asks for, though.
    That floor is a bound on how much one query scans rather than on
    how much memory is free, and it exists because solving from
    headroom alone answers one batch on exactly the machines with the
    most room -- which is where the largest inputs get run.
    """

    floor = group_floor(bytes_per_job, points_per_record)
    if n_jobs <= 0:
        # Nothing to run, so nothing to plan. The batch count is still
        # the scan floor, which is a property of the bytes rather than
        # of the job count.
        return PileupPlan(0, floor)
    cap = min(n_jobs, n_cpus())
    if budget is None:
        budget = memory_budget()
    if budget is None:
        # Nothing to plan against. Four is what this has always
        # guessed, and the scan cap is the one thing still knowable
        # without a memory figure.
        return PileupPlan(max(1, min(cap, 4)), floor)

    # `best[0]` starts at zero so a count that fits nothing cannot
    # win, and `best[1]` is what the answer falls back to when none of
    # them fits: `MAX_CONTIG_GROUPS`, which is the smallest projected
    # peak in range rather than the largest. Over budget at every
    # count, the only thing left to choose is how far over, and the
    # module's asymmetry says lower -- too low a projection ends a run
    # while too high costs wall clock. It is also the answer
    # `memory_floor` already gives in exactly this situation, so the
    # two solvers agree about what to hand back.
    best = (0, MAX_CONTIG_GROUPS)
    for groups, peak in _admissible_counts(bytes_per_job, points_per_record):
        fits = budget // peak
        # Not clamped here: a zero has to reach the comparison, or
        # "one fits" and "none fits" are indistinguishable and the
        # search stops at the first candidate. Clamped at the return.
        workers = min(cap, fits)
        # Strictly better for the first candidate, and materially
        # better for every one after it: see `CONTIG_GROUP_GAIN`.
        enough = best[0] * CONTIG_GROUP_GAIN if best[0] else 0
        if workers > best[0] and workers >= enough:
            best = (workers, groups)
    return PileupPlan(max(1, best[0]), best[1])


def default_partitions(workers):
    """DataFusion ``target_partitions`` for one pileup among ``workers``.

    ``pileup_alignments`` otherwise sets this to every core, and the
    setting is process-global, so K concurrent pileups would each ask
    for the whole machine. Dividing it costs little -- one pileup gets
    only about 2.1x from 14 partitions -- and avoids oversubscribing the
    thread pool K-fold.
    """

    return max(1, n_cpus() // max(1, workers))


def n_cpus():
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


def memory_budget():
    """Bytes the workers may share, or None if unknowable.

    What is left of memory once the headroom and the engine's own
    machine-wide cost are set aside, so that what remains can be
    divided by the per-worker figure. `PEAK_BYTES_PER_CORE` is
    subtracted here rather than charged to each worker because it
    follows the cores and not the fan-out -- measured by confining the
    tree with `taskset` and leaving the worker count alone.

    Unprefixed because `quadrat.pileup.run` reads it: a fan-out reads
    the budget once and hands it to `memory_floor` for every job,
    rather than having it re-derived per job from OS state that cannot
    change during the loop.

    The total is the smaller of the host's physical memory and
    whatever allowance `_allocation_bytes` finds, so the two axes
    of the plan are read the same way: this process's share of the
    machine rather than the machine.

    The per-core term is clamped to half the headroom, and the clamp is
    a bound on the approximation rather than a change to the model. The
    relationship saturates and `PEAK_BYTES_PER_CORE` is fitted linear
    anyway, which over-reserves on a large machine -- its own
    documentation says so -- and the two axes are read from different
    sources: `n_cpus` sees a cpuset but not a cgroup CPU quota, while
    `_allocation_bytes` does see a memory cap. A container run as
    ``--memory=8g`` with no cpuset on a 64-core host therefore charges
    itself 12 GB of per-core cost against 6.4 GB of headroom, and
    without the clamp the subtraction went negative and the `max`
    turned it into a one-byte budget -- which every consumer reads as a
    real bound rather than as "unknowable", so no batch count fits and
    the planner answers the arrangement with the highest peak at
    exactly the allocations where memory is scarcest.
    Clamping keeps the one thing the planner does know, which is how
    large the allocation is.

    The better fix is to derive this term's core count from the cores
    the *allocation* implies rather than from raw affinity, so a
    memory-capped but not CPU-capped container does not charge itself
    for the whole host. That is a change to the model and this is not
    it.
    """

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    allocation = _allocation_bytes()
    if allocation is not None:
        total = min(total, allocation)
    headroom = int(total * MEMORY_HEADROOM)
    machine_cost = min(n_cpus() * PEAK_BYTES_PER_CORE, headroom // 2)
    return max(1, headroom - machine_cost)


def memory_floor(input_bytes, workers, budget=None, points_per_record=1):
    """Fewest contig batches that keep ``workers`` of this job in budget.

    The memory counterpart of `group_floor`, which answers the
    same question on the scan axis. A caller with a fixed concurrency
    and a known input has one free variable left -- the batch count --
    and this solves for it directly rather than through
    `plan_pileup`, which maximises the worker count instead and
    so answers a different question.

    Never below the scan floor, because the two bounds are independent
    and a batch count has to satisfy both.

    :param budget: Bytes the workers may share, from
        `memory_budget`. ``None`` reads it here. A caller solving for
        many jobs passes it once rather than having it re-read from
        ``/sys/fs/cgroup``, the environment and ``sysconf`` per job --
        and the budget comes from mutable OS state, so two reads are
        not guaranteed to agree.
    :param points_per_record: Rows one record contributes to the
        grouping, as `group_floor` takes it, and scales the projected
        peak as `projected_peak` takes it.
    :returns: a `MemoryFloor`. ``fits`` is what makes the batch count
        answerable: the largest count available is the best this can do
        when nothing fits, and silently being over budget is how a
        fan-out gets OOM-killed, so the result says which of the two it
        is rather than leaving a caller to ask a second function.

    .. rubric:: Implementation notes

    One function rather than this and a predicate beside it. Both
    evaluated ``workers * projected_peak(input_bytes, groups) <=
    budget`` -- the same comparison, scored differently -- and the
    caller that needed both called them in sequence, reading the budget
    twice and projecting the same batch count twice.

    ``fits`` answers the resolved count as well as the floor, which is
    what lets a caller raise the count and keep the answer: the peak
    falls monotonically with the batch count, so a count at or above
    the floor fits whenever the floor does, and when nothing in range
    fits then nothing in range fits.
    """

    floor = group_floor(input_bytes, points_per_record)
    if budget is None:
        budget = memory_budget()
    if budget is None:
        # Nothing to solve against. An unknown bound is not a bound
        # that was exceeded, which is why this reports a fit.
        return MemoryFloor(floor, True)
    workers = max(1, int(workers))
    for groups, peak in _admissible_counts(input_bytes, points_per_record):
        if workers * peak <= budget:
            return MemoryFloor(groups, True)
    return MemoryFloor(MAX_CONTIG_GROUPS, False)


def group_floor(input_bytes, points_per_record=1):
    """Fewest contig batches that keep one query's scan under the cap.

    Independent of memory by design: this is how large a single
    grouping may get, not how much of the machine is free. See
    `MAX_GROUP_INPUT_BYTES`.

    :param input_bytes: Alignment bytes the job reads, or ``None``.
    :param points_per_record: Rows one record contributes to the
        grouping, as `quadrat.pileup.aggregate.Aggregate` answers it
        through its private ``_points_per_record``. The cap is divided
        by it, so a job contributing two
        rows per record is split as a job of twice its bytes would be.
    """

    if not input_bytes:
        return 1
    groups = -(-int(input_bytes) * int(points_per_record) // MAX_GROUP_INPUT_BYTES)
    return max(1, min(MAX_CONTIG_GROUPS, groups))


def _admissible_counts(input_bytes, points_per_record):
    """Each batch count a solver may choose, with its projected peak.

    The two solvers above answer different questions -- one maximises
    the worker count, the other minimises batches at a concurrency it
    is handed -- and searched the same candidates to do it: the same
    `range`, bounded below by `group_floor` and above by
    `MAX_CONTIG_GROUPS`, with `projected_peak` evaluated at each. The
    objectives differ and the search does not, so the search is here
    and the scoring stays with each solver.

    Both bounds are load-bearing and neither is arbitrary: below the
    scan floor a single grouping reads past `MAX_GROUP_INPUT_BYTES`
    however much memory is free, and above the ceiling the batches
    cost more wall clock than the headroom buys -- see
    `CONTIG_GROUP_GAIN`. A solver that got either bound wrong on its
    own was wrong in one place and right in the other, which is the
    drift this removes.

    Yields ``(groups, peak)`` in ascending batch count, so the peak
    falls across the sequence and the first candidate to fit is the
    fewest batches that do.
    """

    for groups in range(
        group_floor(input_bytes, points_per_record), MAX_CONTIG_GROUPS + 1
    ):
        yield groups, projected_peak(input_bytes, groups, points_per_record)


def _allocation_bytes():
    """Memory this process is allowed, or None if nothing states one.

    The counterpart of `sched_getaffinity` in `n_cpus`, for the
    other axis. A pileup is usually launched under something that
    restricts it, and a restriction on memory is not visible in
    `SC_PHYS_PAGES` any more than a restriction on cores is visible in
    `cpu_count`: a cgroup limit, a container `--memory` flag and a
    SLURM `--mem` allocation all leave the host's physical total
    untouched.

    Reading the machine instead of the allocation is worse here than
    it is for cores. An oversubscribed thread pool is slow; an
    oversubscribed memory budget is an OOM kill, and the issue's own
    Motivation is that under-estimating ends a run rather than slowing
    it. On a 256-core, 1007 GB host a 64 GB SLURM allocation yields a
    budget near 757 GB if only `sysconf` is consulted.

    Returns the smallest allowance any source states, because they are
    bounds rather than opinions: a job under both a cgroup and a SLURM
    allocation is held to both. Nothing here is available off Linux,
    where the host total is the only answer there is.

    The SLURM spellings accepted are a bare figure in MB, as SLURM
    documents it, and the same figure carrying a `K`, `M`, `G` or `T`
    suffix in either case -- see `_SLURM_MEM_UNITS`. A spelling outside
    that set is skipped rather than guessed at, which reads as no
    allocation from that variable.

    Both paths here are exercised against files written into a
    temporary directory and a patched environment rather than against a
    real cgroup or a real SLURM allocation, which the suite cannot
    create. What is pinned is the arithmetic and which source wins, not
    that the kernel publishes what this expects.
    """

    limits = []
    for path in (_CGROUP_V2_LIMIT, _CGROUP_V1_LIMIT):
        try:
            with open(path) as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if raw == "max":
            # v2 spells "no limit" as a word rather than by absence.
            continue
        try:
            limits.append(int(raw))
        except ValueError:
            continue
    for name in _SLURM_MEM_VARS:
        raw = os.environ.get(name)
        if not raw:
            continue
        raw = raw.strip()
        unit = _SLURM_MEM_UNITS.get(raw[-1:].upper())
        if unit is None:
            # No suffix, so the figure is the bare MB SLURM documents.
            unit = 1.0
        else:
            raw = raw[:-1]
        try:
            megabytes = int(raw)
        except ValueError:
            continue
        if name == "SLURM_MEM_PER_CPU":
            megabytes *= n_cpus()
        limits.append(int(megabytes * unit * 1024**2))
    # A limit of zero or less is not a limit anyone meant; v1 states
    # "unlimited" as a number near 2**63, which the min takes care of.
    limits = [limit for limit in limits if limit > 0]
    return min(limits) if limits else None
