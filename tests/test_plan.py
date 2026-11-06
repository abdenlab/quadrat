"""Tests for how many pileups to run at once, and in how many batches.

Relocated from the ported ``tests/test_pileup.py``, which held tests
for four modules. The assertions are the port's evidence and are
carried over unchanged; what changed is the docstring form, the names,
the phase comments, and the seam each test holds.

The seam is the substantive change. The ported tests read the machine
through ``n_cpus``, ``memory_budget`` and ``group_floor``,
which made each helper the subject rather than the behaviour. Here the
machine is pinned at the OS boundary by the ``machine`` fixture, the
budget is recomputed from public constants by `budget` below, and the
scan floor is read as ``plan_pileup(0, bytes)[1]`` -- the batch count
the planner reports when there is no work to do, which is the floor by
construction.

This module needs none of the ``pileup`` extra: `quadrat.pileup.plan`
is a leaf whose only import is `os`, so unlike the file these came from
it runs on a base install.
"""

import pytest

from quadrat.pileup import plan
from tests import budget


def test_projected_peak_should_sum_the_per_worker_cost_and_the_scanned_bytes():
    """Test the peak a single batch projects.

    Given:
        A known input size and one contig batch.
    When:
        The peak is projected.
    Then:
        It should be the fixed per-worker cost plus the input ratio
        times the bytes that batch scans.
    """

    # Arrange
    fixed = plan.PEAK_BYTES_PER_WORKER
    ratio = plan.PEAK_BYTES_PER_INPUT_BYTE
    job = 8 * 1024**3

    # Act & assert
    assert plan.projected_peak(job, 1) == fixed + int(job * ratio)


def test_projected_peak_should_divide_only_the_input_part_when_the_batch_count_rises():
    """Test that batches divide the input part in proportion.

    Given:
        A known input size at one batch and at four.
    When:
        The part above the fixed per-worker cost is compared.
    Then:
        It should fall in proportion rather than as a square root,
        because four batches scan a quarter of the bytes and hold a
        quarter of the distinct positions.
    """

    # Arrange
    fixed = plan.PEAK_BYTES_PER_WORKER
    ratio = plan.PEAK_BYTES_PER_INPUT_BYTE
    job = 8 * 1024**3

    # Act
    quarter = plan.projected_peak(job, 4) - fixed

    # Assert
    assert quarter == pytest.approx(int(job * ratio) // 4, rel=1e-6)


def test_projected_peak_should_stay_near_the_per_worker_cost_when_split_far():
    """Test the floor the batches cannot divide away.

    Given:
        A known input size split far past any real batch count.
    When:
        The peak is projected.
    Then:
        It should stay above the fixed per-worker cost and within a
        percent of it, that part being what a worker holds whatever it
        reads.

    The single multiple this replaced had nowhere to put the engine, the
    interpreter and the buffers a worker holds regardless, so it
    believed eight batches made all of it nearly three times smaller.
    Measured across the twelve arms in
    cherimoya's post-fix benchmark ladder, that part does not move.
    """

    # Arrange
    fixed = plan.PEAK_BYTES_PER_WORKER
    job = 8 * 1024**3

    # Act & assert
    assert plan.projected_peak(job, 1024) > fixed
    assert plan.projected_peak(job, 1024) < fixed * 1.01


def test_projected_peak_should_return_the_unknown_input_figure_when_no_size_is_given():
    """Test the projection for an input that cannot be sized.

    Given:
        No input size.
    When:
        The peak is projected.
    Then:
        It should be the measured unknown-input figure, which is better
        than guessing low and having the run die partway.
    """

    # Act & assert
    assert plan.projected_peak(None) == plan.UNKNOWN_INPUT_PEAK_BYTES


def test_default_partitions_should_divide_the_cores_among_the_workers(machine):
    """Test that the machine is divided rather than claimed by each worker.

    Given:
        A machine pinned to a known core count.
    When:
        The default partition count is taken for one, two and that many
        workers.
    Then:
        It should divide the cores among the workers, and never answer
        zero however many are asked for.

    The setting is process-global in DataFusion, so eight pileups each
    asking for every core would oversubscribe the thread pool eightfold.
    """

    # Arrange
    cpus = 64
    machine(cpus, 64 * 1024**3)

    # Act & assert
    assert plan.default_partitions(1) == cpus
    assert plan.default_partitions(cpus) == 1
    assert plan.default_partitions(2) == max(1, cpus // 2)
    assert plan.default_partitions(cpus * 100) == 1


def test_default_partitions_should_divide_by_one_when_the_worker_count_is_not_positive(
    machine,
):
    """Test the clamp below one worker.

    Given:
        A worker count of zero, and a negative one.
    When:
        The default partition count is taken.
    Then:
        It should answer as it does for one worker, a fan-out of none
        being a fan-out of one as far as dividing the cores goes.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act & assert
    assert plan.default_partitions(0) == plan.default_partitions(1)
    assert plan.default_partitions(-5) == plan.default_partitions(1)


def test_default_partitions_should_read_the_allocation_when_the_process_is_confined(
    machine,
):
    """Test that the core count is the restriction, not the machine.

    Given:
        A process confined to eight cores of a 256-core machine.
    When:
        The default partition count is taken for one worker, which is
        the whole allocation by construction.
    Then:
        It should report the confinement rather than what the machine
        has.

    A pileup is usually launched under something that confines it --
    SLURM, a cgroup, a container quota, ``taskset`` -- and ``cpu_count``
    reports the whole node whatever the allocation. Believing it sets
    ``target_partitions`` to a parallelism the job cannot have and lets
    the worker cap oversubscribe by the restriction's factor. Measured
    on js031 under ``taskset``, the harness went on recording 256 cores
    while confined to 32.
    """

    # Arrange
    machine(8, 64 * 1024**3)

    # Act & assert
    assert plan.default_partitions(1) == 8


def test_default_partitions_should_read_the_machine_when_affinity_is_unavailable(
    machine,
):
    """Test the fallback where the platform has no affinity call.

    Given:
        A 256-core machine whose ``sched_getaffinity`` does not exist.
    When:
        The default partition count is taken for one worker.
    Then:
        It should report the machine's core count, that being the only
        answer there is.
    """

    # Arrange
    machine(256, 64 * 1024**3, affinity=False)

    # Act & assert
    assert plan.default_partitions(1) == 256


def test_plan_pileup_should_plan_no_workers_when_there_are_no_jobs(machine):
    """Test the answer for an empty fan-out.

    Given:
        No jobs and a known input size.
    When:
        The fan-out is planned.
    Then:
        It should answer no workers and the scan floor those bytes ask
        for, the floor being a property of the bytes rather than of the
        job count.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act
    workers, groups = plan.plan_pileup(0, plan.MAX_GROUP_INPUT_BYTES + 1)

    # Assert
    assert workers == 0
    assert groups == 2


def test_plan_pileup_should_plan_no_workers_when_the_job_count_is_negative(machine):
    """Test the answer for a negative job count.

    Given:
        A job count below zero.
    When:
        The fan-out is planned.
    Then:
        It should answer no workers, as it does for zero, since zero or
        fewer jobs plans nothing.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act & assert
    assert plan.plan_pileup(-7, 1024**4)[0] == 0


def test_plan_pileup_should_cap_the_workers_at_the_job_count(machine):
    """Test the cap on the worker count.

    Given:
        Fewer jobs than the machine has cores.
    When:
        The fan-out is planned.
    Then:
        It should never ask for more workers than there are jobs.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act & assert
    assert plan.plan_pileup(0)[0] == 0
    assert plan.plan_pileup(1)[0] == 1
    assert plan.plan_pileup(3)[0] <= 3


def test_plan_pileup_should_count_batches_from_the_bytes_one_scan_may_hold(machine):
    """Test the scan floor at and either side of the cap.

    Given:
        Input sizes at, just over and far over the scan cap, and none at
        all.
    When:
        The batch floor is read from the plan.
    Then:
        It should be the fewest batches keeping one scan under the cap,
        and never more than the batch ceiling.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    cap = plan.MAX_GROUP_INPUT_BYTES

    # Act & assert
    assert plan.group_floor(None) == 1
    assert plan.group_floor(0) == 1
    assert plan.group_floor(cap) == 1
    assert plan.group_floor(cap + 1) == 2

    # The ceiling is real and the floor does not pretend otherwise: past
    # `MAX_CONTIG_GROUPS` batches it asks for no more than it can get.
    assert plan.group_floor(1024**4) == plan.MAX_CONTIG_GROUPS


def test_the_two_solvers_should_choose_from_the_same_batch_counts(machine):
    """Test that the scan floor binds both solvers where memory does not.

    Given:
        A 32 GiB input on a machine with enough memory to hold a single
        genome-wide aggregate of it four times over.
    When:
        `plan_pileup` and `memory_floor` are each asked for a batch
        count for those bytes.
    Then:
        Both should answer the input's scan floor, so a search whose
        lower bound drifts in one solver and not the other fails here.

    The two search the same candidates and score them differently, and
    for a while each expressed the search itself -- so a wrong bound
    was wrong in one solver, right in the other, and invisible to a
    suite that only ever asked one of them.

    The geometry is the whole of the test. On a machine where memory is
    the binding constraint, every candidate below the scan floor is
    refused on the memory axis anyway and a missing floor is
    unobservable; 512 GiB against a 408 GiB budget leaves one worker at
    one batch fitting comfortably at 50 GiB, so the only thing standing
    between the solvers and a single aggregate reading 32 GiB is the
    floor itself.
    """

    # Arrange
    machine(8, 512 * 1024**3)
    size = 4 * plan.MAX_GROUP_INPUT_BYTES
    floor = plan.group_floor(size)

    # Act & assert
    assert floor == 4
    assert plan.plan_pileup(1, size).contig_groups == floor
    assert plan.memory_floor(size, 1).groups == floor


def test_plan_pileup_should_admit_fewer_workers_when_the_input_is_larger(machine):
    """Test that the memory bound tracks the input.

    Given:
        A 4 GB input and a 256 MB one, planned on the same 64-core,
        64 GB machine.
    When:
        Each fan-out is planned.
    Then:
        It should admit fewer workers for the larger input, buy batches
        to win most of them back, and leave the small input unsplit.

    A pileup peaks at roughly three times the bytes it reads, so what is
    safe for a 300 MB alignment is fatal for a 4 GB one. The constant
    this replaced said 2 GB regardless, which on a 64 GB machine
    admitted 32 workers against a real need of about 380 GB.

    The bound is now something to spend rather than only to obey, and
    the numbers are the planner's own documented worked example: one
    batch of a 4 GB input admits four pileups, and splitting it five
    ways brings the peak to 3.2 GB and buys back twelve.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act
    big, big_groups = plan.plan_pileup(32, 4 * 1024**3)
    small, small_groups = plan.plan_pileup(32, 256 * 1024**2)

    # Assert
    assert big < small, "a larger input must not permit more workers"
    assert (big, big_groups) == (12, 5)
    assert plan.projected_peak(4 * 1024**3, big_groups) == pytest.approx(
        3.2 * 1024**3, rel=1e-6
    )

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
    assert small * plan.PEAK_BYTES_PER_WORKER <= budget(
        plan, 64 * 1024**3, plan.n_cpus()
    )


def test_plan_pileup_should_bound_the_fan_out_conservatively_when_no_size_is_given(
    machine,
):
    """Test the fan-out for inputs that cannot be sized.

    Given:
        Thirty-two jobs whose alignments cannot be sized, on a 64-core,
        64 GB machine.
    When:
        The fan-out is planned.
    Then:
        It should bound the workers from the measured unknown-input
        figure and still split the genome, rather than guessing the peak
        is smaller than it is.
    """

    # Arrange
    machine(64, 64 * 1024**3)

    # Act
    workers, groups = plan.plan_pileup(32)

    # Assert
    # Twelve GB per worker ungrouped, so four at once on a 64 GB
    # machine; batches bring that down without ever guessing the peak
    # is smaller than it is.
    assert 4 <= workers < 32
    assert groups > 1


def test_plan_pileup_should_admit_fewer_workers_when_the_machine_has_more_cores(
    machine,
):
    """Test that the engine's machine-wide cost is spent once.

    Given:
        Two 256 GB machines differing only in core count, and the same
        8 GB input across sixteen jobs.
    When:
        Each fan-out is planned.
    Then:
        It should admit fewer workers on the machine with more cores,
        because what the engine holds per core comes off the shared
        budget rather than being charged to each worker.

    Measured by confining the process tree with ``taskset`` and changing
    nothing else: at twelve workers and eight partitions the peak goes
    50.2, 62.4, 70.0 and 73.8 GB at 32, 64, 128 and 256 cores. Holding
    the cores and sweeping ``target_partitions`` 4 to 32 instead moves it
    72.0 to 76.8 GB, inside the replicate spread -- so the cost follows
    the cores and not the partitions, and not the fan-out either.

    The *budget* falls monotonically with the core count and is checked
    across all three machines. The *worker* count is checked between 16
    and 64 cores only, and deliberately: at 256 cores it rises again to
    16, because the smaller budget lowers the gain gate enough to admit
    a higher batch count. Generalizing this into "more cores never
    admits more workers" states something false of the planner.
    """

    # Arrange
    ram = 256 * 1024**3
    job = 8 * 1024**3

    # Act
    machine(16, ram)
    few_cores, small_budget = (
        plan.plan_pileup(16, job)[0],
        budget(plan, ram, plan.n_cpus()),
    )
    machine(64, ram)
    many_cores, mid_budget = (
        plan.plan_pileup(16, job)[0],
        budget(plan, ram, plan.n_cpus()),
    )
    machine(256, ram)
    large_budget = budget(plan, ram, plan.n_cpus())

    # Assert
    assert many_cores < few_cores
    assert large_budget < mid_budget < small_budget
    assert small_budget - large_budget == 240 * plan.PEAK_BYTES_PER_CORE


def test_memory_budget_should_keep_room_for_a_worker_when_the_cores_outweigh_it(
    machine,
):
    """Test the budget on a machine whose cores cost more than it has.

    Given:
        An 8 GB allocation and 4096 visible cores, so the per-core term
        is two orders of magnitude larger than the headroom.
    When:
        The budget is computed.
    Then:
        It should still be large enough to hold one worker's fixed
        cost, rather than collapsing to the floor.

    The subtraction was unguarded, and the floor turned every such
    machine into a one-byte budget -- which reads as a real bound
    rather than as "unknowable", so nothing fit at any batch count. A
    budget half the headroom is an over-reservation; a budget of one
    byte is a wrong answer.
    """

    # Arrange
    machine(4096, 8 * 1024**3)

    # Act
    allowance = budget(plan, 8 * 1024**3, plan.n_cpus())

    # Assert
    assert allowance > plan.PEAK_BYTES_PER_WORKER


def test_plan_pileup_should_split_the_genome_when_a_memory_cap_meets_many_cores(
    machine,
):
    """Test the plan inside a memory-capped container on a large host.

    Given:
        An 8 GB cgroup allocation on a 64-core host with no cpuset, and
        a single 3.9 GB alignment.
    When:
        One pileup is planned.
    Then:
        It should answer more than one contig batch, because one batch
        projects about 7.9 GB against the 8 GB the container has.

    The two axes are read from different sources -- affinity sees the
    whole host while the allocation is capped -- so this regime is
    reachable with ample real memory. It is also the regime the
    module's own asymmetry is inverted in: the planner used to answer
    the batch count with the *highest* peak here.
    """

    # Arrange
    machine(cores=64, ram_bytes=256 * 1024**3, cgroup=8 * 1024**3)

    # Act
    plan_for_one = plan.plan_pileup(1, 3_900_000_000)

    # Assert
    assert plan_for_one.workers == 1
    assert plan_for_one.contig_groups > 1
    assert plan_for_one.workers * plan.projected_peak(
        3_900_000_000, plan_for_one.contig_groups
    ) <= budget(plan, 8 * 1024**3, plan.n_cpus())


def test_plan_pileup_should_keep_one_batch_when_splitting_buys_a_marginal_gain(machine):
    """Test the gain gate on spending a batch.

    Given:
        An input small enough that the fixed per-worker cost dominates.
    When:
        The batch count is planned.
    Then:
        It should stay at one, because splitting would buy a few percent
        of concurrency for a quarter more wall clock.

    The part of the peak the batches divide can always be driven to
    nearly nothing, so without this the planner takes every batch it is
    allowed on inputs where none of them help.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    size = 256 * 1024**2
    free = budget(plan, 64 * 1024**3, plan.n_cpus())

    # Act
    workers, groups = plan.plan_pileup(32, size)

    # Assert
    assert groups == 1

    # Not because splitting was unavailable -- it would have admitted
    # more workers, just not enough more to be worth the wall clock.
    more = 0
    for count in range(2, plan.MAX_CONTIG_GROUPS + 1):
        fits = free // plan.projected_peak(size, count)
        more = max(more, min(32, fits))
    assert workers < more < workers * plan.CONTIG_GROUP_GAIN


def test_plan_pileup_should_spend_batches_when_they_buy_the_required_gain(machine):
    """Test that a real gain is taken.

    Given:
        A 4 GB input, where one batch admits far fewer workers than
        several.
    When:
        The fan-out is planned.
    Then:
        It should split the genome and admit at least the gain multiple
        of what one batch would have allowed.
    """

    # Arrange
    machine(64, 64 * 1024**3)
    size = 4 * 1024**3

    # Act
    big, big_groups = plan.plan_pileup(32, size)

    # Assert
    whole = budget(plan, 64 * 1024**3, plan.n_cpus()) // plan.projected_peak(size, 1)
    assert big_groups > 1
    assert big >= whole * plan.CONTIG_GROUP_GAIN


def test_plan_pileup_should_split_a_large_job_however_large_the_machine_is(machine):
    """Test that the scan cap binds independently of free memory.

    Given:
        One 21.2 GB input on a 256-core, 1007 GB machine with memory to
        spare.
    When:
        The fan-out is planned.
    Then:
        It should split the genome anyway, the bound that matters being
        how much one query scans rather than how much memory is free.

    Solving from headroom alone answers one batch here, which is how a
    21.2 GB alignment came to be read as a single aggregate over
    291,063,148 records.
    """

    # Arrange
    # js031, where the overflow was found: 256 cores and 1007 GB, which
    # is enough memory that the planner never had to split anything.
    machine(256, 1007 * 1024**3)

    # Act & assert
    assert plan.plan_pileup(1, int(21.2 * 1024**3)) == (1, 3)

    # And not otherwise: batches cost 25% of the pileup wall clock at
    # the sizes the ladder measured, so the cap must stay clear of them.
    assert plan.plan_pileup(1, 2 * 1024**3)[1] == 1


def test_plan_pileup_should_spend_batches_until_one_worker_fits_the_budget(machine):
    """Test the memory half of the batch solve, for a single job.

    Given:
        One pileup on a 32-core, 16 GB machine whose peak at the scan
        floor exceeds the budget.
    When:
        The fan-out is planned.
    Then:
        It should spend batches past the floor until the projected peak
        fits.

    The scan cap and the memory budget are separate bounds and either
    can be the one that asks for more batches. This is the memory half,
    which a single job could not reach while the worker count was
    clamped to one inside the search: "one worker fits" and "not even
    one fits" scored the same, the first candidate took the answer, and
    the batch count stayed at the scan floor however far over budget it
    projected. Measured at 1.18x here and 2.83x on the smaller machine
    below.
    """

    # Arrange
    machine(32, 16 * 1024**3)
    size = 4 * 1024**3
    free = budget(plan, 16 * 1024**3, plan.n_cpus())
    floor = plan.group_floor(size)
    assert plan.projected_peak(size, floor) > free

    # Act
    workers, groups = plan.plan_pileup(1, size)

    # Assert
    assert groups > floor
    assert workers * plan.projected_peak(size, groups) <= free


def test_memory_floor_should_report_the_count_and_whether_it_fits(machine):
    """Test both elements of what the batch-count solver answers.

    Given:
        An 8-core, 64 GB machine and a 4 GB job at two workers.
    When:
        The batch count is solved.
    Then:
        It should answer a count readable by name, and report that the
        count fits.
    """

    # Arrange
    machine(8, 64 * 1024**3)

    # Act
    floor = plan.memory_floor(4 * 1024**3, 2)

    # Assert
    assert floor.fits
    assert floor.groups == min(
        groups
        for groups in range(1, plan.MAX_CONTIG_GROUPS + 1)
        if 2 * plan.projected_peak(4 * 1024**3, groups)
        <= budget(plan, 64 * 1024**3, plan.n_cpus())
    )


def test_memory_floor_should_report_no_fit_when_no_count_in_range_works(machine):
    """Test the answer that is the best available failure.

    Given:
        An 8-core, 8 GB machine and a 32 GB job that fits at no batch
        count.
    When:
        The batch count is solved.
    Then:
        It should answer the largest count and report that it does not
        fit.

    The count alone cannot say which of the two it is --
    ``MAX_CONTIG_GROUPS`` is both the count a concurrency genuinely
    needs and the count returned when nothing works -- and a caller
    that warns on the second has to be able to tell them apart.
    """

    # Arrange
    machine(8, 8 * 1024**3)

    # Act
    floor = plan.memory_floor(32 * 1024**3, 1)

    # Assert
    assert floor == (plan.MAX_CONTIG_GROUPS, False)


def test_memory_floor_should_report_a_fit_when_the_budget_is_unknowable(machine):
    """Test the answer when there is nothing to solve against.

    Given:
        A machine whose physical memory cannot be read.
    When:
        The batch count is solved.
    Then:
        It should answer the scan floor and report a fit, because an
        unknown bound is not a bound that was exceeded.
    """

    # Arrange
    machine(8, None)

    # Act
    floor = plan.memory_floor(32 * 1024**3, 1)

    # Assert
    assert floor == (plan.group_floor(32 * 1024**3), True)


def test_memory_floor_should_solve_against_a_budget_it_is_handed(machine, mocker):
    """Test that a passed budget replaces reading the machine.

    Given:
        A budget passed by the caller, and a machine read that raises
        if it is attempted.
    When:
        The batch count is solved.
    Then:
        It should answer from the passed budget.

    A fan-out solves for every job in a set, and the budget comes from
    two cgroup files, the SLURM environment, ``sysconf`` and the
    affinity mask -- none of which can change during the loop, and all
    of which were re-read per job. The budget is also mutable OS state,
    so two reads inside one plan are not guaranteed to agree.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    mocker.patch.object(
        plan, "memory_budget", side_effect=AssertionError("budget re-read")
    )

    # Act
    floor = plan.memory_floor(
        4 * 1024**3, 2, budget=budget(plan, 64 * 1024**3, plan.n_cpus())
    )

    # Assert
    assert floor == plan.MemoryFloor(1, True)


def test_plan_pileup_should_solve_against_a_budget_it_is_handed(machine, mocker):
    """Test that a passed budget replaces reading the machine here too.

    Given:
        A budget passed by the caller, and a machine read that raises
        if it is attempted.
    When:
        The fan-out is planned.
    Then:
        It should answer from the passed budget.

    The counterpart of the `memory_floor` test above, and the residual
    the per-job fix did not name: `pileup_many` reads the budget once
    and said so, then called this function, which read it again -- so
    the auto-planned path paid two full reads per fan-out and solved
    the worker count against one figure while checking every per-job
    floor against the other.
    """

    # Arrange
    machine(8, 64 * 1024**3)
    free = budget(plan, 64 * 1024**3, plan.n_cpus())
    expected = plan.plan_pileup(4, 4 * 1024**3)
    mocker.patch.object(
        plan, "memory_budget", side_effect=AssertionError("budget re-read")
    )

    # Act
    answer = plan.plan_pileup(4, 4 * 1024**3, budget=free)

    # Assert
    assert answer == expected


def test_plan_pileup_should_take_the_lowest_peak_when_no_batch_count_fits(machine):
    """Test which arrangement is handed back when none of them fits.

    Given:
        One 32 GB job on an 8-core, 8 GB machine that fits at no batch
        count.
    When:
        The fan-out is planned.
    Then:
        It should answer the largest batch count and one worker, which
        is the lowest peak available rather than the highest.

    Over budget at every count, how far over is the only thing left to
    choose, and the module's asymmetry decides it: too low a projection
    ends a run while too high costs wall clock. The scan floor was the
    answer here -- 14.00 GB against 8.00 GB at eight batches -- which
    inverted that asymmetry at exactly the allocations where memory is
    scarcest. It is also what `memory_floor` already answered in this
    situation, so the two solvers disagreed.
    """

    # Arrange
    machine(8, 8 * 1024**3)
    size = 32 * 1024**3
    free = budget(plan, 8 * 1024**3, plan.n_cpus())
    assert all(
        plan.projected_peak(size, groups) > free
        for groups in range(1, plan.MAX_CONTIG_GROUPS + 1)
    )

    # Act
    planned = plan.plan_pileup(1, size)

    # Assert
    assert planned == (1, plan.MAX_CONTIG_GROUPS)
    assert plan.projected_peak(size, planned.contig_groups) == min(
        plan.projected_peak(size, groups)
        for groups in range(plan.group_floor(size), plan.MAX_CONTIG_GROUPS + 1)
    )


def test_plan_pileup_should_plan_against_the_headroom_fraction_of_memory(machine):
    """Test that the headroom is held back rather than planned against.

    Given:
        A 64-core, 64 GB machine and an input whose peak the memory
        bound is what limits.
    When:
        The fan-out is planned.
    Then:
        It should keep the workers' projected total inside the headroom
        fraction of memory, leaving the rest unplanned.

    `plan.MEMORY_HEADROOM` is otherwise referenced by no test: the one
    assertion that could have caught it compares two budgets to each
    other, and the headroom cancels. Nothing would have failed if it
    became 1.0 and the planner started handing out the whole machine.
    """

    # Arrange
    total = 64 * 1024**3
    machine(64, total)
    size = 4 * 1024**3

    # Act
    workers, groups = plan.plan_pileup(32, size)

    # Assert
    assert workers * plan.projected_peak(size, groups) <= total * plan.MEMORY_HEADROOM
    assert plan.MEMORY_HEADROOM < 1


def test_plan_pileup_should_guess_four_workers_when_the_memory_total_is_unknowable(
    machine,
):
    """Test the fan-out where the machine's memory cannot be read.

    Given:
        A machine whose memory total cannot be read, and a hundred jobs.
    When:
        The fan-out is planned.
    Then:
        It should guess four workers and keep the scan floor, four being
        what this has always guessed and the floor being the one thing
        still knowable without a memory figure.

    The only statement in the module that the rest of the suite never
    executes.
    """

    # Arrange
    machine(64, None)
    size = 32 * 1024**3

    # Act & assert
    assert plan.plan_pileup(100, size) == (4, plan.group_floor(size))
    assert plan.plan_pileup(2) == (2, 1)


def test_plan_pileup_should_cap_the_guess_at_the_job_count_when_memory_is_unknowable(
    machine,
):
    """Test that the guess respects the job count.

    Given:
        The same unreadable machine and three jobs.
    When:
        The fan-out is planned.
    Then:
        It should cap the guess at the job count rather than ask for
        four workers to run three jobs.
    """

    # Arrange
    machine(64, None)

    # Act & assert
    assert plan.plan_pileup(3, 32 * 1024**3)[0] == 3


def test_plan_pileup_should_read_the_cgroup_limit_rather_than_the_host_total(machine):
    """Test the plan on a host whose memory the process may not all have.

    Given:
        A 256 GB host and a cgroup v2 limit of 16 GB.
    When:
        A fan-out is planned.
    Then:
        It should be the plan a 16 GB host gets, not the plan the 256 GB
        host gets.
    """

    # Arrange
    machine(cores=8, ram_bytes=16 * 1024**3)
    allowance = plan.plan_pileup(64, 4 * 1024**3)
    machine(cores=8, ram_bytes=256 * 1024**3)
    host = plan.plan_pileup(64, 4 * 1024**3)
    # The two have to disagree or the assertion below holds for the
    # wrong reason.
    assert allowance != host

    # Act
    machine(cores=8, ram_bytes=256 * 1024**3, cgroup=16 * 1024**3)
    confined = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert confined == allowance


def test_plan_pileup_should_ignore_a_cgroup_limit_of_max(machine):
    """Test the plan under a cgroup that states no limit.

    Given:
        A 256 GB host whose cgroup v2 limit file reads ``max``.
    When:
        A fan-out is planned.
    Then:
        It should be the plan the host's own total gives, because
        ``max`` is how v2 spells the absence of a limit.
    """

    # Arrange
    machine(cores=8, ram_bytes=256 * 1024**3)
    host = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(cores=8, ram_bytes=256 * 1024**3, cgroup="max")
    unlimited = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert unlimited == host


def test_plan_pileup_should_read_the_cgroup_v1_limit_when_there_is_no_v2_file(machine):
    """Test the plan on a host running cgroup v1.

    Given:
        A 256 GB host with no v2 limit file and a v1 limit of 16 GB.
    When:
        A fan-out is planned.
    Then:
        It should be the plan a 16 GB host gets.
    """

    # Arrange
    machine(cores=8, ram_bytes=16 * 1024**3)
    allowance = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(cores=8, ram_bytes=256 * 1024**3, cgroup_v1=16 * 1024**3)
    confined = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert confined == allowance


def test_plan_pileup_should_read_the_slurm_node_allocation(machine):
    """Test the plan under a SLURM allocation stated per node.

    Given:
        A 256 GB host and ``SLURM_MEM_PER_NODE`` of 16384, in MB as
        SLURM states it.
    When:
        A fan-out is planned.
    Then:
        It should be the plan a 16 GB host gets.
    """

    # Arrange
    machine(cores=8, ram_bytes=16 * 1024**3)
    allowance = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(
        cores=8,
        ram_bytes=256 * 1024**3,
        slurm={"SLURM_MEM_PER_NODE": 16 * 1024},
    )
    confined = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert confined == allowance


def test_plan_pileup_should_read_a_suffixed_slurm_allocation(machine):
    """Test the plan under an allocation SLURM states with a unit.

    Given:
        A 256 GB host and ``SLURM_MEM_PER_NODE`` of ``"16G"``.
    When:
        A fan-out is planned.
    Then:
        It should be the plan the same figure in bare MB gets.

    The suffix was stripped and its multiplier discarded, so ``"16G"``
    read as 16 MB -- a cluster fan-out collapsing to one worker at the
    scan floor on an allocation three orders of magnitude larger than
    the one it planned against.
    """

    # Arrange
    machine(cores=8, ram_bytes=256 * 1024**3, slurm={"SLURM_MEM_PER_NODE": 16 * 1024})
    bare = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(cores=8, ram_bytes=256 * 1024**3, slurm={"SLURM_MEM_PER_NODE": "16G"})
    suffixed = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert suffixed == bare


def test_plan_pileup_should_read_a_slurm_allocation_stated_in_kilobytes(machine):
    """Test the unit that is smaller than the one the figure is in.

    Given:
        A 256 GB host and ``SLURM_MEM_PER_NODE`` of ``"131072K"``,
        which is 128 MB.
    When:
        A fan-out of sixteen 4 GB jobs is planned.
    Then:
        It should answer one worker, because 128 MB admits none.

    ``K`` is the suffix the discarded multiplier was wrong about in the
    unsafe direction: stripping it read this allocation as 131072 MB,
    a thousandfold over-read on a cluster, where every other suffix
    under-read. It is also why the product is taken in bytes rather
    than in megabytes -- a sub-MB allocation computed in integer
    megabytes floors to zero, and the ``limit > 0`` filter then drops
    it as though nothing had been stated.
    """

    # Arrange
    machine(cores=8, ram_bytes=256 * 1024**3)
    unconfined = plan.plan_pileup(16, 4 * 1024**3)

    # Act
    machine(cores=8, ram_bytes=256 * 1024**3, slurm={"SLURM_MEM_PER_NODE": "131072K"})
    confined = plan.plan_pileup(16, 4 * 1024**3)

    # Assert
    assert unconfined.workers > 1
    assert confined.workers == 1


def test_plan_pileup_should_multiply_the_slurm_per_cpu_allocation_by_the_cores(machine):
    """Test the plan under a SLURM allocation stated per CPU.

    Given:
        A 256 GB host, eight cores, and ``SLURM_MEM_PER_CPU`` of 2048 MB.
    When:
        A fan-out is planned.
    Then:
        It should be the plan a 16 GB host gets, because the per-CPU
        figure is an allowance per core rather than in total.
    """

    # Arrange
    machine(cores=8, ram_bytes=16 * 1024**3)
    allowance = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(
        cores=8,
        ram_bytes=256 * 1024**3,
        slurm={"SLURM_MEM_PER_CPU": 2 * 1024},
    )
    confined = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert confined == allowance


def test_plan_pileup_should_hold_to_the_smallest_allowance_any_source_states(machine):
    """Test the plan when a cgroup and a SLURM variable disagree.

    Given:
        A 256 GB host under a 32 GB cgroup limit and a 16 GB SLURM
        allocation.
    When:
        A fan-out is planned.
    Then:
        It should be the plan a 16 GB host gets, because the two are
        bounds rather than opinions and the job is held to both.
    """

    # Arrange
    machine(cores=8, ram_bytes=16 * 1024**3)
    allowance = plan.plan_pileup(64, 4 * 1024**3)

    # Act
    machine(
        cores=8,
        ram_bytes=256 * 1024**3,
        cgroup=32 * 1024**3,
        slurm={"SLURM_MEM_PER_NODE": 16 * 1024},
    )
    confined = plan.plan_pileup(64, 4 * 1024**3)

    # Assert
    assert confined == allowance


def test_plan_pileup_should_name_both_halves_of_the_plan(machine):
    """Test that the plan is readable by name as well as by position.

    Given:
        A machine the planner can size a fan-out for.
    When:
        A plan is made.
    Then:
        Its two elements should be reachable as ``workers`` and
        ``contig_groups``, and it should still compare equal to the
        pair it used to be.

    The pair was read by position at three call sites and the summary
    had to spell the shape out, which is what positional indexing
    becoming a published read protocol looks like. Tuple compatibility
    is kept deliberately so the assertions written against the pair did
    not have to move.
    """

    # Arrange
    machine(8, 64 * 1024**3)

    # Act
    plan_made = plan.plan_pileup(4, 4 * 1024**3)

    # Assert
    assert plan_made.workers == plan_made[0]
    assert plan_made.contig_groups == plan_made[1]
    assert plan_made == (plan_made.workers, plan_made.contig_groups)
