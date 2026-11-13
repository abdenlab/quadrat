"""Tests for the samplers: the draw, the plan, and the rank's share of it.

What the mix sampler promises is each peak drawn exactly once per
epoch, exact slot proportions, a plan that is a pure function of the
seed and the epoch, and an epoch that advances on iteration and can be
set outright. That the plan is still the one every earlier run drew,
generator call for generator call, is what the golden digests pin; they
are literal values and are never to be regenerated to make a test pass.

What the sharded sampler promises is a partition: the ranks together
read exactly what one rank would, step for step.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import hashlib
import pickle
from types import MappingProxyType

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat import sampling
from tests import CONFIG_IDS, GOLDEN_CONFIGS

# The plan digests, recorded over `GOLDEN_CONFIGS` and two epochs each.
# Carried as literals: a value that stops matching means the draw
# sequence moved, and the fix is in the sampler.
GOLDEN_PLANS = MappingProxyType(
    {
        "baseline": "0f7a75a773e2455c3f85558d9faaf15a31947a934e2d73a28202400e74003bfa",
        "grouped_ctl": "cb6ec38d60dafd83b6f1689a14e61c64731be6db4931cd04720468b62fc0bf3e",
        "no_jitter": "b8b257f2efdbeeedc245fbd28218f911e98f94225693254932c4ffb85c983253",
        "no_negatives": "52e448caacbd0204d8e4a3f6cf49b00b7be4e50080e78412854a5fdd48d8e23f",
        "no_rc": "f4b187c0ba524b6e188ec9a259e9fb92711815f6cc1a6a2050de43a7fff7489c",
        "sequential": "20269d857ab581a88eea3e70828393c51b9c0e0bcd640f0a495e7593a468f0b7",
    }
)


# Sampler arguments over every branch of the plan: with and without
# negatives, jitter, mirroring and shuffling.
sampler_arguments = st.fixed_dictionaries(
    {
        "n_peaks": st.integers(min_value=0, max_value=40),
        "n_negatives": st.integers(min_value=1, max_value=12),
        "negative_ratio": st.sampled_from([0.0, 0.1, 0.5, 1.0, 2.5]),
        "max_jitter": st.integers(min_value=0, max_value=6),
        "reverse_complement": st.booleans(),
        "shuffle": st.booleans(),
        "random_state": st.integers(min_value=0, max_value=2**32),
    }
)


# An epoch size, a global batch and a world that divides it.
shard_geometries = st.builds(
    lambda n, local, world: (n, local * world, world),
    st.integers(min_value=0, max_value=120),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=4),
)


def make_sampler(config_id, **overrides):
    """Return a mix sampler under one golden configuration."""

    return sampling.PeakNegativeMixSampler(
        **dict(GOLDEN_CONFIGS[config_id], **overrides)
    )


def draws(plan):
    """Return every draw of a plan as plain tuples."""

    return [tuple(draw) for draw in plan]


def plan_digest(config, n_epochs=2):
    """Return the sha256 of a configuration's draw plan over its first epochs."""

    sampler = sampling.PeakNegativeMixSampler(**config)
    digest = hashlib.sha256()
    for epoch in range(n_epochs):
        plan = sampler.plan(epoch)
        for name, array in (
            (b"is_peak", plan.is_peak),
            (b"index", plan.index),
            (b"jitter", plan.jitter),
            (b"reverse_complement", plan.reverse_complement),
        ):
            digest.update(name)
            digest.update(numpy.ascontiguousarray(array).astype(numpy.int64).tobytes())
    return digest.hexdigest()


class TestLocusDraw:
    """The one decision a sampler hands a dataset."""

    def test___new___should_expose_the_four_decisions_by_name(self):
        """Test the fields of a draw.

        Given:
            A locus set, a row, a jitter and an orientation.
        When:
            A draw is made of them.
        Then:
            It should answer each by name and unpack in that order.
        """

        # Act
        draw = sampling.LocusDraw(True, 3, 2, False)

        # Assert
        assert (draw.is_peak, draw.index, draw.jitter, draw.reverse_complement) == (
            True,
            3,
            2,
            False,
        )
        assert tuple(draw) == (True, 3, 2, False)

    def test_pickling_should_restore_an_equal_draw(self):
        """Test that a draw crosses a process boundary intact.

        Given:
            A draw.
        When:
            It is pickled and restored.
        Then:
            It should be an equal `LocusDraw`, which is how a worker
            tells one from an ``(epoch, index)`` pair.
        """

        # Arrange
        draw = sampling.LocusDraw(False, 7, 0, True)

        # Act
        restored = pickle.loads(pickle.dumps(draw))

        # Assert
        assert restored == draw
        assert type(restored) is sampling.LocusDraw


class TestEpochPlan:
    """One epoch's draws, as arrays and as draws."""

    def test___len___should_count_the_slots(self):
        """Test the length of a plan.

        Given:
            A plan whose arrays hold three slots, over two peaks.
        When:
            It is measured.
        Then:
            It should report three, the slot count, not the peak count.
        """

        # Arrange
        plan = sampling.EpochPlan(
            0,
            numpy.array([1, 0]),
            numpy.array([True, False, True]),
            numpy.array([1, 2, 0]),
            numpy.array([0, 0, 0]),
            numpy.array([False, False, False]),
        )

        # Act
        length = len(plan)

        # Assert
        assert length == 3

    def test___getitem___should_answer_plain_python_scalars(self):
        """Test what indexing a plan returns.

        Given:
            A plan over numpy arrays.
        When:
            A slot is indexed.
        Then:
            It should be a `LocusDraw` of that slot's values as plain
            `bool` and `int`, not numpy scalars.
        """

        # Arrange
        plan = sampling.EpochPlan(
            4,
            numpy.array([1, 0]),
            numpy.array([True, False, True]),
            numpy.array([1, 5, 0], dtype=numpy.int64),
            numpy.array([3, 0, 2], dtype=numpy.int64),
            numpy.array([False, True, True]),
        )

        # Act
        draw = plan[1]

        # Assert
        assert draw == sampling.LocusDraw(False, 5, 0, True)
        assert [type(value) for value in draw] == [bool, int, int, bool]

    def test___iter___should_yield_every_slot_in_order(self):
        """Test iterating a plan.

        Given:
            A plan of three slots.
        When:
            It is iterated.
        Then:
            It should yield the three draws in slot order.
        """

        # Arrange
        plan = sampling.EpochPlan(
            0,
            numpy.array([0, 1]),
            numpy.array([True, True, False]),
            numpy.array([0, 1, 4]),
            numpy.array([1, 2, 0]),
            numpy.array([False, False, True]),
        )

        # Act
        yielded = list(plan)

        # Assert
        assert yielded == [plan[0], plan[1], plan[2]]
        assert yielded[2] == sampling.LocusDraw(False, 4, 0, True)


class TestPeakNegativeMixSampler:
    """The per-epoch draw plan and the epoch it is at."""

    @pytest.mark.parametrize(
        "negative_ratio,max_jitter,n_negatives,match",
        [
            (0.5, -1, 4, "max_jitter"),
            (-0.1, 0, 4, "negative_ratio"),
            (0.5, 0, 0, "no negatives"),
        ],
    )
    def test_validate_arguments_should_raise_what_the_constructor_refuses(
        self, negative_ratio, max_jitter, n_negatives, match
    ):
        """Test the refusals a caller can make before it has a peak count.

        Given:
            A negative jitter bound, a negative ratio, and a positive
            ratio over no negatives.
        When:
            The arguments are validated without a sampler.
        Then:
            Each should raise `ValueError` naming what is wrong, as the
            constructor does.
        """

        # Act & assert
        with pytest.raises(ValueError, match=match):
            sampling.PeakNegativeMixSampler.validate_arguments(
                negative_ratio, max_jitter, n_negatives
            )

    def test_validate_arguments_should_accept_a_ratio_when_the_count_is_unknown(self):
        """Test a negative count not known yet.

        Given:
            A positive ratio and no negative count.
        When:
            The arguments are validated.
        Then:
            It should raise nothing, the one refusal that needs the
            count being skipped.
        """

        # Act
        result = sampling.PeakNegativeMixSampler.validate_arguments(0.5, 2)

        # Assert
        assert result is None

    def test___init___should_raise_when_max_jitter_is_negative(self):
        """Test the refusal of a negative jitter bound.

        Given:
            A negative ``max_jitter``.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError` naming the argument.
        """

        # Act & assert
        with pytest.raises(ValueError, match="max_jitter"):
            sampling.PeakNegativeMixSampler(8, 4, max_jitter=-1)

    def test___init___should_raise_when_negative_ratio_is_negative(self):
        """Test the refusal of a negative ratio.

        Given:
            A ``negative_ratio`` below zero.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError` naming the argument.
        """

        # Act & assert
        with pytest.raises(ValueError, match="negative_ratio"):
            sampling.PeakNegativeMixSampler(8, 4, negative_ratio=-0.1)

    def test___init___should_raise_when_negatives_are_wanted_and_none_exist(self):
        """Test the refusal of a ratio over an empty negative set.

        Given:
            A positive ``negative_ratio`` and no negatives.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError`, rather than plan draws of row
            zero of nothing.
        """

        # Act & assert
        with pytest.raises(ValueError, match="no negatives"):
            sampling.PeakNegativeMixSampler(8, 0, negative_ratio=0.5)

    def test___init___should_accept_no_negatives_when_the_ratio_is_zero(self):
        """Test a peak-only sampler.

        Given:
            A ``negative_ratio`` of zero and no negatives.
        When:
            The sampler is constructed and an epoch planned.
        Then:
            It should plan one slot per peak, every one a peak.
        """

        # Act
        sampler = sampling.PeakNegativeMixSampler(8, 0, negative_ratio=0.0)

        # Assert
        assert len(sampler) == 8
        assert sampler.plan(0).is_peak.all()

    def test___init___should_capture_one_seed_when_random_state_is_none(self):
        """Test that an unseeded sampler is still one sampler.

        Given:
            A sampler constructed with no seed.
        When:
            It is pickled, and the original and the copy plan an epoch.
        Then:
            Both should plan the same draws, because the seed is drawn
            once at construction and travels with the sampler.
        """

        # Arrange
        sampler = sampling.PeakNegativeMixSampler(
            12, 4, negative_ratio=0.5, max_jitter=3, reverse_complement=True
        )

        # Act
        restored = pickle.loads(pickle.dumps(sampler))

        # Assert
        assert draws(restored.plan(1)) == draws(sampler.plan(1))

    def test___len___should_count_the_peaks_and_the_negative_slots(self):
        """Test the epoch length.

        Given:
            Twenty-four peaks at a ratio of one half, and ten at a
            ratio whose product is not whole.
        When:
            Each sampler's length is taken.
        Then:
            It should be the peaks plus the ratio's share of them,
            rounded down.
        """

        # Act & assert
        assert len(make_sampler("baseline")) == 36
        assert len(sampling.PeakNegativeMixSampler(10, 5, negative_ratio=0.25)) == 12

    @pytest.mark.parametrize("config_id", CONFIG_IDS)
    def test_plan_should_match_the_golden_digest(self, config_id):
        """Test that the draw sequence is the recorded one.

        Given:
            A configuration that exercises one branch of the plan's
            generator calls.
        When:
            Its plans for the first two epochs are hashed.
        Then:
            It should equal the digest recorded for that
            configuration, so no call was added, dropped or reordered.
        """

        # Act
        digest = plan_digest(GOLDEN_CONFIGS[config_id])

        # Assert
        assert digest == GOLDEN_PLANS[config_id]

    @pytest.mark.parametrize("config_id", CONFIG_IDS)
    def test_plan_should_draw_every_peak_exactly_once(self, config_id):
        """Test the peak draws of one epoch.

        Given:
            Any golden configuration.
        When:
            One epoch is planned.
        Then:
            It should draw each peak row exactly once among its peak
            slots.
        """

        # Arrange
        sampler = make_sampler(config_id)

        # Act
        plan = sampler.plan(0)

        # Assert
        assert sorted(plan.index[plan.is_peak].tolist()) == list(range(sampler.n_peaks))

    @pytest.mark.parametrize("config_id", CONFIG_IDS)
    def test_plan_should_fill_exactly_the_slots_the_ratio_gives(self, config_id):
        """Test the slot proportions of one epoch.

        Given:
            Any golden configuration.
        When:
            One epoch is planned.
        Then:
            It should hold one peak slot per peak and negatives in
            every remaining slot of the epoch's length.
        """

        # Arrange
        sampler = make_sampler(config_id)

        # Act
        plan = sampler.plan(0)

        # Assert
        assert len(plan) == len(sampler)
        assert int(plan.is_peak.sum()) == sampler.n_peaks
        assert int((~plan.is_peak).sum()) == len(sampler) - sampler.n_peaks

    @pytest.mark.parametrize("config_id", CONFIG_IDS)
    def test_plan_should_give_every_negative_no_jitter(self, config_id):
        """Test the jitter of the negative draws.

        Given:
            Any golden configuration.
        When:
            One epoch is planned.
        Then:
            It should give every negative slot a jitter of zero,
            because a negative's window carries no jitter margin to
            read into.
        """

        # Act
        plan = make_sampler(config_id).plan(0)

        # Assert
        assert not plan.jitter[~plan.is_peak].any()

    def test_plan_should_keep_peak_jitter_inside_the_margin(self):
        """Test the range a peak's jitter is drawn from.

        Given:
            A jitter bound of two.
        When:
            Several epochs are planned.
        Then:
            Every jitter should lie in ``[0, 4)``, and both ends of
            that range should occur.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        jitter = numpy.concatenate([sampler.plan(epoch).jitter for epoch in range(8)])

        # Assert
        assert jitter.min() == 0
        assert jitter.max() == 3

    def test_plan_should_mirror_nothing_when_reverse_complement_is_off(self):
        """Test the orientation of the draws without augmentation.

        Given:
            A sampler with reverse-complement off.
        When:
            One epoch is planned.
        Then:
            It should mirror no draw.
        """

        # Act
        plan = make_sampler("no_rc").plan(0)

        # Assert
        assert not plan.reverse_complement.any()

    def test_plan_should_mirror_some_draws_and_not_others_when_asked(self):
        """Test the orientation of the draws under augmentation.

        Given:
            A sampler with reverse-complement on.
        When:
            One epoch is planned.
        Then:
            It should mirror some draws and leave others, peaks and
            negatives both being eligible.
        """

        # Act
        plan = make_sampler("baseline").plan(0)

        # Assert
        assert plan.reverse_complement.any()
        assert not plan.reverse_complement.all()

    def test_plan_should_keep_the_peaks_in_row_order_when_shuffle_is_off(self):
        """Test the peak order of an unshuffled sampler.

        Given:
            A sampler with shuffling off.
        When:
            Three successive epochs are planned.
        Then:
            The peak order should be the row order every time, and the
            peak slots should carry the rows in that order.
        """

        # Arrange
        sampler = make_sampler("sequential")
        rows = numpy.arange(sampler.n_peaks)

        # Act
        plans = [sampler.plan(epoch) for epoch in range(3)]

        # Assert
        for plan in plans:
            assert numpy.array_equal(plan.peak_ordering, rows)
            assert numpy.array_equal(plan.index[plan.is_peak], rows)

    def test_plan_should_differ_between_epochs(self):
        """Test that successive epochs are different draws.

        Given:
            A shuffled sampler.
        When:
            Two consecutive epochs are planned.
        Then:
            Their rows should not be identical.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        first = sampler.plan(0).index
        second = sampler.plan(1).index

        # Assert
        assert not numpy.array_equal(first, second)

    def test_plan_should_differ_between_seeds(self):
        """Test that the seed decides the draws.

        Given:
            Two samplers differing only in seed.
        When:
            Each plans the first epoch.
        Then:
            Their rows should differ.
        """

        # Arrange
        first = make_sampler("baseline")
        second = make_sampler("baseline", random_state=12345)

        # Act & assert
        assert not numpy.array_equal(first.plan(0).index, second.plan(0).index)

    def test_plan_should_not_depend_on_what_was_planned_before(self):
        """Test that a plan is a function of the seed and the epoch alone.

        Given:
            Two samplers of one seed, one of which has already planned
            other epochs in another order.
        When:
            Each plans the same epochs.
        Then:
            The draws should agree epoch for epoch.
        """

        # Arrange
        fresh = make_sampler("baseline")
        used = make_sampler("baseline")
        for epoch in (3, 0, 1):
            used.plan(epoch)

        # Act & assert
        for epoch in (0, 1, 3):
            assert draws(fresh.plan(epoch)) == draws(used.plan(epoch))

    def test_plan_should_leave_the_epoch_where_it_is(self):
        """Test that planning is not iterating.

        Given:
            A fresh sampler.
        When:
            A later epoch is planned.
        Then:
            The epoch the next iteration emits should still be the
            first.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        sampler.plan(5)

        # Assert
        assert sampler.epoch == 0

    def test_pickling_should_restore_a_sampler_that_plans_the_same_draws(self):
        """Test that a sampler plans the same draws after a round trip.

        Given:
            A sampler that has already planned an epoch.
        When:
            It is pickled and restored.
        Then:
            The copy should plan the same draws, as every spawned
            worker holding one requires.
        """

        # Arrange
        sampler = make_sampler("baseline")
        sampler.plan(0)

        # Act
        restored = pickle.loads(pickle.dumps(sampler))

        # Assert
        assert draws(restored.plan(2)) == draws(sampler.plan(2))

    @settings(max_examples=50)
    @given(arguments=sampler_arguments, epoch=st.integers(min_value=0, max_value=50))
    def test_plan_should_hold_its_invariants_for_any_arguments(self, arguments, epoch):
        """Test the plan's contract over arbitrary samplers and epochs.

        Given:
            Any peak count, negative count, ratio, jitter bound,
            orientation and shuffle setting, seed and epoch.
        When:
            That epoch is planned twice, by two samplers.
        Then:
            Each peak should be drawn once, every negative row should
            exist, every jitter should lie inside the margin and be
            zero for a negative, and the two plans should agree.
        """

        # Arrange
        sampler = sampling.PeakNegativeMixSampler(**arguments)
        other = sampling.PeakNegativeMixSampler(**arguments)
        other.plan(epoch + 1)

        # Act
        plan = sampler.plan(epoch)

        # Assert
        assert len(plan) == len(sampler)
        assert sorted(plan.index[plan.is_peak].tolist()) == list(range(sampler.n_peaks))
        negatives = plan.index[~plan.is_peak]
        assert ((negatives >= 0) & (negatives < sampler.n_negatives)).all()
        assert (plan.jitter >= 0).all()
        assert (plan.jitter < max(1, 2 * sampler.max_jitter)).all()
        assert not plan.jitter[~plan.is_peak].any()
        assert draws(plan) == draws(other.plan(epoch))

    def test___iter___should_emit_epoch_zero_first(self):
        """Test the first pass over a fresh sampler.

        Given:
            A fresh sampler.
        When:
            It is iterated once.
        Then:
            It should yield the draws of epoch zero.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        first = [tuple(draw) for draw in sampler]

        # Assert
        assert first == draws(make_sampler("baseline").plan(0))

    def test___iter___should_advance_the_epoch(self):
        """Test successive passes over one sampler.

        Given:
            A fresh sampler.
        When:
            It is iterated three times.
        Then:
            It should yield epochs zero, one and two in order, and
            report three as the epoch it is at.
        """

        # Arrange
        sampler = make_sampler("baseline")
        reference = make_sampler("baseline")

        # Act
        seen = [[tuple(draw) for draw in sampler] for _ in range(3)]

        # Assert
        assert seen == [draws(reference.plan(epoch)) for epoch in range(3)]
        assert sampler.epoch == 3

    def test___iter___should_advance_the_epoch_when_the_pass_starts(self):
        """Test when a pass moves the epoch.

        Given:
            A fresh sampler.
        When:
            An iteration is started and no draw is taken from it.
        Then:
            It should report epoch one, and the pass should still yield
            the draws of epoch zero.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        started = iter(sampler)

        # Assert
        assert sampler.epoch == 1
        assert [tuple(draw) for draw in started] == draws(
            make_sampler("baseline").plan(0)
        )

    def test_set_epoch_should_choose_the_plan_the_next_pass_emits(self):
        """Test positioning a sampler at an epoch.

        Given:
            A sampler set to epoch five.
        When:
            It is iterated.
        Then:
            It should yield epoch five's draws and move on to six.
        """

        # Arrange
        sampler = make_sampler("baseline")

        # Act
        sampler.set_epoch(5)
        emitted = [tuple(draw) for draw in sampler]

        # Assert
        assert emitted == draws(make_sampler("baseline").plan(5))
        assert sampler.epoch == 6


class TestShardedEpochSampler:
    """One rank's share of every global batch."""

    @pytest.mark.parametrize(
        "batch_size,rank,world_size,match",
        [
            (0, 0, 1, "at least 1"),
            (6, 0, 4, "divisible"),
            (8, 2, 2, r"rank must be in range\(2\)"),
        ],
    )
    def test_validate_arguments_should_raise_what_the_constructor_refuses(
        self, batch_size, rank, world_size, match
    ):
        """Test the refusals a caller can make before it has an epoch's size.

        Given:
            An empty batch, a batch the world does not divide, and a
            rank outside the world.
        When:
            The arguments are validated without a sampler.
        Then:
            Each should raise `ValueError` naming what is wrong, as the
            constructor does.
        """

        # Act & assert
        with pytest.raises(ValueError, match=match):
            sampling.ShardedEpochSampler.validate_arguments(batch_size, rank, world_size)

    def test___init___should_raise_when_the_ranks_cannot_split_a_batch(self):
        """Test the refusal of a batch the world does not divide.

        Given:
            A global batch of six over four ranks.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError` saying the batch must be
            divisible.
        """

        # Act & assert
        with pytest.raises(ValueError, match="divisible"):
            sampling.ShardedEpochSampler(36, 6, world_size=4)

    @pytest.mark.parametrize("rank", [-1, 2, 5])
    def test___init___should_raise_when_the_rank_is_outside_the_world(self, rank):
        """Test the refusal of a rank that is no member of the world.

        Given:
            Two ranks, and a rank that is negative or not below two.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError` naming the range, rather than
            hand that rank indices another step owns.
        """

        # Act & assert
        with pytest.raises(ValueError, match=r"rank must be in range\(2\)"):
            sampling.ShardedEpochSampler(36, 8, rank=rank, world_size=2)

    @pytest.mark.parametrize("batch_size,world_size", [(0, 1), (-8, 1), (8, 0), (8, -2)])
    def test___init___should_raise_when_a_size_is_not_positive(
        self, batch_size, world_size
    ):
        """Test the refusal of an empty batch or an empty world.

        Given:
            A batch size or a world size below one.
        When:
            The sampler is constructed.
        Then:
            It should raise `ValueError` naming both, rather than
            divide by zero later.
        """

        # Act & assert
        with pytest.raises(ValueError, match="at least 1"):
            sampling.ShardedEpochSampler(36, batch_size, world_size=world_size)

    @pytest.mark.parametrize(
        "n,batch_size,world_size,expected",
        [(36, 8, 1, 32), (36, 8, 2, 16), (36, 8, 4, 8), (7, 8, 1, 0), (16, 8, 2, 8)],
    )
    def test___len___should_count_only_whole_global_batches(
        self, n, batch_size, world_size, expected
    ):
        """Test the length of one rank's epoch.

        Given:
            An epoch size, a global batch and a world size.
        When:
            The sampler's length is taken and it is iterated.
        Then:
            It should report, and yield, one rank's share of the whole
            global batches and nothing of a trailing partial one.
        """

        # Arrange
        sampler = sampling.ShardedEpochSampler(n, batch_size, world_size=world_size)

        # Act
        yielded = list(sampler)

        # Assert
        assert len(sampler) == expected
        assert len(yielded) == expected

    def test___iter___should_visit_the_sequential_order_when_there_is_one_rank(self):
        """Test a world of one.

        Given:
            Ten indices in batches of four over one rank.
        When:
            The sampler is iterated.
        Then:
            It should yield the first eight indices in order, each
            paired with epoch zero, dropping the partial batch.
        """

        # Arrange
        sampler = sampling.ShardedEpochSampler(10, 4)

        # Act
        yielded = list(sampler)

        # Assert
        assert yielded == [(0, index) for index in range(8)]

    def test___iter___should_yield_the_slice_of_each_batch_the_rank_owns(self):
        """Test one rank's share, index by index.

        Given:
            Twenty indices in global batches of eight over four ranks,
            and the sampler of rank two.
        When:
            It is iterated.
        Then:
            It should yield the third pair of each whole batch.
        """

        # Arrange
        sampler = sampling.ShardedEpochSampler(20, 8, rank=2, world_size=4)

        # Act
        yielded = [index for _, index in sampler]

        # Assert
        assert sampler.local_batch_size == 2
        assert yielded == [4, 5, 12, 13]

    @settings(max_examples=100)
    @given(geometry=shard_geometries)
    def test___iter___should_partition_each_global_batch_across_the_ranks(
        self, geometry
    ):
        """Test that the ranks together read what one rank would.

        Given:
            Any epoch size, and any global batch a world divides.
        When:
            Every rank's sampler is iterated and the ranks' local
            batches are concatenated step by step, in rank order.
        Then:
            It should equal, step for step, the global batches a world
            of one yields.
        """

        # Arrange
        n, batch_size, world_size = geometry
        local = batch_size // world_size
        single = [index for _, index in sampling.ShardedEpochSampler(n, batch_size)]

        # Act
        per_rank = [
            [
                index
                for _, index in sampling.ShardedEpochSampler(
                    n, batch_size, rank=rank, world_size=world_size
                )
            ]
            for rank in range(world_size)
        ]

        # Assert
        combined = []
        for step in range(n // batch_size):
            for indices in per_rank:
                combined.extend(indices[step * local : (step + 1) * local])
        assert combined == single

    def test___iter___should_leave_the_epoch_where_it_was_set(self):
        """Test that a pass does not advance the epoch.

        Given:
            A sampler over eight indices.
        When:
            It is iterated twice.
        Then:
            Both passes should pair every index with epoch zero.
        """

        # Arrange
        sampler = sampling.ShardedEpochSampler(8, 4)

        # Act
        passes = [list(sampler), list(sampler)]

        # Assert
        assert passes[0] == passes[1]
        assert {epoch for epoch, _ in passes[1]} == {0}

    def test_set_epoch_should_pair_every_index_with_the_epoch_named(self):
        """Test naming the epoch of the next pass.

        Given:
            A sampler set to epoch three.
        When:
            It is iterated.
        Then:
            It should pair every index with three.
        """

        # Arrange
        sampler = sampling.ShardedEpochSampler(8, 4)

        # Act
        sampler.set_epoch(3)
        yielded = list(sampler)

        # Assert
        assert yielded == [(3, index) for index in range(8)]
