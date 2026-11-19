"""Tests for the jitter and exclusion rules and the shorthand they resolve.

`Jitter` is a value, so it is pinned on its fields, its margin, its
value semantics and its refusal of subclasses. `Exclusion` holds
regions that may be frames, so it is pinned on its defaults, its
refusals and its identity semantics. `as_jitter` and `as_exclusions`
are pinned on each shape of argument they resolve.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import pickle

import numpy
import pandas
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat.rules import (
    EXCLUSION_BIN_WIDTH,
    Exclusion,
    Jitter,
    as_exclusions,
    as_jitter,
)


class TestJitter:
    """The rule a peak's draw is shifted by."""

    def test___init___should_hold_both_bounds_as_int(self):
        """Test construction from integer bounds of either type.

        Given:
            A Python `int` low bound and a numpy integer high bound,
            low below high.
        When:
            A `Jitter` is constructed.
        Then:
            It should answer both bounds by name, each as an `int`.
        """

        # Act
        jitter = Jitter(-3, numpy.int64(4))

        # Assert
        assert (jitter.low, jitter.high) == (-3, 4)
        assert type(jitter.low) is int
        assert type(jitter.high) is int

    def test___init___should_raise_value_error_when_the_bounds_are_out_of_order(self):
        """Test the refusal of a low bound above the high bound.

        Given:
            A low bound of 3 and a high bound of 2.
        When:
            A `Jitter` is constructed.
        Then:
            It should raise `ValueError` naming both bounds.
        """

        # Act & assert
        with pytest.raises(ValueError, match="got low=3 and high=2"):
            Jitter(3, 2)

    def test___init___should_accept_lopsided_and_one_sided_ranges(self):
        """Test the acceptance of ranges that need not straddle ``0``.

        Given:
            An empty range at ``0``, a range ending at ``0``, and ranges
            lying wholly right and wholly left of ``0``.
        When:
            Each `Jitter` is constructed.
        Then:
            It should construct each with the bounds given.
        """

        # Act
        rules = [Jitter(0, 0), Jitter(-1, 0), Jitter(2, 5), Jitter(-5, -2)]

        # Assert
        assert [(rule.low, rule.high) for rule in rules] == [
            (0, 0),
            (-1, 0),
            (2, 5),
            (-5, -2),
        ]

    @pytest.mark.parametrize("bad", [1.0, "1", True], ids=["float", "str", "bool"])
    @pytest.mark.parametrize("field", ["low", "high"])
    def test___init___should_raise_type_error_when_a_bound_is_no_integer(
        self, field, bad
    ):
        """Test the refusal of a bound that is not an integer.

        Given:
            A float, a str or a `bool` for one bound and ``0`` for the
            other.
        When:
            A `Jitter` is constructed.
        Then:
            It should raise `TypeError` naming the field.
        """

        # Arrange
        bounds = {"low": 0, "high": 0, field: bad}

        # Act & assert
        with pytest.raises(
            TypeError, match=r"Jitter\.{} must be an integer".format(field)
        ):
            Jitter(**bounds)

    @pytest.mark.parametrize(
        ("low", "high", "margin"),
        [
            (-500, 500, 500),
            (-500, 499, 500),
            (0, 0, 0),
            (0, 5, 5),
            (-5, 2, 5),
            (2, 5, 5),
            (-5, -2, 5),
        ],
        ids=[
            "symmetric",
            "before_92",
            "none",
            "right_heavy",
            "left_heavy",
            "right_only",
            "left_only",
        ],
    )
    def test_margin_should_answer_the_widest_shift_either_way(self, low, high, margin):
        """Test the margin of symmetric, lopsided and one-sided rules.

        Given:
            A rule's two bounds.
        When:
            Its margin is read.
        Then:
            It should answer the larger of ``-low`` and ``high``, and
            never less than ``0``.
        """

        # Arrange
        jitter = Jitter(low, high)

        # Act
        answered = jitter.margin

        # Assert
        assert answered == margin

    @settings(max_examples=100)
    @given(
        bounds=st.tuples(st.integers(-(10**6), 10**6), st.integers(-(10**6), 10**6)).map(
            sorted
        )
    )
    def test_margin_should_be_the_least_widening_that_holds_every_shift_for_any_bounds(
        self, bounds
    ):
        """Test that the margin is the least widening holding the rule.

        Given:
            Any ordered pair of bounds within a million of ``0``.
        When:
            The margin of the rule they make is read.
        Then:
            It should be no less than ``0``, a window widened by it
            should hold every shift from ``low`` to ``high``, and one
            base less should not.
        """

        # Arrange
        low, high = bounds
        jitter = Jitter(low, high)

        # Act
        answered = jitter.margin

        # Assert
        assert answered >= 0
        assert low + answered >= 0
        assert high <= answered
        assert answered == 0 or low + (answered - 1) < 0 or high > answered - 1

    def test___eq___should_compare_by_its_fields(self):
        """Test the equality of rules.

        Given:
            Two rules with equal bounds and one with different bounds.
        When:
            They are compared.
        Then:
            It should make the equal rules equal and the different one
            unequal.
        """

        # Arrange
        first, second, other = Jitter(-2, 1), Jitter(-2, 1), Jitter(-2, 2)

        # Act & assert
        assert first == second
        assert first != other

    def test___hash___should_agree_for_equal_rules(self):
        """Test the hash of rules.

        Given:
            Two rules with equal bounds, built separately.
        When:
            Both are hashed.
        Then:
            It should answer the same hash for both.
        """

        # Arrange
        first, second = Jitter(-2, 1), Jitter(-2, 1)

        # Act & assert
        assert hash(first) == hash(second)

    @settings(max_examples=100)
    @given(
        bounds=st.tuples(st.integers(-(10**6), 10**6), st.integers(-(10**6), 10**6)).map(
            sorted
        )
    )
    def test___reduce_ex___should_round_trip_any_rule_through_pickle(self, bounds):
        """Test the property a process boundary relies on.

        Given:
            Any rule over an ordered pair of bounds within a million of
            ``0``.
        When:
            It is pickled and unpickled.
        Then:
            It should equal the original, hash alike, and hold ``low``
            and ``high`` as `int`.
        """

        # Arrange
        rule = Jitter(*bounds)

        # Act
        copy = pickle.loads(pickle.dumps(rule))

        # Assert
        assert copy == rule
        assert hash(copy) == hash(rule)
        assert type(copy.low) is int
        assert type(copy.high) is int

    def test___init_subclass___should_raise_type_error(self):
        """Test the refusal of a subclass.

        Given:
            A class statement deriving from `Jitter`.
        When:
            The class is defined.
        Then:
            It should raise `TypeError` saying `Jitter` cannot be
            subclassed.
        """

        # Act & assert
        with pytest.raises(TypeError, match="Jitter cannot be subclassed"):

            class Wider(Jitter):
                pass


class TestExclusion:
    """The rule a set of excluded regions removes windows by."""

    def test___init___should_take_the_default_rule(self):
        """Test construction from regions alone.

        Given:
            One region path.
        When:
            An `Exclusion` is constructed with no other argument.
        Then:
            It should take a bin width of `EXCLUSION_BIN_WIDTH`, 100, and
            no end padding.
        """

        # Act
        exclusion = Exclusion("a.bed")

        # Assert
        assert exclusion.bin_width == EXCLUSION_BIN_WIDTH == 100
        assert exclusion.end_padding == 0

    def test___init___should_hold_a_list_of_regions_as_a_tuple(self):
        """Test the normalisation of listed regions.

        Given:
            A list of two region paths.
        When:
            An `Exclusion` is constructed.
        Then:
            It should hold the paths as a tuple, in order.
        """

        # Act
        exclusion = Exclusion(["a.bed", "b.bed"])

        # Assert
        assert exclusion.regions == ("a.bed", "b.bed")

    def test___init___should_hold_numpy_integers_as_int(self):
        """Test the normalisation of the rule's integers.

        Given:
            A numpy integer bin width and end padding.
        When:
            An `Exclusion` is constructed.
        Then:
            It should hold both as `int`.
        """

        # Act
        exclusion = Exclusion("a.bed", numpy.int64(7), numpy.int32(2))

        # Assert
        assert (exclusion.bin_width, exclusion.end_padding) == (7, 2)
        assert type(exclusion.bin_width) is int
        assert type(exclusion.end_padding) is int

    @pytest.mark.parametrize(
        ("field", "value", "floor"), [("bin_width", 0, 1), ("end_padding", -1, 0)]
    )
    def test___init___should_raise_value_error_when_an_integer_is_below_its_floor(
        self, field, value, floor
    ):
        """Test the refusal of a bin width or end padding below its floor.

        Given:
            A bin width of 0 or an end padding of -1.
        When:
            An `Exclusion` is constructed.
        Then:
            It should raise `ValueError` naming the field and its floor.
        """

        # Act & assert
        with pytest.raises(
            ValueError,
            match=r"Exclusion\.{} must be at least {}, got {}".format(
                field, floor, value
            ),
        ):
            Exclusion("a.bed", **{field: value})

    @pytest.mark.parametrize("bad", [1.0, True], ids=["float", "bool"])
    @pytest.mark.parametrize("field", ["bin_width", "end_padding"])
    def test___init___should_raise_type_error_when_an_integer_is_no_integer(
        self, field, bad
    ):
        """Test the refusal of a bin width or end padding that is not an integer.

        Given:
            A float or a `bool` for the bin width or the end padding.
        When:
            An `Exclusion` is constructed.
        Then:
            It should raise `TypeError` naming the field.
        """

        # Act & assert
        with pytest.raises(
            TypeError, match=r"Exclusion\.{} must be an integer".format(field)
        ):
            Exclusion("a.bed", **{field: bad})

    def test___init___should_raise_type_error_when_the_regions_are_an_exclusion(self):
        """Test the refusal of one rule nested in another.

        Given:
            An `Exclusion` as the regions of another.
        When:
            The outer `Exclusion` is constructed.
        Then:
            It should raise `TypeError` telling the caller to pass the
            rules side by side.
        """

        # Arrange
        inner = Exclusion("a.bed")

        # Act & assert
        with pytest.raises(
            TypeError,
            match=r"Pass the rules side by side, as \[Exclusion\(a\), Exclusion\(b\)\]",
        ):
            Exclusion(inner, end_padding=1)

    @pytest.mark.parametrize("container", [list, tuple])
    def test___init___should_raise_type_error_when_the_regions_hold_an_exclusion(
        self, container
    ):
        """Test the refusal of a rule nested among the regions of another.

        Given:
            A `list` or `tuple` of regions holding a path and an
            `Exclusion`.
        When:
            An `Exclusion` is constructed over them.
        Then:
            It should raise `TypeError` with the message the top-level
            nesting gets, rather than fail when the regions are read.
        """

        # Arrange
        regions = container(["b.bed", Exclusion("a.bed")])

        # Act & assert
        with pytest.raises(
            TypeError,
            match=r"Pass the rules side by side, as \[Exclusion\(a\), Exclusion\(b\)\]",
        ):
            Exclusion(regions)

    def test___eq___should_compare_by_identity(self):
        """Test that two rules with equal fields are not equal.

        Given:
            Two `Exclusion` values with the same regions and rule.
        When:
            They are compared.
        Then:
            It should answer each equal to itself alone.
        """

        # Arrange
        first, second = Exclusion("a.bed"), Exclusion("a.bed")

        # Act
        answered = (first == first, first == second)

        # Assert
        assert answered == (True, False)

    def test___init_subclass___should_raise_type_error(self):
        """Test the refusal of a subclass.

        Given:
            A class statement deriving from `Exclusion`.
        When:
            The class is defined.
        Then:
            It should raise `TypeError` saying `Exclusion` cannot be
            subclassed.
        """

        # Act & assert
        with pytest.raises(TypeError, match="Exclusion cannot be subclassed"):

            class Padded(Exclusion):
                pass


def test_as_jitter_should_answer_the_symmetric_rule_for_an_integer():
    """Test the shorthand an integer stands for.

    Given:
        The integer 6.
    When:
        It is resolved.
    Then:
        It should answer ``Jitter(-6, 6)``.
    """

    # Act
    jitter = as_jitter(6)

    # Assert
    assert jitter == Jitter(-6, 6)


@settings(max_examples=100)
@given(margin=st.integers(min_value=0, max_value=10**6))
def test_as_jitter_should_answer_the_symmetric_rule_for_any_integer(margin):
    """Test the shorthand any non-negative integer stands for.

    Given:
        Any integer from ``0`` to a million.
    When:
        It is resolved.
    Then:
        It should answer the rule from minus it to it, whose margin is
        the integer itself.
    """

    # Act
    jitter = as_jitter(margin)

    # Assert
    assert jitter == Jitter(-margin, margin)
    assert jitter.margin == margin


def test_as_jitter_should_answer_a_jitter_as_it_is():
    """Test the passthrough of a rule.

    Given:
        A `Jitter`.
    When:
        It is resolved.
    Then:
        It should answer the same object.
    """

    # Arrange
    jitter = Jitter(-2, 1)

    # Act
    answered = as_jitter(jitter)

    # Assert
    assert answered is jitter


def test_as_jitter_should_raise_value_error_when_the_integer_is_negative():
    """Test the refusal of a negative shorthand.

    Given:
        The integer -1.
    When:
        It is resolved.
    Then:
        It should raise `ValueError` naming ``max_jitter``.
    """

    # Act & assert
    with pytest.raises(ValueError, match="max_jitter must be non-negative, got -1"):
        as_jitter(-1)


@pytest.mark.parametrize("bad", [1.0, True, None], ids=["float", "bool", "none"])
def test_as_jitter_should_raise_type_error_when_given_neither_shape(bad):
    """Test the refusal of a value that is neither a rule nor an integer.

    Given:
        A float, a `bool` or ``None``.
    When:
        It is resolved.
    Then:
        It should raise `TypeError` naming ``max_jitter`` and the type.
    """

    # Act & assert
    with pytest.raises(
        TypeError,
        match="max_jitter must be a Jitter or an integer, got {}".format(
            type(bad).__name__
        ),
    ):
        as_jitter(bad)


def test_as_exclusions_should_answer_no_rule_for_none():
    """Test the resolution of no exclusion.

    Given:
        ``None``.
    When:
        It is resolved.
    Then:
        It should answer an empty list.
    """

    # Act
    rules = as_exclusions(None)

    # Assert
    assert rules == []


@pytest.mark.parametrize(
    "regions",
    ["a.bed", pandas.DataFrame([("chr1", 0, 10)])],
    ids=["path", "frame"],
)
def test_as_exclusions_should_answer_one_default_rule_for_one_locus_set(regions):
    """Test the resolution of a bare locus set.

    Given:
        One region path or one frame.
    When:
        It is resolved.
    Then:
        It should answer one `Exclusion` holding it under the default
        rule.
    """

    # Act
    rules = as_exclusions(regions)

    # Assert
    assert len(rules) == 1
    assert rules[0].regions is regions
    assert (rules[0].bin_width, rules[0].end_padding) == (EXCLUSION_BIN_WIDTH, 0)


def test_as_exclusions_should_answer_an_exclusion_alone_in_a_list():
    """Test the resolution of a single rule.

    Given:
        One `Exclusion`.
    When:
        It is resolved.
    Then:
        It should answer a list holding that same object.
    """

    # Arrange
    exclusion = Exclusion("a.bed", end_padding=1)

    # Act
    rules = as_exclusions(exclusion)

    # Assert
    assert len(rules) == 1
    assert rules[0] is exclusion


@pytest.mark.parametrize("container", [list, tuple])
def test_as_exclusions_should_gather_the_bare_locus_sets_after_the_rules(container):
    """Test the grouping of a list mixing rules and bare locus sets.

    Given:
        A list or tuple interleaving two bare paths and two `Exclusion`
        values.
    When:
        It is resolved.
    Then:
        It should answer the two `Exclusion` values in the order given,
        then one default `Exclusion` holding both paths in order.
    """

    # Arrange
    first = Exclusion("c.bed", bin_width=1)
    second = Exclusion("d.bed", end_padding=1)
    value = container(["a.bed", first, "b.bed", second])

    # Act
    rules = as_exclusions(value)

    # Assert
    assert len(rules) == 3
    assert rules[0] is first
    assert rules[1] is second
    assert rules[2].regions == ("a.bed", "b.bed")
    assert (rules[2].bin_width, rules[2].end_padding) == (EXCLUSION_BIN_WIDTH, 0)


def test_as_exclusions_should_answer_no_rule_for_an_empty_list():
    """Test the resolution of an empty list.

    Given:
        An empty list.
    When:
        It is resolved.
    Then:
        It should answer an empty list.
    """

    # Act
    rules = as_exclusions([])

    # Assert
    assert rules == []
