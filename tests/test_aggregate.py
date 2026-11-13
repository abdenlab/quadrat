"""Tests for the record a pileup is told what to count by.

Everything here runs on a base install: the record is data, and what
the pileup does with one is tested through the pileup, in
``tests/test_track.py`` and ``tests/test_run.py``.
"""

import dataclasses
import math
import pickle

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat import Aggregate, read_manifest, tables
from tests import FILE_BACKED


@st.composite
def aggregates(draw):
    """Any record `Aggregate` accepts."""

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
    )


class TestAggregate:
    """The record, its validation and what it derives."""

    def test___init___should_default_every_field_to_counting_primary_five_prime_ends(
        self,
    ):
        """Test the default of every field.

        Given:
            No arguments.
        When:
            An `Aggregate` is constructed.
        Then:
            It should exclude unmapped, secondary, QC-failed, duplicate
            and supplementary records and nothing else, require no
            flag, and count stranded 5' ends unshifted and unscaled.
        """

        # Act
        aggregate = Aggregate()

        # Assert
        assert aggregate.exclude_flags == 0x4 | 0x100 | 0x200 | 0x400 | 0x800
        assert aggregate.require_flags == 0
        assert aggregate.min_mapq == 0
        assert aggregate.min_template_length == 0
        assert aggregate.max_template_length is None
        assert aggregate.position == "five_prime"
        assert (aggregate.start_shift, aggregate.end_shift) == (0, 0)
        assert aggregate.unstranded is False
        assert aggregate.scale_factor == 1.0

    @pytest.mark.parametrize(
        "field, value",
        [
            ("exclude_flags", 1.5),
            ("exclude_flags", True),
            ("require_flags", "64"),
            ("min_mapq", None),
            ("min_template_length", 2.0),
            ("max_template_length", "300"),
            ("start_shift", 4.0),
            ("end_shift", False),
            ("unstranded", 1),
            ("unstranded", "yes"),
            ("scale_factor", "2"),
            ("scale_factor", True),
        ],
    )
    def test___init___should_refuse_a_field_of_the_wrong_type(self, field, value):
        """Test the type each field is held to.

        Given:
            One field set to a value of a type it does not take,
            including a `bool` where an integer or a real is asked for.
        When:
            An `Aggregate` is constructed.
        Then:
            It should raise `TypeError` naming the field.
        """

        # Act & assert
        with pytest.raises(TypeError, match=field):
            Aggregate(**{field: value})

    @pytest.mark.parametrize(
        "fields, message",
        [
            ({"position": "fragment"}, "position must be one of"),
            ({"exclude_flags": -1}, "between 0 and 0xFFFF"),
            ({"require_flags": 0x10000}, "between 0 and 0xFFFF"),
            ({"require_flags": 0x400}, "no record could qualify"),
            ({"exclude_flags": 0, "require_flags": 0x4}, "no record could qualify"),
            ({"min_mapq": -1}, "min_mapq must not be negative"),
            ({"min_template_length": -1}, "must not be negative"),
            (
                {"min_template_length": 200, "max_template_length": 100},
                "no record could qualify",
            ),
            (
                {"position": "midpoint", "max_template_length": 0},
                "admits no record under 'midpoint'",
            ),
            ({"scale_factor": float("inf")}, "must be finite"),
            ({"scale_factor": float("nan")}, "must be finite"),
            ({"scale_factor": 10**400}, "too large to hold as a float"),
        ],
    )
    def test___init___should_refuse_a_field_no_pileup_can_run(self, fields, message):
        """Test the values each field is held to.

        Given:
            A field outside its range, or two that together admit no
            record.
        When:
            An `Aggregate` is constructed.
        Then:
            It should raise `ValueError` saying why.
        """

        # Act & assert
        with pytest.raises(ValueError, match=message):
            Aggregate(**fields)

    @pytest.mark.parametrize("position", ["midpoint", "span"])
    @pytest.mark.parametrize("shift", ["start_shift", "end_shift"])
    def test___init___should_refuse_a_shift_under_a_rule_that_takes_none(
        self, position, shift
    ):
        """Test that a shift is refused rather than given a meaning.

        Given:
            A rule not taken from one end of an alignment, and a
            non-zero shift.
        When:
            An `Aggregate` is constructed.
        Then:
            It should raise `ValueError`.
        """

        # Act & assert
        with pytest.raises(ValueError, match="takes no shift"):
            Aggregate(position=position, unstranded=True, **{shift: 4})

    def test___init___should_refuse_a_stranded_midpoint(self):
        """Test that the midpoint rule is not split by strand.

        Given:
            The midpoint rule with the strands kept apart explicitly.
        When:
            An `Aggregate` is constructed.
        Then:
            It should raise `ValueError`, because the midpoint is taken
            from the mate with the positive template length, which is
            almost always the forward one, so the split would carry no
            strand.
        """

        # Act & assert
        with pytest.raises(ValueError, match=r"'midpoint' is always unstranded"):
            Aggregate(position="midpoint", unstranded=False)

    @pytest.mark.parametrize(
        "position, unstranded",
        [
            ("five_prime", False),
            ("three_prime", False),
            ("span", False),
            ("midpoint", True),
        ],
    )
    def test___init___should_resolve_strandedness_from_the_rule_by_default(
        self, position, unstranded
    ):
        """Test the default ``unstranded`` each rule resolves to.

        Given:
            Each position rule and no ``unstranded``.
        When:
            An `Aggregate` is constructed.
        Then:
            It should hold the `bool` the rule implies, pooled under
            ``"midpoint"`` and kept apart otherwise, and equal a record
            given that value explicitly, field for field.
        """

        # Act
        aggregate = Aggregate(position=position)

        # Assert
        assert aggregate.unstranded is unstranded
        assert aggregate == Aggregate(position=position, unstranded=unstranded)
        assert dataclasses.asdict(aggregate) == dataclasses.asdict(
            Aggregate(position=position, unstranded=unstranded)
        )

    def test___init___should_refuse_fields_given_by_position(self):
        """Test that every field is passed by keyword.

        Given:
            Three integers passed positionally, which would otherwise
            bind to the first three fields.
        When:
            An `Aggregate` is constructed.
        Then:
            It should raise `TypeError`, so inserting a field later
            cannot silently rebind a caller's arguments.
        """

        # Act & assert
        with pytest.raises(TypeError, match="positional"):
            Aggregate(1796, 64, 30)

    def test___init_subclass___should_refuse_any_subclass(self):
        """Test that the record cannot be subclassed.

        Given:
            A class deriving from `Aggregate`.
        When:
            It is defined.
        Then:
            It should raise `TypeError` naming the subclass, so no
            object can pass the acceptance checks while overriding the
            validation, the signal or the fields a pileup reads.
        """

        # Act & assert
        with pytest.raises(TypeError, match="Aggregate cannot be subclassed, and Mine"):

            class Mine(Aggregate):
                @property
                def signal(self):
                    return "counts"

    def test___init___should_hold_a_numpy_integer_as_an_int(self):
        """Test that a numpy integer is normalized.

        Given:
            Flag masks and a shift given as numpy integers.
        When:
            An `Aggregate` is constructed.
        Then:
            It should hold them as `int`, equal to one built from ints.
        """

        # Act
        aggregate = Aggregate(
            exclude_flags=numpy.uint16(1796),
            require_flags=numpy.int64(64),
            start_shift=numpy.int32(4),
        )

        # Assert
        assert type(aggregate.exclude_flags) is int
        assert aggregate == Aggregate(
            exclude_flags=1796, require_flags=64, start_shift=4
        )

    def test___init___should_hold_numpy_scalars_as_plain_values(self):
        """Test that a numpy string and a numpy bool are normalized.

        Given:
            A position rule given as a `numpy.str_` and ``unstranded``
            as a `numpy.bool_`.
        When:
            An `Aggregate` is constructed and pickled.
        Then:
            It should hold them as `str` and `bool`, equal a record
            built from plain values, and pickle to the same bytes as
            that record, so nothing of numpy crosses a process boundary.
        """

        # Arrange
        plain = Aggregate(position="span", unstranded=True)

        # Act
        aggregate = Aggregate(position=numpy.str_("span"), unstranded=numpy.bool_(True))

        # Assert
        assert type(aggregate.position) is str
        assert type(aggregate.unstranded) is bool
        assert aggregate == plain
        assert pickle.dumps(aggregate) == pickle.dumps(plain)

    def test___setattr___should_raise_on_a_frozen_record(self):
        """Test that the record is frozen.

        Given:
            An `Aggregate`.
        When:
            A field is assigned.
        Then:
            It should raise rather than change what a pileup counts.
        """

        # Arrange
        aggregate = Aggregate()

        # Act & assert
        with pytest.raises(dataclasses.FrozenInstanceError):
            aggregate.position = "span"

    @settings(max_examples=100)
    @given(aggregate=aggregates())
    def test___reduce_ex___should_round_trip_any_record_through_pickle(self, aggregate):
        """Test the property a process boundary relies on.

        Given:
            Any record `Aggregate` accepts.
        When:
            It is pickled and unpickled.
        Then:
            It should equal the original, hash alike, hold
            ``unstranded`` as a `bool`, and derive the same signal.
        """

        # Act
        copy = pickle.loads(pickle.dumps(aggregate))

        # Assert
        assert copy == aggregate
        assert hash(copy) == hash(aggregate)
        assert type(copy.unstranded) is bool
        assert copy.signal == aggregate.signal

    @settings(max_examples=200)
    @given(
        scale=st.one_of(
            st.floats(),
            st.integers(min_value=-(10**400), max_value=10**400),
        )
    )
    def test___init___should_hold_a_scale_factor_exactly_when_it_is_finite(self, scale):
        """Test the domain ``scale_factor`` is held to.

        Given:
            Any float, finite or not, and any integer up to 400 digits,
            including those too large to hold as a float.
        When:
            An `Aggregate` is constructed with it as ``scale_factor``.
        Then:
            It should hold the value as the equal `float` when that
            float is finite, and raise `ValueError` otherwise, never
            another exception.
        """

        # Arrange
        try:
            held = float(scale)
        except OverflowError:
            held = math.inf

        # Act & assert
        if math.isfinite(held):
            aggregate = Aggregate(scale_factor=scale)
            assert type(aggregate.scale_factor) is float
            assert aggregate.scale_factor == held
        else:
            with pytest.raises(ValueError, match="scale_factor must be finite"):
                Aggregate(scale_factor=scale)

    @settings(FILE_BACKED, max_examples=50)
    @given(aggregate=aggregates())
    def test___init___should_rebuild_any_record_from_the_provenance_a_track_records(
        self, tmp_path, aggregate
    ):
        """Test the provenance round-trip through a track's record.

        Given:
            Any record `Aggregate` accepts.
        When:
            Its fields are recorded in a track's manifest as a pileup
            records them, under ``parameters``, read back, and passed
            to `Aggregate` again.
        Then:
            It should equal the original.
        """

        # Arrange
        path = str(tmp_path / ("p" + tables.TRACK_SUFFIX))
        with tables.TrackWriter(
            path, parameters={"aggregate": dataclasses.asdict(aggregate)}
        ):
            pass

        # Act
        rebuilt = Aggregate(**read_manifest(path)["parameters"]["aggregate"])

        # Assert
        assert rebuilt == aggregate

    @pytest.mark.parametrize(
        "position, signal",
        [
            ("five_prime", "counts"),
            ("three_prime", "counts"),
            ("midpoint", "counts"),
            ("span", "coverage_deltas"),
        ],
    )
    def test_signal_should_name_coverage_deltas_for_span_alone(self, position, signal):
        """Test what each rule's values mean.

        Given:
            Each position rule.
        When:
            The record's signal is read.
        Then:
            It should be coverage deltas for ``"span"`` and counts for
            the three rules that name one position.
        """

        # Act & assert
        assert Aggregate(position=position, unstranded=True).signal == signal
