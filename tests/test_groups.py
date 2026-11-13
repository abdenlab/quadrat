"""Tests for signal groups: the check, the flattening and the permutation.

The check is the package's one public name here, so its refusals are
pinned value by value. The permutation is pinned by what it does to a
signal rather than by its entries alone: reversing within a group and
never across one is a statement about every group layout, so it is a
property over generated layouts beside the two examples a reader
would write by hand.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import numpy
import polars
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import quadrat
from quadrat import groups

# Group layouts as a specification states them: a few groups of a few
# channels each.
group_layouts = st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=6)

# Track specifications as a caller writes them: each entry a bare track
# name, or a non-empty list or tuple of names, the whole a list or a
# tuple.
track_names = st.text(min_size=1, max_size=4)
specification_entries = st.one_of(
    track_names,
    st.lists(track_names, min_size=1, max_size=4),
    st.lists(track_names, min_size=1, max_size=4).map(tuple),
)
track_specifications = st.one_of(
    st.lists(specification_entries, max_size=6),
    st.lists(specification_entries, max_size=6).map(tuple),
)


@pytest.mark.parametrize(
    "sizes",
    [
        [1],
        [1, 1, 1],
        [2],
        [1, 2],
        (2, 1, 3),
        [numpy.int64(2)],
        (numpy.int32(1), numpy.uint8(2)),
    ],
)
def test_validate_signal_groups_should_accept_positive_sizes(sizes):
    """Test the check on specifications it admits.

    Given:
        A non-empty list or tuple of positive integers, Python's or
        numpy's.
    When:
        It is validated.
    Then:
        It should raise nothing.
    """

    # Act & assert
    assert groups.validate_signal_groups(sizes) is None


@pytest.mark.parametrize(
    "value,named", [(2, "int"), (None, "NoneType"), ("12", "str"), ({1, 2}, "set")]
)
def test_validate_signal_groups_should_raise_type_error_when_not_a_list(value, named):
    """Test the check on something that is no sequence of sizes.

    Given:
        A value that is neither a list nor a tuple.
    When:
        It is validated.
    Then:
        It should raise `TypeError` saying what a specification is and
        naming the type it got.
    """

    # Act & assert
    with pytest.raises(
        TypeError, match="list or tuple of positive integers; got {}".format(named)
    ):
        groups.validate_signal_groups(value)


def test_validate_signal_groups_should_raise_when_empty():
    """Test the check on a specification describing no channel.

    Given:
        An empty list.
    When:
        It is validated.
    Then:
        It should raise `ValueError` saying it must be non-empty.
    """

    # Act & assert
    with pytest.raises(ValueError, match="non-empty"):
        groups.validate_signal_groups([])


@pytest.mark.parametrize(
    "sizes,named",
    [
        ([1, 1.5], "float"),
        ([1, "2"], "str"),
        ([1, True], "bool"),
        ([None], "NoneType"),
        ([numpy.float64(2.0)], "float64"),
    ],
)
def test_validate_signal_groups_should_raise_type_error_when_a_size_is_not_an_integer(
    sizes, named
):
    """Test the check on a size of no integer type.

    Given:
        A list holding a float, a string, a boolean, ``None`` or a
        numpy float.
    When:
        It is validated.
    Then:
        It should raise `TypeError` naming the type it got, a boolean
        included although it is an `int` to ``isinstance``.
    """

    # Act & assert
    with pytest.raises(
        TypeError, match="entries must be integers; got {} in".format(named)
    ):
        groups.validate_signal_groups(sizes)


@pytest.mark.parametrize("sizes", [[0], [1, 0], [-1, 2], [numpy.int64(0)]])
def test_validate_signal_groups_should_raise_value_error_when_a_size_is_not_positive(
    sizes,
):
    """Test the check on an integer size no group can have.

    Given:
        A list holding a zero or a negative integer.
    When:
        It is validated.
    Then:
        It should raise `ValueError` saying entries must be positive.
    """

    # Act & assert
    with pytest.raises(ValueError, match="entries must be positive"):
        groups.validate_signal_groups(sizes)


def test_validate_signal_groups_should_name_the_label_in_its_message():
    """Test that a refusal says which argument it is about.

    Given:
        An empty specification and a caller's own name for it.
    When:
        It is validated under that name.
    Then:
        It should raise naming it, so a bad control grouping is not
        reported as a bad signal grouping.
    """

    # Act & assert
    with pytest.raises(ValueError, match="control_groups"):
        groups.validate_signal_groups([], label="control_groups")


def test_validate_signal_groups_should_be_exported_by_the_package():
    """Test that the check is reachable from the package itself.

    Given:
        The package and the module that defines the check.
    When:
        The package's attribute is compared with the module's.
    Then:
        It should be the same function, listed in the export list.
    """

    # Act & assert
    assert quadrat.validate_signal_groups is groups.validate_signal_groups
    assert "validate_signal_groups" in quadrat.__all__


def test_normalize_signal_groups_should_return_no_tracks_when_given_none():
    """Test the flattening of an absent specification.

    Given:
        ``None``.
    When:
        It is normalized.
    Then:
        It should answer no track list and no groups.
    """

    # Act
    tracks, sizes = groups.normalize_signal_groups(None)

    # Assert
    assert tracks is None
    assert sizes == []


def test_normalize_signal_groups_should_make_each_flat_entry_its_own_group():
    """Test that a flat list is unstranded channels, not one stranded set.

    Given:
        A flat list of three tracks.
    When:
        It is normalized.
    Then:
        It should answer the tracks in order and three groups of one.
    """

    # Act
    tracks, sizes = groups.normalize_signal_groups(["a", "b", "c"])

    # Assert
    assert tracks == ["a", "b", "c"]
    assert sizes == [1, 1, 1]


def test_normalize_signal_groups_should_make_a_nested_entry_one_group():
    """Test that a nested pair is one two-channel group.

    Given:
        A list whose only entry is a pair of tracks, as a list and as
        a tuple.
    When:
        Each is normalized.
    Then:
        It should answer the two tracks in order and one group of two.
    """

    # Act
    from_list = groups.normalize_signal_groups([["plus", "minus"]])
    from_tuple = groups.normalize_signal_groups((("plus", "minus"),))

    # Assert
    assert from_list == (["plus", "minus"], [2])
    assert from_tuple == (["plus", "minus"], [2])


def test_normalize_signal_groups_should_keep_channel_order_when_entries_are_mixed():
    """Test a specification mixing bare tracks with nested groups.

    Given:
        A bare track, a nested pair and another bare track.
    When:
        The list is normalized.
    Then:
        It should answer the four tracks in the order written and the
        sizes one, two, one.
    """

    # Act
    tracks, sizes = groups.normalize_signal_groups(["atac", ["ctcf+", "ctcf-"], "k27"])

    # Assert
    assert tracks == ["atac", "ctcf+", "ctcf-", "k27"]
    assert sizes == [1, 2, 1]


@settings(max_examples=100)
@given(specification=track_specifications)
def test_normalize_signal_groups_should_flatten_any_nesting_in_order(specification):
    """Test the flattening over arbitrary specifications.

    Given:
        Any list or tuple of up to six entries, each a bare track or a
        non-empty list or tuple of tracks.
    When:
        It is normalized.
    Then:
        It should answer every track in the order written, as a list,
        and one size per entry: one for a bare track and the length of
        a nested one.
    """

    # Act
    tracks, sizes = groups.normalize_signal_groups(specification)

    # Assert
    nested = [
        entry if isinstance(entry, (list, tuple)) else [entry] for entry in specification
    ]
    assert type(tracks) is list
    assert tracks == [track for entry in nested for track in entry]
    assert sizes == [len(entry) for entry in nested]


def test_normalize_signal_groups_should_pass_a_track_through_when_it_is_no_path():
    """Test that a track is carried as it was given, whatever it is.

    Given:
        A specification whose tracks are in-memory frames.
    When:
        It is normalized.
    Then:
        It should answer the same frame objects, unopened, because a
        track is refused or read by whatever opens it.
    """

    # Arrange
    first = polars.DataFrame({"chrom": ["chr1"], "pos": [1], "value": [1.0]})
    second = first.lazy()

    # Act
    tracks, sizes = groups.normalize_signal_groups([first, [second, first]])

    # Assert
    assert [id(track) for track in tracks] == [id(first), id(second), id(first)]
    assert sizes == [1, 2]


def test_normalize_signal_groups_should_raise_when_a_group_is_empty():
    """Test the refusal of a group holding no track.

    Given:
        A specification with an empty nested entry.
    When:
        It is normalized.
    Then:
        It should raise `ValueError` saying a group is empty.
    """

    # Act & assert
    with pytest.raises(ValueError, match="empty"):
        groups.normalize_signal_groups([["a"], []])


@pytest.mark.parametrize("value", ["a.track.parquet", b"a.track.parquet", 3, {"a"}])
def test_normalize_signal_groups_should_raise_when_not_a_list(value):
    """Test the refusal of a specification that is no sequence of groups.

    Given:
        A single path, an integer or a set.
    When:
        It is normalized.
    Then:
        It should raise `TypeError`, rather than read a path as one
        group per character.
    """

    # Act & assert
    with pytest.raises(TypeError, match="one entry per group"):
        groups.normalize_signal_groups(value)


def test_channel_permutation_from_groups_should_swap_only_the_stranded_pair():
    """Test the permutation of an unstranded channel beside a stranded pair.

    Given:
        Group sizes of one and two.
    When:
        The permutation is taken.
    Then:
        It should leave the first channel in place and swap the other
        two, as an ``int64`` tensor.
    """

    # Act
    perm = groups.channel_permutation_from_groups([1, 2])

    # Assert
    assert perm.dtype == torch.int64
    assert perm.tolist() == [0, 2, 1]


def test_channel_permutation_from_groups_should_be_the_identity_when_unstranded():
    """Test the permutation of channels that are each their own group.

    Given:
        Three groups of one.
    When:
        The permutation is taken.
    Then:
        It should move no channel.
    """

    # Act
    perm = groups.channel_permutation_from_groups([1, 1, 1])

    # Assert
    assert perm.tolist() == [0, 1, 2]


@settings(max_examples=100)
@given(sizes=group_layouts)
def test_channel_permutation_from_groups_should_reverse_within_every_group(sizes):
    """Test the permutation over arbitrary group layouts.

    Given:
        Any layout of one to six groups of one to five channels.
    When:
        The permutation is taken.
    Then:
        Each group's slice should be that group's own channels in
        reverse, so no channel leaves its group, and applying the
        permutation twice should restore the order.
    """

    # Act
    perm = groups.channel_permutation_from_groups(sizes).tolist()

    # Assert
    offset = 0
    for size in sizes:
        assert perm[offset : offset + size] == list(range(offset, offset + size))[::-1]
        offset += size
    assert len(perm) == offset
    assert [perm[channel] for channel in perm] == list(range(offset))


@pytest.mark.parametrize(
    "sizes,error",
    [([], ValueError), ([0], ValueError), ([1, True], TypeError), (None, TypeError)],
)
def test_channel_permutation_from_groups_should_raise_when_sizes_are_unusable(
    sizes, error
):
    """Test that the permutation refuses what the check refuses.

    Given:
        An empty layout, a zero size, a boolean size or ``None``.
    When:
        The permutation is taken.
    Then:
        It should raise what the check raises for each, naming
        ``group_sizes``.
    """

    # Act & assert
    with pytest.raises(error, match="group_sizes"):
        groups.channel_permutation_from_groups(sizes)


def test_channel_permutation_from_groups_should_accept_numpy_sizes():
    """Test a layout of numpy integers.

    Given:
        Group sizes as numpy integers, as a layout read out of an array
        arrives.
    When:
        The permutation is taken.
    Then:
        It should be the permutation the same Python integers give.
    """

    # Act
    perm = groups.channel_permutation_from_groups([numpy.int64(1), numpy.int64(2)])

    # Assert
    assert perm.tolist() == groups.channel_permutation_from_groups([1, 2]).tolist()
