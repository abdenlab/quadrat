"""Signal groups: which channels turn into which under reverse-complement.

Provides `validate_signal_groups`, the check a group specification
passes; `normalize_signal_groups`, which reads a nested track
specification as its tracks and their group sizes; and
`channel_permutation_from_groups`, the channel permutation those sizes
imply for a mirrored draw.

Only `validate_signal_groups` is public. Nothing here imports a tensor
library with the module: the one function that answers a tensor imports
``torch`` on first use.
"""

import numbers

from quadrat._engine import import_on_first_use


# public
def validate_signal_groups(groups, label="signal_groups"):
    """Raise unless ``groups`` is a usable group specification.

    A group is a run of consecutive channels measuring one modality in
    one orientation each, e.g., an unstranded track on its own or a
    stranded ``(+, -)`` pair, and a specification states the size of
    every group in channel order. It is usable when it is a non-empty
    `list` or `tuple` of positive integers, so it describes at least
    one channel and no group without one. A size is any
    `numbers.Integral`, numpy's integers included, except a `bool`.

    :param groups: The candidate specification.
    :param label: The name the caller knows ``groups`` by, for the
        message.
    :raises TypeError: if ``groups`` is not a `list` or `tuple`, or
        holds anything but integers, naming the type it got.
    :raises ValueError: if ``groups`` is empty, or holds an integer
        below ``1``.
    """

    if not isinstance(groups, (list, tuple)):
        raise TypeError(
            "{} must be a list or tuple of positive integers; got {}.".format(
                label, type(groups).__name__
            )
        )
    if len(groups) == 0:
        raise ValueError("{} must be non-empty; got an empty list".format(label))
    for size in groups:
        if not isinstance(size, numbers.Integral) or isinstance(size, bool):
            raise TypeError(
                "{} entries must be integers; got {} in {!r}.".format(
                    label, type(size).__name__, list(groups)
                )
            )
        if size < 1:
            raise ValueError(
                "{} entries must be positive; got {!r} in {!r}.".format(
                    label, size, list(groups)
                )
            )


def normalize_signal_groups(signals):
    """Return a nested track specification as its tracks and their group sizes.

    Each entry of ``signals`` is one group. An entry that is a `list`
    or `tuple` is a group of the tracks it holds, in order; any other
    entry is one track in a group of its own. So a flat list of tracks
    is that many one-channel groups, and a stranded pair is written as
    a nested pair.

    The tracks themselves are not looked at: each is refused or read
    by whatever opens it.

    :param signals: A `list` or `tuple` of entries as above, or
        ``None`` for no tracks.
    :returns: ``(tracks, group_sizes)``. ``tracks`` is every track in
        channel order, as a `list`, and ``group_sizes`` the channel
        count of each group, in the same order; ``(None, [])`` where
        ``signals`` is ``None``.
    :raises TypeError: if ``signals`` is neither ``None`` nor a `list`
        or `tuple`, a single track included.
    :raises ValueError: if a group holds no track.

    An unstranded track beside a stranded pair, which answers the
    three tracks in that order and the sizes ``[1, 2]``::

        normalize_signal_groups(["atac", ["ctcf.+", "ctcf.-"]])

    .. rubric:: Implementation notes

    A single track is refused by type rather than wrapped, for the
    reason `quadrat.cache.track_list` gives.
    """

    if signals is None:
        return None, []
    if not isinstance(signals, (list, tuple)):
        raise TypeError(
            "a track specification is a list or tuple with one entry per "
            "group, each entry a track or a list of tracks. Got {}. Pass a "
            "one-element list for one channel.".format(type(signals).__name__)
        )

    tracks = []
    group_sizes = []
    for entry in signals:
        if isinstance(entry, (list, tuple)):
            if len(entry) == 0:
                raise ValueError(
                    "a signal group is empty: every group holds at least one track."
                )
            tracks.extend(entry)
            group_sizes.append(len(entry))
        else:
            tracks.append(entry)
            group_sizes.append(1)
    return tracks, group_sizes


def channel_permutation_from_groups(group_sizes):
    """Return the channel permutation a mirrored draw applies to its signal.

    Each group's channels are reversed in place and the groups keep
    their order, so a one-channel group stays where it is, a ``(+, -)``
    pair swaps, and no channel leaves its group. The permutation is
    its own inverse.

    :param group_sizes: The size of every group in channel order, as
        `validate_signal_groups` accepts them.
    :returns: A `torch.Tensor` of dtype ``int64`` and shape
        ``(sum(group_sizes),)``, ``perm``, such that
        ``signal[perm].flip(-1)`` is the reverse-complement of a
        ``(channels, length)`` signal.
    :raises TypeError: if ``group_sizes`` is of a type
        `validate_signal_groups` refuses.
    :raises ValueError: if ``group_sizes`` holds a value
        `validate_signal_groups` refuses.
    """

    torch = import_on_first_use("torch")
    validate_signal_groups(group_sizes, label="group_sizes")
    perm = []
    offset = 0
    for size in group_sizes:
        perm.extend(range(offset + size - 1, offset - 1, -1))
        offset += size
    return torch.tensor(perm, dtype=torch.long)
