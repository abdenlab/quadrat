"""The two rules a training set is drawn under.

Provides `Jitter`, which says how far a peak's draw may be shifted
from its locus's centre, and `Exclusion`, which says which windows an
excluded region removes; and `as_jitter` and `as_exclusions`, which
resolve the shorthand the loader's arguments accept into them.

Nothing here reads a file or imports what the ``pileup`` extra
installs or ``torch``: `quadrat.loci`, which must not import
``torch``, and `quadrat.sampling`, which does, both read these.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`.
"""

import dataclasses
import numbers

#: Width of the bins an `Exclusion` tests on by default, in bases.
EXCLUSION_BIN_WIDTH = 100


@dataclasses.dataclass(frozen=True)
class Jitter:
    """Choose the shifts a peak's draw may take from its locus's centre.

    A draw is shifted by a whole number of bases drawn uniformly from
    ``low`` through ``high``, both included; a negative shift moves the
    window towards the contig's start. Every window is widened by
    `margin` on both sides, so every shift reads inside it, and a
    lopsided rule pays for its longer side on both: ``Jitter(0, 1000)``
    drops a locus within 1,000 bases of its contig's start as surely as
    ``Jitter(-1000, 1000)`` does.

    ``Jitter(-500, 500)`` is the rule upstream cherimoya has drawn
    since jmschrei/cherimoya#92. ``Jitter(-500, 499)`` is the rule
    before it, so a run seeded under it still draws what it drew; at a
    bound of ``0`` both rules are ``Jitter(0, 0)``, which is no jitter.

    A value: it compares, hashes and pickles by its two fields, which
    are held as `int` whatever integer type was given. It cannot be
    subclassed: every reader trusts `margin` to contain every shift.

    :param low: The smallest shift, in bases.
    :param high: The largest shift, in bases.
    :raises TypeError: if either bound is not an integer, a `bool`
        included.
    :raises ValueError: if ``low`` is above ``high``.
    """

    low: int
    high: int

    def __post_init__(self):
        for name in ("low", "high"):
            value = getattr(self, name)
            # `bool` is an `Integral`, and `True` read as one base is
            # the mistake this exists to refuse.
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError(
                    "Jitter.{} must be an integer, got {}.".format(
                        name, type(value).__name__
                    )
                )
            # Normalized so a numpy integer pickles and keys as one.
            object.__setattr__(self, name, int(value))
        if self.low > self.high:
            raise ValueError(
                "Jitter draws shifts from low through high, so low must not "
                "exceed high; got low={} and high={}.".format(self.low, self.high)
            )

    def __init_subclass__(cls, **kwargs):
        raise TypeError(
            "Jitter cannot be subclassed, and {} derives from it. Every "
            "reader trusts its margin to contain every shift; build one "
            "with the bounds you need instead.".format(cls.__name__)
        )

    @property
    def margin(self):
        """Return how far a window is widened on each side, in bases.

        The larger of ``-low`` and ``high``, and never below ``0``. A
        draw's offset into the widened window is its shift plus this, so
        offsets run from ``low + margin`` through ``high + margin``.
        """

        return max(-self.low, self.high, 0)


@dataclasses.dataclass(frozen=True, eq=False)
class Exclusion:
    """Choose which windows a set of excluded regions removes.

    The test is made on a grid of bins ``bin_width`` bases wide rather
    than on bases. A half-open interval ``[start, end)``, a region or a
    window alike, covers every bin from ``start // bin_width`` through
    ``(end + end_padding - 1) // bin_width``, and covers none when
    ``end + end_padding <= start``. A window is excluded when it covers
    a bin a region covers on the same contig.

    ``Exclusion(regions)`` is tangermeme 1.6's rule, which upstream
    cherimoya follows, for every window that is not empty: an interval
    covers the bins it overlaps, and an empty region covers none. An
    empty window covers none either, where tangermeme 1.6 has an empty
    window inside a bin cover that bin. ``end_padding=1`` is tangermeme
    1.5's for every region whose end is not before its start: an
    interval ending exactly on a bin edge also covers the bin beginning
    there, and an empty region or window covers the bin holding it.
    ``bin_width=1`` tests bases, and is the overlap a half-open BED tool
    reports.

    Compared and hashed by identity, since its regions may be frames,
    which do not compare as values; and frozen only in its fields, so
    a frame it holds can still be changed. It cannot be subclassed:
    its readers trust its fields as validated. ``bin_width`` and
    ``end_padding`` are held as `int` whatever integer type was given.
    The regions are checked where they are read, not here.

    :param regions: The excluded regions, in any shape
        `quadrat.loci.load_windows` takes loci in. A `list` is held as a
        `tuple`.
    :param bin_width: The width of the bins, in bases.
    :param end_padding: Bases added to every interval's end before it
        is binned.
    :raises TypeError: if ``bin_width`` or ``end_padding`` is not an
        integer, a `bool` included, or if ``regions`` is an
        `Exclusion`.
    :raises ValueError: if ``bin_width`` is below ``1`` or
        ``end_padding`` below ``0``.
    """

    regions: object
    bin_width: int = EXCLUSION_BIN_WIDTH
    end_padding: int = 0

    def __post_init__(self):
        if isinstance(self.regions, Exclusion):
            raise TypeError(
                "Exclusion.regions is the regions themselves, and got an "
                "Exclusion. Pass both in a list to apply both rules."
            )
        if isinstance(self.regions, list):
            object.__setattr__(self, "regions", tuple(self.regions))
        for name, floor in (("bin_width", 1), ("end_padding", 0)):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError(
                    "Exclusion.{} must be an integer, got {}.".format(
                        name, type(value).__name__
                    )
                )
            if value < floor:
                raise ValueError(
                    "Exclusion.{} must be at least {}, got {}.".format(
                        name, floor, value
                    )
                )
            object.__setattr__(self, name, int(value))

    def __init_subclass__(cls, **kwargs):
        raise TypeError(
            "Exclusion cannot be subclassed, and {} derives from it. Its "
            "rule is read as data; build one with the fields you need "
            "instead.".format(cls.__name__)
        )


def as_jitter(value):
    """Return a jitter argument as a `Jitter`.

    :param value: A `Jitter`, answered as it is, or an integer ``m``,
        answered as ``Jitter(-m, m)``.
    :returns: The `Jitter`.
    :raises TypeError: if ``value`` is neither, a `bool` included.
    :raises ValueError: if ``value`` is a negative integer.
    """

    if isinstance(value, Jitter):
        return value
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(
            "max_jitter must be a Jitter or an integer, got {}.".format(
                type(value).__name__
            )
        )
    if value < 0:
        raise ValueError("max_jitter must be non-negative, got {}".format(value))
    return Jitter(-int(value), int(value))


def as_exclusions(value):
    """Return an exclusion argument as a list of `Exclusion`.

    :param value: ``None``, answered as no exclusion; an `Exclusion`;
        a locus set in any shape `Exclusion` takes as its regions; or a
        `list` or `tuple` mixing `Exclusion` values and locus sets.
        Every locus set in a list is gathered, in order, into one
        `Exclusion` under the default rule, answered after the
        `Exclusion` values in the order they were given.
    :returns: The list, empty for ``None`` or an empty list or tuple.
    """

    if value is None:
        return []
    if isinstance(value, Exclusion):
        return [value]
    if not isinstance(value, (list, tuple)):
        return [Exclusion(value)]
    rules = [item for item in value if isinstance(item, Exclusion)]
    bare = [item for item in value if not isinstance(item, Exclusion)]
    if bare:
        rules.append(Exclusion(bare))
    return rules
