"""What a pileup counts, as one value.

Provides `Aggregate`, the record `quadrat.pileup.pileup_alignments`
takes in place of loose keywords. It is data and nothing else, so it
imports and constructs without the ``pileup`` extra.
"""

import dataclasses
import math
import numbers
import sys

from quadrat.tables import COUNTS, COVERAGE_DELTAS

#: The position rules `Aggregate` accepts, in the order its docstring
#: states them.
POSITIONS = ("five_prime", "three_prime", "midpoint", "span")


#: Flags that exclude a record by default: unmapped (``0x4``),
#: secondary (``0x100``), QC-failed (``0x200``), duplicate (``0x400``)
#: and supplementary (``0x800``).
DEFAULT_EXCLUDE_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800


# public
@dataclasses.dataclass(frozen=True, kw_only=True)
class Aggregate:
    """Choose which records a pileup counts, at which positions, and how.

    A frozen record of three choices. The *filter* decides which
    records qualify, over the flags, the mapping quality and the
    template length. The *position rule* decides which positions a
    qualifying record contributes. What accumulates follows from the
    rule: a rule naming one position per record accumulates a count
    there, and ``"span"`` accumulates coverage over the reference span
    the record is aligned to.

    Being a value, it compares, hashes and pickles by its fields, so
    it crosses into a `quadrat.pileup_many` worker process
    unchanged whatever started the process. Every field is validated
    when it is constructed, so a record that exists is one a pileup
    can run, and every field is passed by keyword. It cannot be
    subclassed: a pileup reads its fields as data and trusts the
    validation, which a subclass could override. Building one needs
    nothing the ``pileup`` extra installs; running a pileup with it
    does.

    **Coordinates.** ``start`` is a record's leftmost aligned reference
    base, 0-based, and ``end`` is one past its rightmost; soft clips
    are not reference, and deletions and skipped regions between them
    are inside the span. A record's *5' end* is ``start`` on the
    forward strand and ``end - 1`` on the reverse (flag ``0x10``), and
    its *3' end* is the other one.

    **Position rules.**

    ``"five_prime"``
        The 5' end, after the shifts.

    ``"three_prime"``
        The 3' end, after the shifts.

    ``"midpoint"``
        The middle of the fragment a pair spans, from the one record of
        the pair whose template length is positive. The fragment runs
        ``template_length`` bases from that record's 5' end towards its
        mate, i.e., ``[start, start + tlen)`` forward and ``[end -
        tlen, end)`` reverse, so it is the same fragment whichever mate
        carries the positive length. The position is its leftmost base
        plus ``(tlen - 1) // 2``: the base at the centre, or the lower
        of the two when the length is even. A record whose template
        length is zero or negative contributes nothing, so an unpaired
        record never does.

    ``"span"``
        Every base from ``start`` to ``end``, each gaining one. The
        track records this as two signed boundary events per record,
        ``+1`` at ``start`` and ``-1`` at ``end``, so its values are
        coverage deltas; see `quadrat.read_manifest` for how they read
        back.

    A position outside its contig is dropped, as is a record on a
    contig the reference does not name. ``"span"`` clips a record's
    span to its contig instead: one ending on or past the contig's last
    base closes with a ``-1`` at the contig's length, and one starting
    at or past the length contributes nothing, so each contig's deltas
    sum to zero. A record reaches past its contig only where the
    alignment's header declares the contig longer than the reference
    does. A stranded pileup puts each contribution on the
    strand of the record that made it. ``"midpoint"`` is always
    unstranded: the record it is taken from is the mate carrying the
    positive template length, which in a forward-reverse pair is almost
    always the forward one, so its strand says nothing about the
    fragment's.

    Integer fields and ``scale_factor`` are held as `int` and `float`,
    ``position`` as `str` and ``unstranded`` as `bool`, whatever numpy
    scalar or `str` subclass was given, so a record pickles and is
    recorded as plain Python values.

    :param exclude_flags: A record carrying any of these flags is
        skipped. The default skips unmapped, secondary, QC-failed,
        duplicate and supplementary records, so a fragment contributes
        through its primary alignments only. An unmapped record never
        contributes, whatever this says.
    :param require_flags: A record lacking any of these flags is
        skipped. ``64`` keeps the first mate of each pair alone, so a
        pair contributes once under ``"five_prime"`` rather than twice.
    :param min_mapq: A record whose mapping quality is below this is
        skipped. Quality 255, which SAM reserves for "unavailable", is
        compared as the number it is.
    :param min_template_length: A record whose absolute template length
        is below this is skipped, so any value above zero also skips
        every record that has no template length.
    :param max_template_length: A record whose absolute template length
        is above this is skipped. ``None`` sets no ceiling. Under
        ``"midpoint"`` it must be at least ``1``, since only a record
        with a positive template length contributes there.
    :param position: The position rule, one of ``"five_prime"``,
        ``"three_prime"``, ``"midpoint"`` or ``"span"``, as above.
    :param start_shift: Added to ``start`` before a position is taken
        from it. It therefore moves a forward record's 5' end and a
        reverse record's 3' end; ATAC's Tn5 offset is ``start_shift=4,
        end_shift=-4`` under ``"five_prime"``.
    :param end_shift: Added to ``end`` before a position is taken from
        it, so it moves a reverse record's 5' end and a forward
        record's 3' end.
    :param unstranded: Pool both strands into one track rather than
        writing one per strand. ``None``, the default, pools them under
        ``"midpoint"`` and keeps them apart under every other rule;
        ``False`` under ``"midpoint"`` is refused. Held as a `bool`
        whichever is given.
    :param scale_factor: Multiplier applied to every value once the
        values have accumulated. Under ``"span"`` it is recorded as the
        track's ``scale`` rather than applied to the deltas written;
        see `quadrat.read_manifest`.
    :raises TypeError: if a field is passed by position; if a flag
        mask, a quality, a length or a shift is not an integer, if
        ``unstranded`` is neither ``None`` nor a `bool` (a numpy
        ``numpy.bool_`` is one), or if ``scale_factor`` is not a real
        number; or on defining a subclass.
    :raises ValueError: if ``position`` is not one of the four rules; if
        a flag mask is outside ``0`` to ``0xFFFF``; if ``require_flags``
        shares a flag with ``exclude_flags`` or asks for ``0x4``, so
        that no record could qualify; if a quality or a length is
        negative, the ceiling is below the floor, or ``"midpoint"`` is
        given a ceiling below ``1``; if a shift is given with
        ``"midpoint"`` or ``"span"``, which take no shift; if
        ``"midpoint"`` is given ``unstranded=False``; or if
        ``scale_factor`` is not finite, or is too large to hold as a
        `float`.
    """

    exclude_flags: int = DEFAULT_EXCLUDE_FLAGS
    require_flags: int = 0
    min_mapq: int = 0
    min_template_length: int = 0
    max_template_length: int | None = None
    position: str = "five_prime"
    start_shift: int = 0
    end_shift: int = 0
    unstranded: bool | None = None
    scale_factor: float = 1.0

    def __post_init__(self):
        integers = [
            "exclude_flags",
            "require_flags",
            "min_mapq",
            "min_template_length",
            "start_shift",
            "end_shift",
        ]
        if self.max_template_length is not None:
            integers.append("max_template_length")
        for name in integers:
            value = getattr(self, name)
            # `bool` is an `Integral`, and `True` read as a mask of one
            # is the mistake this exists to refuse.
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError(
                    "Aggregate.{} must be an integer, got {}.".format(
                        name, type(value).__name__
                    )
                )
            # Normalized so a numpy integer pickles and records as one.
            object.__setattr__(self, name, int(value))
        if self.unstranded is not None:
            if not _is_bool(self.unstranded):
                raise TypeError(
                    "Aggregate.unstranded must be a bool or None, got {}.".format(
                        type(self.unstranded).__name__
                    )
                )
            object.__setattr__(self, "unstranded", bool(self.unstranded))
        if not isinstance(self.scale_factor, numbers.Real) or isinstance(
            self.scale_factor, bool
        ):
            raise TypeError(
                "Aggregate.scale_factor must be a real number, got {}.".format(
                    type(self.scale_factor).__name__
                )
            )
        try:
            scale_factor = float(self.scale_factor)
        except OverflowError:
            # A real number too large for a float, e.g., ``10**400``, is
            # no finite float a pileup could multiply by, so it takes the
            # documented refusal rather than surfacing `OverflowError`.
            raise ValueError(
                "Aggregate.scale_factor must be finite, and this {} is too "
                "large to hold as a float.".format(type(self.scale_factor).__name__)
            ) from None
        object.__setattr__(self, "scale_factor", scale_factor)

        if self.position not in POSITIONS:
            raise ValueError(
                "Aggregate.position must be one of {}, got {!r}.".format(
                    ", ".join(repr(rule) for rule in POSITIONS), self.position
                )
            )
        object.__setattr__(self, "position", str(self.position))
        if self.unstranded is None:
            object.__setattr__(self, "unstranded", self.position == "midpoint")
        for name in ("exclude_flags", "require_flags"):
            if not 0 <= getattr(self, name) <= 0xFFFF:
                raise ValueError(
                    "Aggregate.{} is a mask over SAM's 16 flag bits, so it "
                    "must be between 0 and 0xFFFF; got {}.".format(
                        name, getattr(self, name)
                    )
                )
        clash = self.require_flags & (self.exclude_flags | 0x4)
        if clash:
            raise ValueError(
                "Aggregate requires flags {:#x} that it also excludes, so no "
                "record could qualify (an unmapped record, 0x4, is always "
                "excluded).".format(clash)
            )
        for name in ("min_mapq", "min_template_length"):
            if getattr(self, name) < 0:
                raise ValueError(
                    "Aggregate.{} must not be negative, got {}.".format(
                        name, getattr(self, name)
                    )
                )
        if (
            self.max_template_length is not None
            and self.max_template_length < self.min_template_length
        ):
            raise ValueError(
                "Aggregate.max_template_length {} is below "
                "min_template_length {}, so no record could qualify.".format(
                    self.max_template_length, self.min_template_length
                )
            )
        if (
            self.position == "midpoint"
            and self.max_template_length is not None
            and self.max_template_length < 1
        ):
            raise ValueError(
                "Aggregate.max_template_length {} admits no record under "
                "'midpoint', which counts only a record whose template "
                "length is positive, so no record could qualify.".format(
                    self.max_template_length
                )
            )
        if self.position in ("midpoint", "span") and (
            self.start_shift or self.end_shift
        ):
            raise ValueError(
                "Aggregate.position {!r} takes no shift: a shift moves one "
                "end of an alignment, and {} is not taken from one end. Got "
                "start_shift={} and end_shift={}.".format(
                    self.position,
                    "a fragment's middle"
                    if self.position == "midpoint"
                    else "a span's coverage",
                    self.start_shift,
                    self.end_shift,
                )
            )
        if self.position == "midpoint" and not self.unstranded:
            raise ValueError(
                "Aggregate.position 'midpoint' is always unstranded: the "
                "midpoint is taken from the mate with the positive template "
                "length, which is almost always the forward one, so a strand "
                "split would put nearly every fragment on '+'."
            )
        if not math.isfinite(self.scale_factor):
            raise ValueError(
                "Aggregate.scale_factor must be finite, got {}.".format(
                    self.scale_factor
                )
            )

    def __init_subclass__(cls, **kwargs):
        raise TypeError(
            "Aggregate cannot be subclassed, and {} derives from it. A "
            "pileup reads an Aggregate's fields as data and relies on its "
            "validation; build one with the fields you need instead.".format(
                cls.__name__
            )
        )

    @property
    def signal(self):
        """Return what the values of a track this writes mean.

        ``"coverage_deltas"`` under ``"span"`` and ``"counts"`` under
        every other rule, as the manifest of every track a pileup
        writes records it; see `quadrat.read_manifest`.
        """

        return COVERAGE_DELTAS if self.position == "span" else COUNTS

    @property
    def _points_per_record(self):
        """The most rows one record contributes to the grouping.

        Two for coverage deltas, a record's two boundary events, and one
        for counts. It is the ``points_per_record`` the planner takes;
        `quadrat.pileup.plan.group_floor` and
        `quadrat.pileup.plan.projected_peak` state what it changes.
        """

        return 2 if self.signal == COVERAGE_DELTAS else 1

    @property
    def _manifest_scale(self):
        """The ``scale`` a track this writes records, or ``None``.

        ``scale_factor`` for coverage deltas, which are written unscaled
        and scaled once summed, and ``None`` for counts, whose values
        carry the scale already; see `quadrat.read_manifest`.
        """

        return self.scale_factor if self.signal == COVERAGE_DELTAS else None


def _is_bool(value):
    """Whether ``value`` is a `bool` or a numpy `numpy.bool_`.

    Imports nothing, numpy included.

    .. rubric:: Implementation notes

    numpy is asked for through `sys.modules` rather than imported: a
    numpy scalar exists only once numpy is loaded, and this module
    imports nothing a record does not need.
    """

    numpy = sys.modules.get("numpy")
    return isinstance(value, bool) or (
        numpy is not None and isinstance(value, numpy.bool_)
    )
