"""Tests for locus loading and extraction.

The contract is that a locus set becomes the windows its coordinates
imply and the tensors the files hold at those windows, and that a
contig-naming mismatch between any two inputs is refused rather than
answered with nothing. So the assertions are of four kinds: windows,
masks and tensors derived independently of the code under test; the
same extraction made by tangermeme; the rows of the composition
against what the readers it composes read at the same windows; and a
refusal whose message names both sides of the mismatch.

Three cases the refusals must leave alone each have a test of their
own: a window contig with no position on it, a count filter that keeps
nothing, and an exclusion list that keeps nothing.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import os
import pathlib
import sys

import numpy
import pandas
import polars
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from tangermeme.io import extract_loci as tangermeme_extract_loci

from quadrat import loci
from tests import (
    TWO_CONTIGS,
    UNREADABLE_IDS,
    UNREADABLE_RECORDS,
    builders,
    dense,
    one_hot,
    parse_fasta,
    quadrat_tables,
    run_child,
    window_frame,
)

IN_WINDOW, OUT_WINDOW = 60, 30


# Six-column peaks, so a reader has to take the leading three. Each
# row is a boundary of some rule at an input window of 60 and a jitter
# of 5, which widen a locus to 35 bases either side of its centre.
PEAKS = [
    ("chr2", 200, 260, "peak", 0, "."),  # ordinary, and the only chr2 locus
    ("chr1", 100, 140, "peak", 0, "."),  # ordinary, over an N run
    ("chr1", 20, 40, "peak", 0, "."),  # starts before the contig
    ("chr1", 960, 1000, "peak", 0, "."),  # ends past the contig
    ("chr1", 955, 975, "peak", 0, "."),  # ends exactly at the contig's length
    ("chr1", 690, 720, "peak", 0, "."),  # excluded only through a bin edge
    ("chr1", 800, 840, "peak", 0, "."),  # over the soft-masked run
    ("chr1", 930, 940, "peak", 0, "."),  # ordinary, near the end
    ("chr1", 540, 560, "peak", 0, "."),  # inside the excluded region
]


# The peaks on which every release of tangermeme agrees with this
# package. The sixth is left out: its window starts in the bin that
# begins where the excluded region ends, which a region marks under
# the rule `exclusion_mask` documents and does not mark under the rule
# tangermeme 1.6 adopted. That rule has a test of its own below.
AGREED_PEAKS = PEAKS[:5] + PEAKS[6:]


EXCLUSION = [("chr1", 500, 600)]


SOFT_MASK = (("chr1", 790, 850),)


N_RUNS = (("chr1", 110, 120), ("chr2", 220, 230))


TRACK_ROWS = builders.scattered_rows(TWO_CONTIGS, 3)


CONTROL_ROWS = builders.scattered_rows(TWO_CONTIGS, 4)


# Coordinates drawn on and beside the edges of the 100-base bins, which
# is where the bin rule differs from a test on bases: uniform
# coordinates land on an edge once in a hundred draws.
bin_coordinates = st.builds(
    lambda bin_index, offset: bin_index * 100 + offset,
    st.integers(min_value=0, max_value=3),
    st.sampled_from([0, 1, 99]),
)


interval_rows = st.lists(
    st.tuples(st.sampled_from(["a", "b"]), bin_coordinates, bin_coordinates),
    min_size=1,
    max_size=6,
)


# A few contig names for a track or a window set to be drawn over.
contig_names = st.sampled_from(["chr1", "chr2", "chr3", "1"])


position_rows = st.lists(
    st.tuples(
        st.sampled_from(["a", "b"]),
        st.integers(min_value=0, max_value=120),
        st.integers(min_value=1, max_value=9),
    ),
    max_size=20,
)


# Loci as ``(contig, start, width)`` on the module's two contigs, some
# starting past a contig's end, with the window geometry to load them
# at.
locus_rows = st.lists(
    st.tuples(
        st.sampled_from(sorted(TWO_CONTIGS)),
        st.integers(min_value=0, max_value=1200),
        st.integers(min_value=0, max_value=90),
    ),
    max_size=8,
)
in_windows = st.integers(min_value=1, max_value=130)
out_windows = st.integers(min_value=0, max_value=130)
jitters = st.integers(min_value=0, max_value=25)


# One to four locus sets of up to five ``(contig, start, end)`` loci
# each, every locus on a contig the module's reference names, and a
# contig filter to load them under.
locus_sets = st.lists(
    st.lists(
        st.builds(
            lambda chrom, start, width: (chrom, start, start + width),
            st.sampled_from(sorted(TWO_CONTIGS)),
            st.integers(min_value=0, max_value=400),
            st.integers(min_value=1, max_value=60),
        ),
        max_size=5,
    ),
    min_size=1,
    max_size=4,
)
contig_filters = st.sampled_from([None, ["chr1"], ["chr2"], ["chr1", "chr2"]])


# One track's positions on the module's two contigs, each on its
# contig, for an in-memory track to be built from. Dense enough that a
# window's count on one track is often on the other side of a low
# floor from its count on another.
track_positions = st.lists(
    st.one_of(
        *(
            st.tuples(
                st.just(chrom),
                st.integers(min_value=0, max_value=size - 1),
                st.integers(min_value=1, max_value=4),
            )
            for chrom, size in sorted(TWO_CONTIGS.items())
        )
    ),
    min_size=20,
    max_size=60,
)


# Two signal tracks' positions and an index naming either, positive or
# negative; one or two control tracks' positions; and a count floor or
# none, drawn low enough to fall among the counts a window of those
# positions has rather than above all of them.
signal_tracks = st.lists(track_positions, min_size=2, max_size=2)
target_indices = st.integers(min_value=-2, max_value=1)
control_tracks = st.lists(track_positions, min_size=1, max_size=2)
count_floors = st.none() | st.integers(min_value=0, max_value=8)


# Counts to filter, any of them not a number, with a floor and a
# ceiling that are each absent, any number, or one of the counts, so
# a bound lands exactly on a count as often as not.
@st.composite
def count_bands(draw):
    counts = draw(st.lists(st.floats(allow_infinity=False, width=32), max_size=12))
    numbers = [count for count in counts if not numpy.isnan(count)]
    bound = st.none() | st.floats(allow_nan=False, allow_infinity=False, width=32)
    if numbers:
        bound = bound | st.sampled_from(numbers)
    return counts, draw(bound), draw(bound)


# Regions to exclude, as ``(contig, start, end)``, or no exclusion
# list at all.
exclusion_rows = st.none() | st.lists(
    st.builds(
        lambda chrom, start, width: (chrom, start, start + width),
        st.sampled_from(sorted(TWO_CONTIGS)),
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=150),
    ),
    min_size=1,
    max_size=3,
)


@pytest.fixture(scope="session")
def engine():
    """Return the engine module, skipping the requesting test without it."""

    pytest.importorskip("giql")
    return pytest.importorskip("polars_bio")


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Build the reference, peaks, exclusion list and two tracks the module reads."""

    return corpus_factory(
        "loci",
        TWO_CONTIGS,
        soft_mask=SOFT_MASK,
        n_runs=N_RUNS,
        peaks=PEAKS,
        exclusion=EXCLUSION,
        tracks=[TRACK_ROWS, CONTROL_ROWS],
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        max_jitter=5,
    )


@pytest.fixture
def no_engine(mocker):
    """Make the engine's two modules unimportable for one test."""

    mocker.patch.dict(sys.modules, {"polars_bio": None, "giql": None})


def dense_signal(channels, rows):
    """Return the signal tensor a list of channels implies over window rows.

    Built by testing every position against every window, with no
    grouping and no join, so it shares nothing with the code under
    test.
    """

    width = rows[0][2] - rows[0][1] if rows else 0
    out = numpy.zeros((len(rows), len(channels), width), dtype=numpy.float32)
    for channel, positions in enumerate(channels):
        for chrom, pos, value in positions:
            for index, (on, start, end) in enumerate(rows):
                if on == chrom and start <= pos < end:
                    out[index, channel, pos - start] += value
    return torch.from_numpy(out)


def raised_without_engine(statement, *arguments):
    """Return the type name and message of what a statement raises with no engine.

    The statement runs in a fresh interpreter in which neither module
    of the ``pileup`` extra will import, with ``loci``, ``pandas`` and
    ``polars`` bound and ``arguments`` as ``sys.argv[1:]``. Both are
    empty strings where it raises nothing.
    """

    program = (
        "import sys\n"
        "sys.modules['polars_bio'] = None\n"
        "sys.modules['giql'] = None\n"
        "import pandas, polars\n"
        "from quadrat import loci\n"
        "try:\n"
        "    {}\n"
        "except Exception as error:\n"
        "    print(type(error).__name__)\n"
        "    print(error)\n"
    ).format(statement)
    result = run_child(
        [sys.executable, "-c", program, *arguments], "the child interpreter failed"
    )
    name, _, message = result.stdout.partition("\n")
    return name, message


def round_robin(locus_sets):
    """Return the loci of several sets taken in turn, one from each set per round.

    The first locus of each set in the order the sets are given, then
    the second of each, and so on, a set that has run out being passed
    over.
    """

    order = []
    for depth in range(max((len(loci_of) for loci_of in locus_sets), default=0)):
        for loci_of in locus_sets:
            if depth < len(loci_of):
                order.append(loci_of[depth])
    return order


def touches(window, regions, width=100):
    """Return whether a window shares a bin with any region on its contig.

    The rule `exclusion_mask` documents, restated as a set intersection
    over the bins each side reaches.
    """

    chrom, start, end = window
    reached = set(range(start // width, end // width + 1))
    return any(
        on == chrom and reached & set(range(lo // width, hi // width + 1))
        for on, lo, hi in regions
    )


class TestExtractedLoci:
    """The record `extract_loci` answers with."""

    def test___new___should_answer_its_fields_by_name_and_by_position(self):
        """Test that the record reads both as a tuple and by field.

        Given:
            Four values standing for the four tensors.
        When:
            A record is built from them and unpacked.
        Then:
            It should answer them in the order sequences, signals,
            controls and kept, and each under its own name.
        """

        # Act
        record = loci.ExtractedLoci("X", "y", "c", "k")
        sequences, signals, controls, kept = record

        # Assert
        assert (sequences, signals, controls, kept) == ("X", "y", "c", "k")
        assert (record.sequences, record.signals) == ("X", "y")
        assert (record.controls, record.kept) == ("c", "k")


def test_load_windows_should_derive_every_bound_from_the_locus_centre(corpus):
    """Test the arithmetic of one window.

    Given:
        One locus over ``[100, 140)`` on ``chr1``, an input window of
        60, an output window of 30 and a jitter of 5.
    When:
        Its windows are loaded.
    Then:
        It should centre on 120, and bound the output window at 15 and
        the input window at 30 either side of it, each widened by 5.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 100, 140)])

    # Act
    windows = loci.load_windows(locus, corpus.fasta, 60, 30, max_jitter=5)

    # Assert
    assert windows.to_dicts() == [
        {
            "chrom": "chr1",
            "mid": 120,
            "outer_start": 85,
            "outer_end": 155,
            "in_bounds": True,
            "sig_start": 100,
            "sig_end": 140,
            "in_start": 85,
            "in_end": 155,
        }
    ]


def test_load_windows_should_answer_the_documented_columns_and_dtypes(corpus):
    """Test the schema of the window frame.

    Given:
        The module's peak file.
    When:
        Its windows are loaded.
    Then:
        It should answer a polars frame whose columns and dtypes are
        the documented ones, in the documented order.
    """

    # Act
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30)

    # Assert
    assert isinstance(windows, polars.DataFrame)
    assert list(windows.schema.items()) == [
        ("chrom", polars.Utf8),
        ("mid", polars.Int64),
        ("outer_start", polars.Int64),
        ("outer_end", polars.Int64),
        ("in_bounds", polars.Boolean),
        ("sig_start", polars.Int64),
        ("sig_end", polars.Int64),
        ("in_start", polars.Int64),
        ("in_end", polars.Int64),
    ]


def test_load_windows_should_put_the_extra_base_of_an_odd_width_on_the_right(corpus):
    """Test where an odd window sits around its centre.

    Given:
        One locus centred on 120, an input window of 61 and an output
        window of 31.
    When:
        Its windows are loaded.
    Then:
        It should reach 30 and 15 bases left of the centre and one
        base further on the right, so each window is its full width.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 100, 140)])

    # Act
    (window,) = loci.load_windows(locus, corpus.fasta, 61, 31).to_dicts()

    # Assert
    assert (window["in_start"], window["in_end"]) == (90, 151)
    assert (window["sig_start"], window["sig_end"]) == (105, 136)


def test_load_windows_should_keep_every_locus_and_mark_the_ones_off_their_contig(corpus):
    """Test that a window off its contig is marked rather than dropped.

    Given:
        The module's nine peaks, of which one window starts before its
        contig, one ends past it and one ends exactly at its length.
    When:
        Their windows are loaded at a jitter of 5.
    Then:
        It should answer nine rows in file order, with only the two
        windows reaching outside their contig marked out of bounds.
    """

    # Act
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, max_jitter=5)

    # Assert
    assert windows["chrom"].to_list() == [row[0] for row in PEAKS]
    assert windows["in_bounds"].to_list() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_load_windows_should_count_the_odd_base_when_marking_bounds(corpus):
    """Test that an odd window's last base decides whether it fits.

    Given:
        A locus centred 30 bases before the end of ``chr1``.
    When:
        Its windows are loaded at an input width of 60 and again at 61.
    Then:
        It should be in bounds at 60, which ends exactly at the contig's
        length, and out of bounds at 61, which ends one base past it.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 960, 980)])

    # Act
    even = loci.load_windows(locus, corpus.fasta, 60, 0)
    odd = loci.load_windows(locus, corpus.fasta, 61, 0)

    # Assert
    assert (even["outer_end"].to_list(), even["in_bounds"].to_list()) == ([1000], [True])
    assert (odd["outer_end"].to_list(), odd["in_bounds"].to_list()) == ([1001], [False])


def test_load_windows_should_let_the_wider_window_decide_the_bounds(corpus):
    """Test which window bounds the union.

    Given:
        A locus centred 40 bases into ``chr1`` and an input window of
        60.
    When:
        Its windows are loaded with no output window and again with an
        output window of 100.
    Then:
        It should be in bounds with no output window, and out of bounds
        once the output window reaches before the contig's start.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 30, 50)])

    # Act
    alone = loci.load_windows(locus, corpus.fasta, 60, 0)
    wider = loci.load_windows(locus, corpus.fasta, 60, 100)

    # Assert
    assert (alone["outer_start"].to_list(), alone["in_bounds"].to_list()) == (
        [10],
        [True],
    )
    assert (wider["outer_start"].to_list(), wider["in_bounds"].to_list()) == (
        [-10],
        [False],
    )


@pytest.mark.parametrize("shape", [str, pathlib.Path, os.fsencode])
def test_load_windows_should_accept_every_path_shape(corpus, shape):
    """Test that the locus file may be named by any path type.

    Given:
        The module's peak file named by a `str`, a `pathlib.Path` or
        `bytes`.
    When:
        Its windows are loaded.
    Then:
        It should answer one window per peak.
    """

    # Act
    windows = loci.load_windows(shape(corpus.peaks), corpus.fasta, 60, 30)

    # Assert
    assert windows["mid"].to_list() == [row[1] + (row[2] - row[1]) // 2 for row in PEAKS]


def test_load_windows_should_read_a_compressed_file(corpus, tmp_path):
    """Test that a gzipped locus file reads as a plain one.

    Given:
        The module's peaks written gzipped.
    When:
        Its windows are loaded.
    Then:
        It should answer the windows the plain file gives.
    """

    # Arrange
    packed = builders.write_bed(str(tmp_path / "peaks.bed.gz"), PEAKS)

    # Act
    windows = loci.load_windows(packed, corpus.fasta, 60, 30)

    # Assert
    assert windows.equals(loci.load_windows(corpus.peaks, corpus.fasta, 60, 30))


def test_load_windows_should_skip_lines_that_hold_no_interval(corpus, tmp_path):
    """Test that header lines are not read as loci.

    Given:
        A locus file opening with a ``track`` line, a ``browser`` line
        and a comment ahead of two intervals.
    When:
        Its windows are loaded.
    Then:
        It should answer the two intervals and nothing else.
    """

    # Arrange
    path = tmp_path / "headed.bed"
    path.write_text(
        "track name=peaks\n"
        "browser position chr1:1-100\n"
        "# a comment\n"
        "chr1\t100\t140\n"
        "chr2\t200\t260\n"
    )

    # Act
    windows = loci.load_windows(str(path), corpus.fasta, 60, 30)

    # Assert
    assert windows.select("chrom", "mid").rows() == [("chr1", 120), ("chr2", 230)]


@pytest.mark.parametrize(
    "content",
    [
        "\ntrack name=peaks\nchr1\t100\t140\nchr2\t200\t260\n",
        "track name=peaks\n\nbrowser position chr1:1-100\n\n# a comment\n"
        "chr1\t100\t140\nchr2\t200\t260\n",
    ],
    ids=["blank-before-a-header", "blank-between-headers"],
)
def test_load_windows_should_skip_blank_lines_among_the_header_lines(
    corpus, tmp_path, content
):
    """Test that a blank line does not move which lines are skipped.

    Given:
        A locus file of two intervals under header lines, with a blank
        line ahead of the ``track`` line, or blank lines between the
        header lines.
    When:
        Its windows are loaded.
    Then:
        It should answer the two intervals and nothing else.
    """

    # Arrange
    path = tmp_path / "spaced.bed"
    path.write_text(content)

    # Act
    windows = loci.load_windows(str(path), corpus.fasta, 60, 30)

    # Assert
    assert windows.select("chrom", "mid").rows() == [("chr1", 120), ("chr2", 230)]


def test_load_windows_should_read_a_frame_as_it_reads_a_file(corpus):
    """Test the in-memory form of a locus set.

    Given:
        The module's peaks as a pandas frame with unnamed columns.
    When:
        Its windows are loaded.
    Then:
        It should answer the windows the file gives.
    """

    # Arrange
    frame = pandas.DataFrame(PEAKS)

    # Act
    windows = loci.load_windows(frame, corpus.fasta, 60, 30)

    # Assert
    assert windows.equals(loci.load_windows(corpus.peaks, corpus.fasta, 60, 30))


def test_load_windows_should_read_a_numeric_contig_name_as_text(tmp_path):
    """Test that a contig named by a number matches the reference's name.

    Given:
        A reference naming a contig ``1`` and a locus file naming it
        the same way, which a table reader takes for an integer.
    When:
        Its windows are loaded.
    Then:
        It should place the locus on the contig ``1`` as a string.
    """

    # Arrange
    fasta = builders.write_fasta(str(tmp_path / "genome.fa"), {"1": 400})
    bed = builders.write_bed(str(tmp_path / "loci.bed"), [(1, 100, 140)])

    # Act
    windows = loci.load_windows(bed, fasta, 60, 30)

    # Assert
    assert windows["chrom"].to_list() == ["1"]
    assert windows["in_bounds"].to_list() == [True]


@pytest.mark.parametrize(
    "header", ["", "track name=peaks\n"], ids=["intervals-only", "under-a-header"]
)
def test_load_windows_should_keep_a_zero_padded_contig_name_as_written(tmp_path, header):
    """Test that a contig name made of digits is not read as a number.

    Given:
        A reference naming ``chr1``, ``01`` and ``1``, and a locus file
        whose one locus is on ``01``, with or without a header line.
    When:
        Its windows are loaded.
    Then:
        It should place the locus on ``01`` and not on ``1``.
    """

    # Arrange
    fasta = builders.write_fasta(
        str(tmp_path / "genome.fa"), {"chr1": 400, "01": 400, "1": 400}
    )
    bed = tmp_path / "loci.bed"
    bed.write_text(header + "01\t100\t200\n")

    # Act
    windows = loci.load_windows(str(bed), fasta, 60, 30)

    # Assert
    assert windows["chrom"].to_list() == ["01"]


def test_load_windows_should_read_a_polars_frame_as_it_reads_a_file(corpus):
    """Test the polars form of a locus set.

    Given:
        The module's peaks as a polars frame.
    When:
        Its windows are loaded.
    Then:
        It should answer the windows the file gives.
    """

    # Arrange
    frame = polars.DataFrame(PEAKS, orient="row")

    # Act
    windows = loci.load_windows(frame, corpus.fasta, 60, 30)

    # Assert
    assert windows.equals(loci.load_windows(corpus.peaks, corpus.fasta, 60, 30))


@settings(max_examples=60, deadline=None)
@given(rows=locus_rows, in_window=in_windows, out_window=out_windows, max_jitter=jitters)
def test_load_windows_should_read_back_as_loci_the_windows_it_answered(
    corpus, rows, in_window, out_window, max_jitter
):
    """Test that a frame of the kind this answers is one it takes, for any geometry.

    Given:
        The contig and input-window columns, as a polars frame, of the
        windows loaded from any loci on the module's two contigs, some
        past a contig's end, at any window widths and jitter.
    When:
        Windows of the same widths and jitter are loaded with that
        frame as the loci.
    Then:
        It should answer the same windows, each input window being
        centred where its locus was.
    """

    # Arrange
    frame = pandas.DataFrame(
        {
            "chrom": [row[0] for row in rows],
            "start": [row[1] for row in rows],
            "end": [row[1] + row[2] for row in rows],
        }
    )
    geometry = dict(in_window=in_window, out_window=out_window, max_jitter=max_jitter)
    windows = loci.load_windows(frame, corpus.fasta, **geometry)
    as_loci = windows.select("chrom", "in_start", "in_end")

    # Act
    again = loci.load_windows(as_loci, corpus.fasta, **geometry)

    # Assert
    assert again.equals(windows)


def test_load_windows_should_interleave_locus_sets(corpus):
    """Test the row order of several locus sets.

    Given:
        A set of three loci and a set of one.
    When:
        Windows are loaded over both.
    Then:
        It should answer the first of each set, then the rest of the
        longer one, in that order.
    """

    # Arrange
    first = pandas.DataFrame(
        [("chr1", 100, 140), ("chr1", 200, 240), ("chr1", 300, 340)]
    )
    second = pandas.DataFrame([("chr2", 100, 140)])

    # Act
    windows = loci.load_windows([first, second], corpus.fasta, 60, 30)

    # Assert
    assert windows.select("chrom", "mid").rows() == [
        ("chr1", 120),
        ("chr2", 120),
        ("chr1", 220),
        ("chr1", 320),
    ]


def test_load_windows_should_filter_each_set_before_interleaving(corpus):
    """Test that the contig filter runs ahead of the interleaving.

    Given:
        Two locus sets, the first opening with a locus on a contig the
        filter drops.
    When:
        Windows are loaded over both, keeping ``chr1``.
    Then:
        It should interleave what the filter left of each set, so the
        first set's surviving locus leads.
    """

    # Arrange
    first = pandas.DataFrame([("chr2", 100, 140), ("chr1", 200, 240)])
    second = pandas.DataFrame([("chr1", 300, 340), ("chr1", 400, 440)])

    # Act
    windows = loci.load_windows([first, second], corpus.fasta, 60, 30, chroms=["chr1"])

    # Assert
    assert windows["mid"].to_list() == [220, 320, 420]


@settings(max_examples=60, deadline=None)
@given(sets=locus_sets, chroms=contig_filters)
def test_load_windows_should_interleave_any_locus_sets_in_turn(corpus, sets, chroms):
    """Test the row order of several locus sets against a round-robin over them.

    Given:
        One to four locus sets of none to five loci each on the
        module's two contigs, under no contig filter or one naming some
        of those contigs that leaves some locus wherever the sets hold
        one.
    When:
        Windows are loaded over the sets.
    Then:
        It should answer, in row order, the centres of the loci the
        filter leaves, taken one from each set in turn in the order
        the sets were given.
    """

    # Arrange
    filtered = [
        [locus for locus in loci_of if chroms is None or locus[0] in chroms]
        for loci_of in sets
    ]
    assume(any(filtered) or not any(sets))
    frames = [
        pandas.DataFrame(loci_of, columns=["chrom", "start", "end"]) for loci_of in sets
    ]

    # Act
    windows = loci.load_windows(frames, corpus.fasta, 60, 30, chroms=chroms)

    # Assert
    assert windows.select("chrom", "mid").rows() == [
        (chrom, start + (end - start) // 2)
        for chrom, start, end in round_robin(filtered)
    ]


def test_load_windows_should_take_the_contig_filter_as_any_collection(corpus):
    """Test that the filter need not be a list.

    Given:
        The module's peaks and a `set` naming ``chr2``.
    When:
        Windows are loaded.
    Then:
        It should answer the one locus on that contig.
    """

    # Act
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, chroms={"chr2"})

    # Assert
    assert windows.select("chrom", "mid").rows() == [("chr2", 230)]


def test_load_windows_should_centre_on_the_summit_when_asked(corpus, tmp_path):
    """Test the summit rule.

    Given:
        A ten-column locus over ``[300, 400)`` whose summit offset is
        10.
    When:
        Its windows are loaded with and without summits.
    Then:
        It should centre on 310 with summits and on 350 without.
    """

    # Arrange
    bed = builders.write_bed(
        str(tmp_path / "summits.bed"),
        [("chr1", 300, 400, "peak", 0, ".", 0, 0, 0, 10)],
    )

    # Act
    on_summit = loci.load_windows(bed, corpus.fasta, 60, 30, summits=True)
    on_centre = loci.load_windows(bed, corpus.fasta, 60, 30)

    # Assert
    assert on_summit["mid"].to_list() == [310]
    assert on_centre["mid"].to_list() == [350]


@pytest.mark.parametrize(
    ("offset", "message"),
    [(-1, "Summits cannot be negative"), (101, "cannot be larger than end")],
)
def test_load_windows_should_raise_when_a_summit_lies_outside_its_locus(
    corpus, tmp_path, offset, message
):
    """Test the refusal for a summit that is not on its locus.

    Given:
        A locus of 100 bases whose summit offset is negative or past
        its end.
    When:
        Its windows are loaded with summits.
    Then:
        It should raise `ValueError` saying which of the two it is.
    """

    # Arrange
    bed = builders.write_bed(
        str(tmp_path / "summits.bed"),
        [("chr1", 300, 400, "peak", 0, ".", 0, 0, 0, offset)],
    )

    # Act & assert
    with pytest.raises(ValueError, match=message):
        loci.load_windows(bed, corpus.fasta, 60, 30, summits=True)


@pytest.mark.parametrize("given", [42, [42], ["peaks.bed", None]])
def test_load_windows_should_raise_type_error_when_loci_is_no_locus_set(corpus, given):
    """Test the refusal for loci of no shape a locus set takes.

    Given:
        An integer, or a list holding something that is neither a path
        nor a frame.
    When:
        Windows are loaded.
    Then:
        It should raise `TypeError` naming the argument and the type
        it was given.
    """

    # Act & assert
    with pytest.raises(
        TypeError, match=r"`loci` is a BED file path.*Got builtins\.(int|NoneType)\."
    ):
        loci.load_windows(given, corpus.fasta, 60, 30)


def test_load_windows_should_name_the_library_of_a_frame_it_refuses(corpus):
    """Test that the refusal tells one library's frame from another's.

    Given:
        The module's peaks as a polars lazy frame, which is no shape a
        locus set takes.
    When:
        Windows are loaded.
    Then:
        It should raise `TypeError` naming the type by its module, so
        the frame is not mistaken for one of the two it accepts.
    """

    # Arrange
    lazy = polars.DataFrame(PEAKS, orient="row").lazy()

    # Act & assert
    with pytest.raises(
        TypeError, match=r"a pandas or polars DataFrame.*Got polars\.\S*LazyFrame\."
    ):
        loci.load_windows(lazy, corpus.fasta, 60, 30)


def test_load_windows_should_raise_type_error_when_chroms_is_a_single_name(corpus):
    """Test the refusal for a filter given as one name.

    Given:
        The contig filter ``"chr1"`` rather than a collection.
    When:
        Windows are loaded.
    Then:
        It should raise `TypeError` showing the one-element list to
        pass instead.
    """

    # Act & assert
    with pytest.raises(TypeError, match=r"`chroms` is a collection.*Pass \['chr1'\]"):
        loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, chroms="chr1")


def test_load_windows_should_raise_type_error_when_sequences_is_no_path(corpus):
    """Test the refusal for a reference that is not a file.

    Given:
        An in-memory mapping in place of the reference.
    When:
        Windows are loaded.
    Then:
        It should raise `TypeError` naming the type it was given.
    """

    # Act & assert
    with pytest.raises(TypeError, match="got dict"):
        loci.load_windows(corpus.peaks, {"chr1": "ACGT"}, 60, 30)


def test_load_windows_should_raise_file_not_found_error_when_the_reference_is_missing(
    tmp_path, recwarn
):
    """Test the refusal for a reference that is not there.

    Given:
        A locus and a path at which there is no reference.
    When:
        Its windows are loaded.
    Then:
        It should raise `FileNotFoundError` naming the path, without
        first warning of a scan for the reference's lengths.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 100, 140)])

    # Act & assert
    with pytest.raises(FileNotFoundError, match=r"^\S*missing\.fa is not a reference"):
        loci.load_windows(locus, str(tmp_path / "missing.fa"), 60, 30)
    assert not [caught for caught in recwarn.list if "scanning" in str(caught.message)]


def test_load_windows_should_raise_file_not_found_error_when_a_locus_file_is_missing(
    corpus, tmp_path
):
    """Test the refusal for a locus file that is not there.

    Given:
        A reference, and a path at which there is no locus file.
    When:
        Its windows are loaded.
    Then:
        It should raise `FileNotFoundError` naming the path.
    """

    # Arrange
    missing = str(tmp_path / "missing.bed")

    # Act & assert
    with pytest.raises(FileNotFoundError, match="missing.bed"):
        loci.load_windows(missing, corpus.fasta, 60, 30)


def test_load_windows_should_raise_when_chroms_names_no_contig_the_loci_use(corpus):
    """Test the refusal for a filter and loci on different conventions.

    Given:
        The module's peaks, named ``chr1`` and ``chr2``, and a filter
        naming ``1`` and ``2``.
    When:
        Windows are loaded.
    Then:
        It should raise `ValueError` naming the locus file, the number
        of loci it holds, the contigs the loci name and the contigs the
        filter names.
    """

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"^none of the 9 loci in \S*peaks\.bed is on a contig `chroms` "
        r"names: the loci name `chr2`, `chr1` and `chroms` names `1`, `2`\.",
    ):
        loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, chroms=["1", "2"])


def test_load_windows_should_name_an_in_memory_locus_set_as_a_frame(corpus):
    """Test how the filter refusal names loci that have no path.

    Given:
        Loci held as a frame and a filter matching none of them.
    When:
        Windows are loaded.
    Then:
        It should raise `ValueError` describing the loci as the frame
        given rather than by a path.
    """

    # Arrange
    frame = pandas.DataFrame(PEAKS)

    # Act & assert
    with pytest.raises(
        ValueError, match="^none of the 9 loci in the locus frame given is on"
    ):
        loci.load_windows(frame, corpus.fasta, 60, 30, chroms=["1"])


def test_load_windows_should_attribute_its_count_to_files_and_frames_alike(corpus):
    """Test how the filter refusal names loci held partly in memory.

    Given:
        The module's nine peaks as a file beside a frame of two more
        loci, and a filter matching none of the eleven.
    When:
        Windows are loaded.
    Then:
        It should raise `ValueError` stating the count of both sources
        together and naming both the file and the frame as holding it.
    """

    # Arrange
    frame = pandas.DataFrame([("chr1", 100, 140), ("chr2", 100, 140)])

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"^none of the 11 loci in \S*peaks\.bed and 1 locus frame is on",
    ):
        loci.load_windows([corpus.peaks, frame], corpus.fasta, 60, 30, chroms=["1"])


@pytest.mark.parametrize("content", ["", "# nothing called\ntrack name=empty\n"])
def test_load_windows_should_answer_no_windows_when_the_locus_file_is_empty(
    corpus, tmp_path, content
):
    """Test that an empty locus file is not blamed on the filter.

    Given:
        A locus file holding no interval, either empty or holding only
        header lines, and a contig filter.
    When:
        Windows are loaded.
    Then:
        It should answer a frame with no rows, raising nothing.
    """

    # Arrange
    path = tmp_path / "empty.bed"
    path.write_text(content)

    # Act
    windows = loci.load_windows(str(path), corpus.fasta, 60, 30, chroms=["chr1"])

    # Assert
    assert windows.height == 0


def test_load_windows_should_answer_no_windows_when_the_locus_frame_is_empty(corpus):
    """Test that an empty in-memory locus set is not blamed on the filter.

    Given:
        A frame with the three locus columns and no rows, and a contig
        filter.
    When:
        Windows are loaded.
    Then:
        It should answer a frame with no rows, raising nothing.
    """

    # Arrange
    empty = pandas.DataFrame({"chrom": [], "start": [], "end": []})

    # Act
    windows = loci.load_windows(empty, corpus.fasta, 60, 30, chroms=["chr1"])

    # Assert
    assert windows.height == 0


def test_load_windows_should_raise_when_a_locus_is_on_a_contig_the_reference_lacks(
    corpus, tmp_path
):
    """Test the refusal for a locus the reference cannot place.

    Given:
        A locus file with one locus on ``chr1`` and one on ``chrZ``,
        which the reference does not name.
    When:
        Windows are loaded.
    Then:
        It should raise `ValueError` naming the locus file, the
        reference, the contig that is absent and the contigs the
        reference holds.
    """

    # Arrange
    bed = builders.write_bed(
        str(tmp_path / "has_chrZ.bed"), [("chr1", 100, 140), ("chrZ", 100, 140)]
    )

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"^1 of the 2 loci in \S*has_chrZ\.bed cannot be placed: "
        r"\S*genome\.fa does not name `chrZ`\. The reference names `chr1`, "
        r"`chr2`\.",
    ):
        loci.load_windows(bed, corpus.fasta, 60, 30)


@pytest.mark.packaging
def test_load_windows_should_run_without_the_pileup_extra(corpus):
    """Test that loading windows needs no engine.

    Given:
        A fresh interpreter in which neither module of the ``pileup``
        extra will import.
    When:
        The module's peaks are loaded into windows.
    Then:
        It should answer one window per peak.
    """

    # Arrange
    program = (
        "import sys\n"
        "sys.modules['polars_bio'] = None\n"
        "sys.modules['giql'] = None\n"
        "from quadrat import loci\n"
        "print(len(loci.load_windows(sys.argv[1], sys.argv[2], 60, 30)))\n"
    )

    # Act
    result = run_child(
        [sys.executable, "-c", program, corpus.peaks, corpus.fasta],
        "the child interpreter failed",
    )

    # Assert
    assert result.stdout.strip() == str(len(PEAKS))


@pytest.mark.packaging
def test_load_windows_should_raise_import_error_when_only_a_scan_gives_the_lengths(
    unindexable_fasta,
):
    """Test the one case in which loading windows needs the engine.

    Given:
        A reference with no index in a directory that cannot be written
        to, in a fresh interpreter in which neither module of the
        ``pileup`` extra will import.
    When:
        Windows are loaded over a locus.
    Then:
        It should raise `ImportError` naming the ``pileup`` extra.
    """

    # Act
    name, message = raised_without_engine(
        "loci.load_windows(pandas.DataFrame([('chr1', 100, 140)]), sys.argv[1], 60, 30)",
        unindexable_fasta,
    )

    # Assert
    assert name == "ImportError"
    assert "`pileup` extra" in message


@settings(max_examples=60, deadline=None)
@given(rows=locus_rows, in_window=in_windows, out_window=out_windows, max_jitter=jitters)
def test_load_windows_should_bound_every_window_consistently_for_any_geometry(
    corpus, rows, in_window, out_window, max_jitter
):
    """Test the invariants tying a window's columns together.

    Given:
        Any loci on the module's two contigs, some past a contig's end,
        and any window widths and jitter.
    When:
        Their windows are loaded.
    Then:
        Every locus should have a row; each window should be its
        nominal width plus the jitter on both sides and hold the
        centre; the union should span exactly both windows; and a row
        should be in bounds exactly when the union lies on its contig.
    """

    # Arrange
    frame = pandas.DataFrame(
        {
            "chrom": [row[0] for row in rows],
            "start": [row[1] for row in rows],
            "end": [row[1] + row[2] for row in rows],
        }
    )

    # Act
    windows = loci.load_windows(
        frame, corpus.fasta, in_window, out_window, max_jitter=max_jitter
    )

    # Assert
    assert windows.height == len(rows)
    for (chrom, start, width), window in zip(rows, windows.to_dicts()):
        assert window["mid"] == start + width // 2
        assert window["in_end"] - window["in_start"] == in_window + 2 * max_jitter
        assert window["sig_end"] - window["sig_start"] == out_window + 2 * max_jitter
        assert window["in_start"] <= window["mid"] - max_jitter
        assert window["sig_start"] <= window["mid"] - max_jitter
        assert window["outer_start"] == min(window["in_start"], window["sig_start"])
        assert window["outer_end"] == max(window["in_end"], window["sig_end"])
        assert window["in_bounds"] == (
            window["outer_start"] >= 0 and window["outer_end"] <= TWO_CONTIGS[chrom]
        )


def test_exclusion_mask_should_flag_the_windows_sharing_a_bin_with_a_region(
    corpus, engine
):
    """Test the mask over the module's peaks.

    Given:
        The module's nine windows at a jitter of 5 and a region over
        ``[500, 600)`` of ``chr1``.
    When:
        The mask is taken.
    Then:
        It should flag the window inside the region and the one
        starting in the bin that begins where the region ends, and no
        other.
    """

    # Arrange
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, max_jitter=5)

    # Act
    mask = loci.exclusion_mask(windows, [corpus.exclusion])

    # Assert
    assert mask.dtype == numpy.bool_
    assert mask.tolist() == [False, False, False, False, False, True, False, False, True]


def test_exclusion_mask_should_reach_the_bin_that_begins_where_a_region_ends(engine):
    """Test the documented reach of a region ending on a bin edge.

    Given:
        A region over ``[500, 600)``, a window over ``[650, 699)`` in
        the bin beginning at 600, and one over ``[700, 760)`` in the
        bin after.
    When:
        The mask is taken.
    Then:
        It should flag the first window and not the second.
    """

    # Arrange
    windows = window_frame(
        [("chr1", 650, 699), ("chr1", 700, 760)], ("outer_start", "outer_end")
    )
    regions = pandas.DataFrame([("chr1", 500, 600)])

    # Act
    mask = loci.exclusion_mask(windows, regions)

    # Assert
    assert mask.tolist() == [True, False]


def test_exclusion_mask_should_reach_the_bin_that_begins_where_a_window_ends(engine):
    """Test the documented reach of a window ending on a bin edge.

    Given:
        A region over ``[500, 510)``, a window over ``[430, 500)``
        ending on the edge of the region's bin, and one over
        ``[430, 499)`` ending a base short of it.
    When:
        The mask is taken.
    Then:
        It should flag the first window and not the second.
    """

    # Arrange
    windows = window_frame(
        [("chr1", 430, 500), ("chr1", 430, 499)], ("outer_start", "outer_end")
    )
    regions = pandas.DataFrame([("chr1", 500, 510)])

    # Act
    mask = loci.exclusion_mask(windows, regions)

    # Assert
    assert mask.tolist() == [True, False]


def test_exclusion_mask_should_flag_nothing_across_contigs(engine):
    """Test that a region excludes on its own contig alone.

    Given:
        A window on ``chr1`` and regions covering the same coordinates
        on ``chr2`` and on a contig no window is on.
    When:
        The mask is taken.
    Then:
        It should flag nothing.
    """

    # Arrange
    windows = window_frame([("chr1", 100, 160)], ("outer_start", "outer_end"))
    regions = pandas.DataFrame([("chr2", 0, 500), ("chrUn", 0, 500)])

    # Act
    mask = loci.exclusion_mask(windows, regions)

    # Assert
    assert mask.tolist() == [False]


@pytest.mark.parametrize("form", ["path", "list", "frame", "polars", "plain"])
def test_exclusion_mask_should_take_regions_in_every_shape_loci_come_in(
    corpus, engine, tmp_path, form
):
    """Test the forms an exclusion list may be given in.

    Given:
        One region as a gzipped path, a list of that path, a pandas
        frame, a polars frame, or an uncompressed path.
    When:
        The mask is taken over the module's windows.
    Then:
        It should flag the same two windows from every form.
    """

    # Arrange
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30, max_jitter=5)
    regions = {
        "path": corpus.exclusion,
        "list": [corpus.exclusion],
        "frame": pandas.DataFrame(EXCLUSION),
        "polars": polars.DataFrame(EXCLUSION, orient="row"),
        "plain": builders.write_bed(str(tmp_path / "exclusion.bed"), EXCLUSION),
    }[form]

    # Act
    mask = loci.exclusion_mask(windows, regions)

    # Assert
    assert numpy.flatnonzero(mask).tolist() == [5, 8]


@pytest.mark.parametrize("regions", [[], pandas.DataFrame({"c": [], "s": [], "e": []})])
def test_exclusion_mask_should_flag_nothing_when_there_is_no_region(corpus, regions):
    """Test the mask over an exclusion list that excludes nothing.

    Given:
        The module's windows, and no exclusion list or an empty one.
    When:
        The mask is taken.
    Then:
        It should answer one false entry per window.
    """

    # Arrange
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30)

    # Act
    mask = loci.exclusion_mask(windows, regions)

    # Assert
    assert mask.tolist() == [False] * len(PEAKS)


def test_exclusion_mask_should_answer_an_empty_mask_when_there_is_no_window(corpus):
    """Test the mask over an empty window frame.

    Given:
        A window frame with no rows and a region.
    When:
        The mask is taken.
    Then:
        It should answer a boolean array of length zero.
    """

    # Arrange
    windows = window_frame([], ("outer_start", "outer_end"))

    # Act
    mask = loci.exclusion_mask(windows, [corpus.exclusion])

    # Assert
    assert (mask.dtype, mask.shape) == (numpy.bool_, (0,))


def test_exclusion_mask_should_raise_type_error_when_windows_is_no_polars_frame(corpus):
    """Test the refusal for windows held in another library's frame.

    Given:
        Windows as a pandas frame.
    When:
        The mask is taken.
    Then:
        It should raise `TypeError` naming the function and the type
        it was given.
    """

    # Arrange
    windows = pandas.DataFrame(
        {"chrom": ["chr1"], "outer_start": [0], "outer_end": [60]}
    )

    # Act & assert
    with pytest.raises(
        TypeError, match="exclusion_mask takes a polars DataFrame.*Got DataFrame"
    ):
        loci.exclusion_mask(windows, [corpus.exclusion])


def test_exclusion_mask_should_raise_type_error_when_a_region_set_is_no_locus_set():
    """Test the refusal for an exclusion list of no shape a locus set takes.

    Given:
        A window and an integer in place of the exclusion lists.
    When:
        The mask is taken.
    Then:
        It should raise `TypeError` naming the argument.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 60)], ("outer_start", "outer_end"))

    # Act & assert
    with pytest.raises(
        TypeError, match=r"`exclusion_lists` is a BED file path.*Got builtins\.int\."
    ):
        loci.exclusion_mask(windows, 42)


def test_exclusion_mask_should_raise_file_not_found_error_when_a_list_is_missing(
    tmp_path,
):
    """Test the refusal for an exclusion list that is not there.

    Given:
        A window, and a path at which there is no exclusion list.
    When:
        The mask is taken.
    Then:
        It should raise `FileNotFoundError` naming the path.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 60)], ("outer_start", "outer_end"))

    # Act & assert
    with pytest.raises(FileNotFoundError, match="missing.bed"):
        loci.exclusion_mask(windows, str(tmp_path / "missing.bed"))


def test_exclusion_mask_should_leave_no_table_bound(corpus, engine):
    """Test that the join's frames do not outlive it.

    Given:
        The session's tables before a mask is taken.
    When:
        A mask is taken over the module's windows.
    Then:
        It should leave the session with the same tables.
    """

    # Arrange
    windows = loci.load_windows(corpus.peaks, corpus.fasta, 60, 30)
    before = quadrat_tables(engine)

    # Act
    loci.exclusion_mask(windows, [corpus.exclusion])

    # Assert
    assert quadrat_tables(engine) == before


def test_exclusion_mask_should_raise_import_error_when_the_extra_is_absent(
    corpus, no_engine
):
    """Test what a mask needs that the base install lacks.

    Given:
        Windows and a region, in a process where the engine will not
        import.
    When:
        The mask is taken.
    Then:
        It should raise `ImportError` naming the ``pileup`` extra.
    """

    # Arrange
    windows = window_frame([("chr1", 500, 560)], ("outer_start", "outer_end"))

    # Act & assert
    with pytest.raises(ImportError, match="`pileup` extra"):
        loci.exclusion_mask(windows, [corpus.exclusion])


@settings(max_examples=60, deadline=None)
@given(window_rows=interval_rows, region_rows=interval_rows)
def test_exclusion_mask_should_match_a_bin_by_bin_test_for_any_intervals(
    engine, window_rows, region_rows
):
    """Test the mask against the rule it documents, over arbitrary inputs.

    Given:
        At least one window and one region over two contigs, each
        bounded on or beside a bin edge and of any width down to zero.
    When:
        The mask is taken.
    Then:
        It should flag exactly the windows whose bins intersect the
        bins of a region on the same contig.
    """

    # Arrange
    windows = [(chrom, min(lo, hi), max(lo, hi)) for chrom, lo, hi in window_rows]
    regions = [(chrom, min(lo, hi), max(lo, hi)) for chrom, lo, hi in region_rows]

    # Act
    mask = loci.exclusion_mask(
        window_frame(windows, ("outer_start", "outer_end")),
        pandas.DataFrame(regions, columns=["chrom", "start", "end"]),
    )

    # Assert
    assert mask.tolist() == [touches(window, regions) for window in windows]


def test_readable_windows_should_keep_the_windows_in_bounds_that_no_region_excludes(
    corpus, engine
):
    """Test both tests of readability over the module's peaks.

    Given:
        The module's nine peaks at a jitter of 5 and its exclusion
        list.
    When:
        Their readable windows are taken.
    Then:
        It should answer the frame `load_windows` answers, and keep
        the windows on their contig apart from the one inside the
        excluded region and the one reaching it through a bin edge.
    """

    # Act
    windows, keep = loci.readable_windows(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        max_jitter=5,
        exclusion_lists=[corpus.exclusion],
    )

    # Assert
    assert windows.equals(
        loci.load_windows(
            corpus.peaks, corpus.fasta, IN_WINDOW, OUT_WINDOW, max_jitter=5
        )
    )
    assert keep.dtype == numpy.bool_
    assert keep.tolist() == [
        True,  # chr2 200-260
        True,  # chr1 100-140
        False,  # chr1 20-40 starts before the contig
        False,  # chr1 960-1000 ends past the contig
        True,  # chr1 955-975 ends exactly at its length
        False,  # chr1 690-720 reaches the region through a bin edge
        True,  # chr1 800-840
        True,  # chr1 930-940
        False,  # chr1 540-560 is inside the region
    ]


def test_readable_windows_should_keep_every_window_in_bounds_when_nothing_is_excluded(
    corpus,
):
    """Test readability without an exclusion list.

    Given:
        The module's nine peaks at a jitter of 5 and no exclusion list.
    When:
        Their readable windows are taken.
    Then:
        It should keep exactly the windows `load_windows` marks
        ``in_bounds``, the two inside or beside the region included.
    """

    # Act
    windows, keep = loci.readable_windows(
        corpus.peaks, corpus.fasta, IN_WINDOW, OUT_WINDOW, max_jitter=5
    )

    # Assert
    assert keep.tolist() == windows["in_bounds"].to_list()
    assert keep.tolist() == [True, True, False, False, True, True, True, True, True]


@settings(max_examples=200)
@given(band=count_bands())
def test_within_counts_should_keep_a_count_inside_the_band_or_not_a_number(band):
    """Test the count band over arbitrary counts and bounds.

    Given:
        Up to twelve counts, any of them not a number, and a floor and
        a ceiling that are each ``None``, any number, or one of the
        counts.
    When:
        The counts are tested against the band.
    Then:
        It should answer one boolean per count, true exactly where the
        count is not a number or lies within the bounds given, both
        ends included.
    """

    # Arrange
    counts, min_counts, max_counts = band

    # Act
    within = loci.within_counts(counts, min_counts, max_counts)

    # Assert
    expected = [
        bool(numpy.isnan(count))
        or (
            (min_counts is None or count >= min_counts)
            and (max_counts is None or count <= max_counts)
        )
        for count in counts
    ]
    assert isinstance(within, numpy.ndarray)
    assert within.dtype == numpy.bool_
    assert within.tolist() == expected


def test_require_track_contigs_should_raise_when_no_track_shares_a_contig(corpus):
    """Test the refusal for tracks and windows on different conventions.

    Given:
        Two tracks naming ``chr1`` and ``chr2``, and windows on ``1``
        and ``2``.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `ValueError` naming the label, the contigs the
        tracks name, the contigs the windows name and the tracks.
    """

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"no `signals` track holds a position on any contig the windows "
        r"are on: the tracks name `chr1`, `chr2` and the windows name `1`, "
        r"`2`\..*The tracks are `\S*t0\.track\.parquet`, `\S*t1\.track\.parquet`\.",
    ):
        loci.require_track_contigs(["1", "2"], corpus.tracks)


def test_require_track_contigs_should_accept_a_track_of_coverage_deltas(tmp_path):
    """Test that the contig check reads names, whatever a track's values mean.

    Given:
        A complete track of coverage deltas on the windows' contig.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should return without raising, since it reads which contigs
        a track holds and no value.
    """

    # Arrange
    root = builders.write_track(
        str(tmp_path / "deltas.track.parquet"),
        [("chr1", 10, 1.0), ("chr1", 40, -1.0)],
        signal="coverage_deltas",
    )

    # Act & assert
    assert loci.require_track_contigs(["chr1"], [root]) is None


def test_require_track_contigs_should_name_the_label_and_in_memory_tracks():
    """Test how the refusal names tracks that have no path.

    Given:
        An in-memory track on ``chr2``, windows on ``chr1`` and the
        label ``controls``.
    When:
        The track is required to share a contig with the windows.
    Then:
        It should raise `ValueError` naming the label and describing
        the tracks as in-memory frames.
    """

    # Arrange
    elsewhere = builders.track_rows_frame([("chr2", 220, 3.0)])

    # Act & assert
    with pytest.raises(
        ValueError, match=r"no `controls` track.*The tracks are in-memory frames\."
    ):
        loci.require_track_contigs(["chr1"], [elsewhere], label="controls")


def test_require_track_contigs_should_count_the_tracks_it_does_not_list(corpus):
    """Test how the refusal names more tracks than it spells out.

    Given:
        Four track paths, drawn from a generator, none holding a
        position on the windows' contig.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `ValueError` spelling out the first three
        paths and counting the one it leaves out.
    """

    # Arrange
    tracks = (path for path in corpus.tracks * 2)

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"The tracks are `\S*t0\.track\.parquet`, `\S*t1\.track\.parquet`, "
        r"`\S*t0\.track\.parquet` and 1 more\.",
    ):
        loci.require_track_contigs(["1"], tracks)


def test_require_track_contigs_should_say_so_when_the_tracks_name_no_contig():
    """Test how the refusal reads for tracks that hold nothing.

    Given:
        A track with no rows and windows on ``chr1``.
    When:
        The track is required to share a contig with the windows.
    Then:
        It should raise `ValueError` saying the tracks name no contigs
        rather than leaving the list blank.
    """

    # Arrange
    nothing = builders.track_rows_frame([])

    # Act & assert
    with pytest.raises(ValueError, match="the tracks name no contigs and the windows"):
        loci.require_track_contigs(["chr1"], [nothing])


def test_require_track_contigs_should_accept_one_quiet_track_beside_another():
    """Test that a track holding nothing on the windows' contigs is ordinary.

    Given:
        Windows on ``chr1``, one track holding a position there and
        one holding positions only on ``chr2``.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise nothing.
    """

    # Arrange
    tracks = [
        builders.track_rows_frame([("chr2", 5, 1.0)]),
        builders.track_rows_frame([("chr1", 5, 1.0)]),
    ]

    # Act & assert
    assert loci.require_track_contigs(["chr1"], tracks) is None


def test_require_track_contigs_should_accept_a_contig_no_track_holds_beside_another():
    """Test that a window contig with no position on it is ordinary.

    Given:
        Windows on ``chr1`` and ``chr2`` and a track holding positions
        only on ``chr1``.
    When:
        The track is required to share a contig with the windows.
    Then:
        It should raise nothing.
    """

    # Arrange
    tracks = [builders.track_rows_frame([("chr1", 5, 1.0)])]

    # Act & assert
    assert loci.require_track_contigs(["chr1", "chr2"], tracks) is None


@pytest.mark.parametrize(
    ("chroms", "tracks"),
    [([], [builders.track_rows_frame([("chr2", 5, 1.0)])]), (["chr1"], [])],
    ids=["no-contigs", "no-tracks"],
)
def test_require_track_contigs_should_check_nothing_when_a_side_is_empty(chroms, tracks):
    """Test that there is no mismatch without two sides.

    Given:
        No window contigs beside a track, or a window contig beside no
        tracks.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise nothing.
    """

    # Act & assert
    assert loci.require_track_contigs(chroms, tracks) is None


@pytest.mark.parametrize("unresolvable", ["missing", "unverified", "integer"])
def test_require_track_contigs_should_resolve_no_track_when_no_contig_is_given(
    tmp_path, unresolvable
):
    """Test that an empty contig list settles the call before any track is opened.

    Given:
        No window contigs, beside a track that would be refused if it
        were opened: a path at which nothing exists, a track directory
        with no manifest, or an integer.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise nothing.
    """

    # Arrange
    unverified = tmp_path / "t.track.parquet"
    unverified.mkdir()
    track = {
        "missing": str(tmp_path / "mistyped.parquet"),
        "unverified": str(unverified),
        "integer": 42,
    }[unresolvable]

    # Act & assert
    assert loci.require_track_contigs([], [track]) is None


@settings(max_examples=100, deadline=None)
@given(
    held=st.lists(st.lists(contig_names, unique=True), min_size=0, max_size=3),
    chroms=st.lists(contig_names, unique=True),
)
def test_require_track_contigs_should_raise_exactly_when_no_track_meets_a_contig(
    held, chroms
):
    """Test the one rule the track check applies, over any tracks and contigs.

    Given:
        Up to three in-memory tracks, each holding a position on a
        drawn subset of a few contig names, possibly none, and a drawn
        collection of window contigs, possibly empty.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `ValueError` if and only if there are window
        contigs, there are tracks, and no track holds a position on
        any of the window contigs.
    """

    # Arrange
    tracks = [
        builders.track_rows_frame([(name, 5, 1.0) for name in names]) for names in held
    ]
    expected = (
        bool(chroms)
        and bool(held)
        and not any(set(names) & set(chroms) for names in held)
    )

    # Act
    try:
        loci.require_track_contigs(chroms, tracks)
    except ValueError:
        raised = True
    else:
        raised = False

    # Assert
    assert raised == expected


def test_require_track_contigs_should_raise_type_error_when_a_track_is_no_track():
    """Test the refusal for something that is not a track.

    Given:
        An integer among the tracks.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `TypeError` naming the type it was given.
    """

    # Act & assert
    with pytest.raises(TypeError, match="Got int"):
        loci.require_track_contigs(["chr1"], [42])


def test_require_track_contigs_should_raise_when_a_track_path_does_not_exist(tmp_path):
    """Test the refusal for a track path with nothing at it.

    Given:
        Windows on ``chr1``, a track holding a position there, and
        after it a single-file track path at which nothing exists.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `FileNotFoundError` naming the path, although
        the first track alone shares a contig with the windows.
    """

    # Arrange
    tracks = [
        builders.track_rows_frame([("chr1", 5, 1.0)]),
        str(tmp_path / "mistyped.parquet"),
    ]

    # Act & assert
    with pytest.raises(FileNotFoundError, match=r"mistyped\.parquet is not a track"):
        loci.require_track_contigs(["chr1"], tracks)


def test_require_track_contigs_should_raise_when_a_track_directory_does_not_verify(
    tmp_path,
):
    """Test the refusal for a track directory that cannot be verified.

    Given:
        Windows on ``chr1``, a track holding a position there, and
        after it a track directory with no manifest in it.
    When:
        The tracks are required to share a contig with the windows.
    Then:
        It should raise `ValueError` naming the directory, although
        the first track alone shares a contig with the windows.
    """

    # Arrange
    unverifiable = tmp_path / "t.track.parquet"
    unverifiable.mkdir()
    tracks = [builders.track_rows_frame([("chr1", 5, 1.0)]), str(unverifiable)]

    # Act & assert
    with pytest.raises(ValueError, match=r"t\.track\.parquet"):
        loci.require_track_contigs(["chr1"], tracks)


def test_require_track_contigs_should_raise_type_error_when_chroms_is_a_single_name():
    """Test the refusal for the windows' contigs given as one name.

    Given:
        The contig ``"1"`` rather than a collection, beside a track
        holding a position on ``1``.
    When:
        The track is required to share a contig with the windows.
    Then:
        It should raise `TypeError` showing the one-element list to
        pass instead.
    """

    # Arrange
    tracks = [builders.track_rows_frame([("1", 5, 1.0)])]

    # Act & assert
    with pytest.raises(TypeError, match=r"`chroms` is a collection.*Pass \['1'\]"):
        loci.require_track_contigs("1", tracks)


@pytest.mark.parametrize("form", ["path", "frame"])
def test_require_track_contigs_should_raise_type_error_when_given_a_single_track(
    corpus, form
):
    """Test the refusal for one track where a collection of them is taken.

    Given:
        A single track, as a path or as a frame, rather than a list.
    When:
        It is required to share a contig with the windows.
    Then:
        It should raise `TypeError` naming the function and saying to
        pass a one-element list.
    """

    # Arrange
    single = corpus.tracks[0] if form == "path" else builders.track_rows_frame([])

    # Act & assert
    with pytest.raises(
        TypeError,
        match="require_track_contigs reads one track per channel.*one-element list",
    ):
        loci.require_track_contigs(["chr1"], single)


def test_extract_signal_should_place_each_position_at_its_offset(corpus, engine):
    """Test the dense signal against the rows it was read from.

    Given:
        The module's two tracks and three windows over two contigs.
    When:
        The signal is extracted.
    Then:
        It should answer a ``float32`` tensor of one row per window
        and one channel per track, equal to the dense arrays the
        tracks' rows imply.
    """

    # Arrange
    rows = [("chr2", 215, 245), ("chr1", 105, 135), ("chr1", 805, 835)]
    windows = window_frame(rows, ("sig_start", "sig_end"))

    # Act
    signal = loci.extract_signal(windows, corpus.tracks)

    # Assert
    assert (signal.dtype, signal.shape) == (torch.float32, (3, 2, 30))
    assert torch.equal(signal, dense_signal([TRACK_ROWS, CONTROL_ROWS], rows))


def test_extract_signal_should_order_channels_as_the_tracks_are_given(corpus, engine):
    """Test that track order is the channel axis.

    Given:
        The module's two tracks in either order.
    When:
        The signal is extracted over one window.
    Then:
        It should answer the same two channels, swapped.
    """

    # Arrange
    windows = window_frame([("chr1", 105, 135)], ("sig_start", "sig_end"))

    # Act
    forward = loci.extract_signal(windows, corpus.tracks)
    backward = loci.extract_signal(windows, corpus.tracks[::-1])

    # Assert
    assert not torch.equal(forward[:, 0], forward[:, 1])
    assert torch.equal(forward, backward.flip(1))


def test_extract_signal_should_total_a_position_recorded_twice(engine):
    """Test the reading of a track that records one position twice.

    Given:
        A track with two values at one position and one at another.
    When:
        The signal is extracted over a window holding both.
    Then:
        It should answer the sum at the repeated position.
    """

    # Arrange
    windows = window_frame([("chr1", 8, 14)], ("sig_start", "sig_end"))
    repeated = builders.track_rows_frame(
        [("chr1", 10, 1.0), ("chr1", 10, 4.0), ("chr1", 12, 3.0)]
    )

    # Act
    signal = loci.extract_signal(windows, [repeated])

    # Assert
    assert signal[0, 0].tolist() == [0.0, 0.0, 5.0, 0.0, 3.0, 0.0]


def test_extract_signal_should_read_a_coverage_track_as_its_scaled_coverage(
    tmp_path, engine
):
    """Test the dense read of a track of coverage deltas.

    Given:
        Two overlapping spans written as the boundary events a
        ``"span"`` pileup writes, at a scale of a half, beside a track
        of counts.
    When:
        The signal is extracted over a window across both spans.
    Then:
        The coverage channel should hold the spans' coverage times the
        half at every base, and the counts channel its values.
    """

    # Arrange
    windows = window_frame([("chr1", 8, 16)], ("sig_start", "sig_end"))
    deltas = builders.write_track(
        str(tmp_path / "d.track.parquet"),
        [("chr1", 9, 1.0), ("chr1", 11, 1.0), ("chr1", 13, -1.0), ("chr1", 20, -1.0)],
        signal="coverage_deltas",
        scale=0.5,
    )
    counts = builders.track_rows_frame([("chr1", 10, 3.0)])

    # Act
    signal = loci.extract_signal(windows, [deltas, counts])

    # Assert
    assert signal[0, 0].tolist() == [0.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5]
    assert signal[0, 1].tolist() == [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_extract_signal_should_read_a_row_on_the_start_and_omit_one_on_the_end(engine):
    """Test that a window is half-open.

    Given:
        A track with a position on a window's start, one inside it and
        one on its end.
    When:
        The signal is extracted.
    Then:
        It should hold the first two and not the third.
    """

    # Arrange
    windows = window_frame([("chr1", 8, 16)], ("sig_start", "sig_end"))
    edges = builders.track_rows_frame(
        [("chr1", 8, 11.0), ("chr1", 12, 3.0), ("chr1", 16, 13.0)]
    )

    # Act
    signal = loci.extract_signal(windows, [edges])

    # Assert
    assert signal[0, 0].tolist() == [11.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]


def test_extract_signal_should_answer_zeros_for_a_contig_a_track_lacks(engine):
    """Test the reading of a window on a contig a track never names.

    Given:
        A track holding positions only on ``chr1`` and windows on
        ``chr1`` and ``chr2``.
    When:
        The signal is extracted.
    Then:
        It should answer the ``chr1`` window's positions and zeros for
        the ``chr2`` window.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 8), ("chr2", 0, 8)], ("sig_start", "sig_end"))
    one = builders.track_rows_frame([("chr1", 3, 2.0)])

    # Act
    signal = loci.extract_signal(windows, [one])

    # Assert
    assert signal[0, 0].tolist() == [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    assert not signal[1].any()


def test_extract_signal_should_read_between_the_columns_it_is_given(corpus, engine):
    """Test that the bound columns decide the window read.

    Given:
        A window frame carrying an output window and a wider input
        window around it.
    When:
        The signal is extracted over the input window's columns.
    Then:
        It should answer the input window's width, equal to the dense
        array over that span.
    """

    # Arrange
    windows = loci.load_windows(
        pandas.DataFrame([("chr1", 100, 140)]), corpus.fasta, 60, 30
    )

    # Act
    signal = loci.extract_signal(
        windows, corpus.tracks[:1], bound_columns=("in_start", "in_end")
    )

    # Assert
    assert torch.equal(signal, dense_signal([TRACK_ROWS], [("chr1", 90, 150)]))


@pytest.mark.parametrize("form", ["frame", "lazy"])
def test_extract_signal_should_read_an_in_memory_track_as_it_reads_one_on_disk(
    corpus, engine, form
):
    """Test the in-memory forms of a track.

    Given:
        The rows of the module's first track as a polars frame, eager
        or lazy.
    When:
        The signal is extracted over two windows.
    Then:
        It should equal the signal read from the track on disk.
    """

    # Arrange
    windows = window_frame(
        [("chr2", 215, 245), ("chr1", 105, 135)], ("sig_start", "sig_end")
    )
    held = (
        builders.track_rows_frame(TRACK_ROWS)
        if form == "frame"
        else builders.track_rows_frame(TRACK_ROWS).lazy()
    )

    # Act
    signal = loci.extract_signal(windows, [held])

    # Assert
    assert torch.equal(signal, loci.extract_signal(windows, corpus.tracks[:1]))


def test_extract_signal_should_answer_no_rows_for_no_windows(corpus, engine):
    """Test the shape answered for an empty window frame.

    Given:
        A window frame with no rows and two tracks.
    When:
        The signal is extracted.
    Then:
        It should answer a ``float32`` tensor of no rows, two channels
        and no width.
    """

    # Arrange
    windows = window_frame([], ("sig_start", "sig_end"))

    # Act
    signal = loci.extract_signal(windows, corpus.tracks)

    # Assert
    assert (signal.dtype, signal.shape) == (torch.float32, (0, 2, 0))


@pytest.mark.parametrize("form", ["path", "frame"])
def test_extract_signal_should_raise_type_error_when_given_a_single_track(corpus, form):
    """Test the refusal for one track where a sequence of them is taken.

    Given:
        A single track, as a path or as a frame, rather than a list.
    When:
        The signal is extracted.
    Then:
        It should raise `TypeError` naming the function and saying to
        pass a one-element list.
    """

    # Arrange
    windows = window_frame([("chr1", 105, 135)], ("sig_start", "sig_end"))
    single = (
        corpus.tracks[0] if form == "path" else builders.track_rows_frame(TRACK_ROWS)
    )

    # Act & assert
    with pytest.raises(
        TypeError, match="extract_signal reads one track per channel.*one-element list"
    ):
        loci.extract_signal(windows, single)


def test_extract_signal_should_raise_type_error_when_the_tracks_are_a_pandas_frame():
    """Test the refusal for a frame of another library given as the tracks.

    Given:
        A pandas frame carrying a track's three columns, in place of a
        list of tracks.
    When:
        The signal is extracted.
    Then:
        It should raise `TypeError` naming the function and the frame,
        rather than read its column names as tracks.
    """

    # Arrange
    windows = window_frame([("chr1", 105, 135)], ("sig_start", "sig_end"))
    frame = pandas.DataFrame({"chrom": ["chr1"], "pos": [110], "value": [1.0]})

    # Act & assert
    with pytest.raises(
        TypeError, match="extract_signal reads one track per channel.*pandas DataFrame"
    ):
        loci.extract_signal(windows, frame)


def test_extract_signal_should_raise_type_error_when_windows_is_no_polars_frame(corpus):
    """Test the refusal for windows held in another library's frame.

    Given:
        Windows as a pandas frame.
    When:
        The signal is extracted.
    Then:
        It should raise `TypeError` naming the function and the type
        it was given.
    """

    # Arrange
    windows = pandas.DataFrame({"chrom": ["chr1"], "sig_start": [0], "sig_end": [8]})

    # Act & assert
    with pytest.raises(TypeError, match="extract_signal takes a polars DataFrame"):
        loci.extract_signal(windows, corpus.tracks)


@pytest.mark.parametrize(
    "rows",
    [[("chr1", 0, 8), ("chr1", 20, 30)], [("chr1", 8, 0)]],
    ids=["unequal", "reversed"],
)
def test_extract_signal_should_raise_when_the_windows_are_not_one_width(corpus, rows):
    """Test the refusal for windows no single tensor can hold.

    Given:
        Windows of two different widths, or one whose end is before
        its start.
    When:
        The signal is extracted.
    Then:
        It should raise `ValueError` naming the function and the two
        bound columns.
    """

    # Arrange
    windows = window_frame(rows, ("sig_start", "sig_end"))

    # Act & assert
    with pytest.raises(
        ValueError,
        match="extract_signal answers one tensor.*between `sig_start` and `sig_end`",
    ):
        loci.extract_signal(windows, corpus.tracks)


def test_extract_signal_should_raise_when_a_track_does_not_verify(engine, tmp_path):
    """Test the refusal for a track directory with no completeness record.

    Given:
        A directory under a track's name holding no manifest beside it.
    When:
        The signal is extracted from it.
    Then:
        It should raise `ValueError` rather than read it as empty.
    """

    # Arrange
    unverifiable = tmp_path / "t.track.parquet"
    unverifiable.mkdir()
    windows = window_frame([("chr1", 0, 8)], ("sig_start", "sig_end"))

    # Act & assert
    with pytest.raises(ValueError, match="t.track.parquet"):
        loci.extract_signal(windows, [str(unverifiable)])


@pytest.mark.parametrize("fields, message", UNREADABLE_RECORDS, ids=UNREADABLE_IDS)
def test_extract_signal_should_raise_when_a_tracks_signal_cannot_be_read(
    engine, tmp_path, fields, message
):
    """Test the refusal for a complete track no reader here can read.

    Given:
        A complete track whose record names a signal no reader knows,
        or a known signal at the other's version.
    When:
        The signal is extracted from it.
    Then:
        It should raise `ValueError` naming what is wrong with the
        record, rather than place its rows as some signal they may not
        be.
    """

    # Arrange
    root = builders.write_track(str(tmp_path / "t.track.parquet"), [("chr1", 4, 1.0)])
    builders.relabel_track(root, **fields)
    windows = window_frame([("chr1", 0, 8)], ("sig_start", "sig_end"))

    # Act & assert
    with pytest.raises(ValueError, match=message):
        loci.extract_signal(windows, [root])


def test_extract_signal_should_raise_import_error_when_the_extra_is_absent(
    corpus, no_engine
):
    """Test what reading signal needs that the base install lacks.

    Given:
        Windows and tracks, in a process where the engine will not
        import.
    When:
        The signal is extracted.
    Then:
        It should raise `ImportError` naming the ``pileup`` extra.
    """

    # Arrange
    windows = window_frame([("chr1", 105, 135)], ("sig_start", "sig_end"))

    # Act & assert
    with pytest.raises(ImportError, match="`pileup` extra"):
        loci.extract_signal(windows, corpus.tracks)


@settings(max_examples=25, deadline=None)
@given(
    channels=st.lists(position_rows, min_size=1, max_size=3),
    starts=st.lists(
        st.tuples(st.sampled_from(["a", "b", "c"]), st.integers(-10, 110)), max_size=6
    ),
    width=st.integers(min_value=1, max_value=40),
)
def test_extract_signal_should_equal_the_dense_array_any_rows_imply(
    engine, channels, starts, width
):
    """Test the extraction against a dense oracle, over arbitrary inputs.

    Given:
        Any one to three tracks of positions, repeats allowed, and any
        windows of one width, overlapping or on a contig no track
        names.
    When:
        The signal is extracted.
    Then:
        It should equal the tensor built by testing every position
        against every window.
    """

    # Arrange
    rows = [(chrom, start, start + width) for chrom, start in starts]
    windows = window_frame(rows, ("sig_start", "sig_end"))

    # Act
    signal = loci.extract_signal(
        windows, [builders.track_rows_frame(rows_of) for rows_of in channels]
    )

    # Assert
    expected = dense_signal(channels, rows)
    assert signal.shape == (len(rows), len(channels), width if rows else 0)
    assert torch.equal(signal.reshape(expected.shape), expected)


def test_extract_sequences_should_one_hot_the_residues_at_each_window(corpus):
    """Test the encoding against the reference's own residues.

    Given:
        Two windows of 40 bases on different contigs.
    When:
        Their sequences are extracted.
    Then:
        It should answer an ``int8`` tensor of one row per window and
        four residue rows, equal to the one-hot of the file's residues
        at those coordinates.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)
    windows = window_frame(
        [("chr2", 300, 340), ("chr1", 400, 440)], ("in_start", "in_end")
    )

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert (encoded.dtype, encoded.shape) == (torch.int8, (2, 4, 40))
    assert numpy.array_equal(encoded[0].numpy(), one_hot(residues["chr2"][300:340]))
    assert numpy.array_equal(encoded[1].numpy(), one_hot(residues["chr1"][400:440]))


def test_extract_sequences_should_upcase_a_soft_masked_residue(corpus):
    """Test the reading of lower-case sequence.

    Given:
        A window inside the reference's soft-masked run.
    When:
        Its sequence is extracted.
    Then:
        It should encode every column as the base it is, none left
        empty.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)["chr1"][800:840]
    windows = window_frame([("chr1", 800, 840)], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert residues.islower()
    assert numpy.array_equal(encoded[0].numpy(), one_hot(residues))
    assert (encoded[0].sum(dim=0) == 1).all()


def test_extract_sequences_should_zero_a_column_for_an_ignored_residue(corpus):
    """Test the reading of an ambiguous base.

    Given:
        A window over the reference's ``N`` run at ``[110, 120)`` of
        ``chr1``.
    When:
        Its sequence is extracted.
    Then:
        It should leave exactly the ten ``N`` columns all-zero.
    """

    # Arrange
    windows = window_frame([("chr1", 100, 130)], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert (encoded[0].sum(dim=0) == 0).nonzero().flatten().tolist() == list(
        range(10, 20)
    )


def test_extract_sequences_should_order_the_rows_by_the_alphabet_given(corpus):
    """Test that the alphabet decides the residue axis.

    Given:
        One window read under the alphabet ``ACGT`` and under ``TGCA``.
    When:
        Its sequence is extracted under each.
    Then:
        It should answer the same rows in reverse order.
    """

    # Arrange
    windows = window_frame([("chr1", 400, 440)], ("in_start", "in_end"))

    # Act
    forward = loci.extract_sequences(windows, corpus.fasta)
    backward = loci.extract_sequences(windows, corpus.fasta, alphabet="TGCA")

    # Assert
    assert torch.equal(forward, backward.flip(1))


def test_extract_sequences_should_raise_when_a_residue_is_neither_encoded_nor_ignored(
    corpus,
):
    """Test the refusal for a residue the caller made no provision for.

    Given:
        A window over the reference's ``N`` run and nothing ignored.
    When:
        Its sequence is extracted.
    Then:
        It should raise `ValueError` saying a residue is in neither the
        alphabet nor the ignored residues, rather than encode it as
        anything.
    """

    # Arrange
    windows = window_frame([("chr1", 100, 130)], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(ValueError, match="not in `alphabet` or in `ignore`"):
        loci.extract_sequences(windows, corpus.fasta, ignore=())


def test_extract_sequences_should_keep_row_order_when_contigs_alternate(corpus):
    """Test that windows on alternating contigs keep their row order.

    Given:
        Four windows alternating between two contigs.
    When:
        Their sequences are extracted.
    Then:
        It should answer row ``i`` for window ``i``.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)
    rows = [("chr2", 10, 30), ("chr1", 10, 30), ("chr2", 50, 70), ("chr1", 50, 70)]
    windows = window_frame(rows, ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    for index, (chrom, start, end) in enumerate(rows):
        assert numpy.array_equal(
            encoded[index].numpy(), one_hot(residues[chrom][start:end])
        )


@pytest.mark.parametrize("shape", [pathlib.Path, os.fsencode])
def test_extract_sequences_should_accept_every_path_shape(corpus, shape):
    """Test that a path object and a bytes path are read as a `str` is.

    Given:
        The reference named by a `pathlib.Path` or by `bytes`.
    When:
        A window's sequence is extracted.
    Then:
        It should answer the one-hot of the file's residues at the
        window.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)["chr2"][300:340]
    windows = window_frame([("chr2", 300, 340)], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, shape(corpus.fasta))

    # Assert
    assert numpy.array_equal(encoded[0].numpy(), one_hot(residues))


def test_extract_sequences_should_read_a_contig_whose_name_holds_a_quote(tmp_path):
    """Test that a contig name is looked up exactly as the reference writes it.

    Given:
        A reference with a contig named ``it's`` and a window on it.
    When:
        The window's sequence is extracted.
    Then:
        It should answer the one-hot of that contig's residues at the
        window.
    """

    # Arrange
    path = builders.write_fasta(str(tmp_path / "genome.fa"), {"it's": 120, "chr2": 80})
    residues = parse_fasta(path)["it's"][20:60]
    windows = window_frame([("it's", 20, 60)], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, path)

    # Assert
    assert numpy.array_equal(encoded[0].numpy(), one_hot(residues))


def test_extract_sequences_should_read_a_replaced_reference_afresh(tmp_path):
    """Test that nothing read from a reference outlives the file it came from.

    Given:
        A reference a window has been extracted from, then replaced at
        the same path by one of other residues.
    When:
        The window is extracted again.
    Then:
        It should answer the one-hot of the replacement's residues.
    """

    # Arrange
    path = builders.write_fasta(str(tmp_path / "genome.fa"), {"chr1": 200}, seed=1)
    windows = window_frame([("chr1", 20, 180)], ("in_start", "in_end"))
    first = loci.extract_sequences(windows, path)
    os.remove(path + ".fai")
    builders.write_fasta(path, {"chr1": 200}, seed=2)

    # Act
    second = loci.extract_sequences(windows, path)

    # Assert
    assert not torch.equal(first, second)
    assert numpy.array_equal(
        second[0].numpy(), one_hot(parse_fasta(path)["chr1"][20:180])
    )


@pytest.mark.parametrize(
    ("row", "span"),
    [(("chr1", -5, 35), r"\[-5, 35\)"), (("chr1", 970, 1010), r"\[970, 1010\)")],
    ids=["before-start", "past-end"],
)
def test_extract_sequences_should_raise_when_a_window_reaches_off_its_contig(
    corpus, row, span
):
    """Test the refusal for a window the reference cannot answer whole.

    Given:
        A window starting before its contig or ending past it.
    When:
        Its sequence is extracted.
    Then:
        It should raise `ValueError` naming the window's span, its
        contig and the contig's length.
    """

    # Arrange
    windows = window_frame([("chr1", 100, 140), row], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"window 1 is {} on `chr1`, which is 1000 bases long".format(span),
    ):
        loci.extract_sequences(windows, corpus.fasta)


def test_extract_sequences_should_raise_when_a_window_is_on_an_unknown_contig(corpus):
    """Test the refusal for a window the reference cannot place.

    Given:
        A window on ``chrZ``, which the reference does not name.
    When:
        Its sequence is extracted.
    Then:
        It should raise `ValueError` naming the reference and the
        contig.
    """

    # Arrange
    windows = window_frame([("chrZ", 100, 140)], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(ValueError, match=r"genome\.fa does not name the contig `chrZ`"):
        loci.extract_sequences(windows, corpus.fasta)


def test_extract_sequences_should_count_the_unknown_contigs_it_does_not_list(corpus):
    """Test how the refusal names more contigs than it spells out.

    Given:
        Windows on five contigs the reference does not name.
    When:
        Their sequences are extracted.
    Then:
        It should raise `ValueError` spelling out the first three,
        counting the two it leaves out, and naming the contigs the
        reference holds.
    """

    # Arrange
    windows = window_frame(
        [(name, 100, 140) for name in "12345"], ("in_start", "in_end")
    )

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"does not name the contigs `1`, `2`, `3` and 2 more\. The "
        r"reference names `chr1`, `chr2`\.",
    ):
        loci.extract_sequences(windows, corpus.fasta)


def test_extract_sequences_should_raise_os_error_when_the_reference_is_missing(
    tmp_path,
):
    """Test the refusal for a reference that is not there.

    Given:
        A window and a path at which there is no reference.
    When:
        Its sequence is extracted.
    Then:
        It should raise `OSError` naming the path.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 8)], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(OSError, match=r"missing\.fa"):
        loci.extract_sequences(windows, str(tmp_path / "missing.fa"))


def test_extract_sequences_should_raise_when_the_windows_are_not_one_width(corpus):
    """Test the refusal for windows no single tensor can hold.

    Given:
        Windows of two different widths.
    When:
        Their sequences are extracted.
    Then:
        It should raise `ValueError` naming the function and the two
        bound columns.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 40), ("chr1", 100, 160)], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(
        ValueError,
        match="extract_sequences answers one tensor.*between `in_start` and `in_end`",
    ):
        loci.extract_sequences(windows, corpus.fasta)


def test_extract_sequences_should_raise_type_error_when_windows_is_no_polars_frame(
    corpus,
):
    """Test the refusal for windows held in another library's frame.

    Given:
        Windows as a pandas frame.
    When:
        Their sequences are extracted.
    Then:
        It should raise `TypeError` naming the function and the type
        it was given.
    """

    # Arrange
    windows = pandas.DataFrame({"chrom": ["chr1"], "in_start": [0], "in_end": [8]})

    # Act & assert
    with pytest.raises(TypeError, match="extract_sequences takes a polars DataFrame"):
        loci.extract_sequences(windows, corpus.fasta)


def test_extract_sequences_should_raise_type_error_when_sequences_is_no_path():
    """Test the refusal for a reference that is not a file.

    Given:
        A window and an in-memory mapping in place of the reference.
    When:
        Its sequence is extracted.
    Then:
        It should raise `TypeError` naming the type it was given.
    """

    # Arrange
    windows = window_frame([("chr1", 0, 4)], ("in_start", "in_end"))

    # Act & assert
    with pytest.raises(TypeError, match="got dict"):
        loci.extract_sequences(windows, {"chr1": "ACGT"})


def test_extract_sequences_should_answer_no_rows_for_no_windows(corpus):
    """Test the shape answered for an empty window frame.

    Given:
        A window frame with no rows.
    When:
        Its sequences are extracted.
    Then:
        It should answer an ``int8`` tensor of no rows, four residue
        rows and no width.
    """

    # Arrange
    windows = window_frame([], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert (encoded.dtype, encoded.shape) == (torch.int8, (0, 4, 0))


def test_extract_sequences_should_answer_no_windows_without_opening_the_reference(
    tmp_path,
):
    """Test that an empty window frame costs no read.

    Given:
        A window frame with no rows and a path at which there is no
        reference.
    When:
        Its sequences are extracted.
    Then:
        It should answer a tensor with no rows, raising nothing.
    """

    # Arrange
    windows = window_frame([], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, str(tmp_path / "missing.fa"))

    # Assert
    assert encoded.shape == (0, 4, 0)


def test_extract_sequences_should_run_without_the_pileup_extra(corpus, no_engine):
    """Test that reading sequence needs nothing the base install lacks.

    Given:
        A window, in a process where the engine will not import.
    When:
        Its sequence is extracted.
    Then:
        It should answer the one-hot of the file's residues at the
        window.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)["chr1"][400:440]
    windows = window_frame([("chr1", 400, 440)], ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert numpy.array_equal(encoded[0].numpy(), one_hot(residues))


@pytest.mark.packaging
def test_extract_sequences_should_raise_os_error_rather_than_scan_an_unindexable_reference(
    unindexable_fasta,
):
    """Test that sequence is never read by a scan of the reference.

    Given:
        A reference with no index in a directory that cannot be written
        to, in a fresh interpreter in which neither module of the
        ``pileup`` extra will import.
    When:
        A window's sequence is extracted.
    Then:
        It should raise `OSError` naming the reference and the command
        that indexes it, the engine never being asked for.
    """

    # Act
    name, message = raised_without_engine(
        "loci.extract_sequences(polars.DataFrame({'chrom': ['chr1'], "
        "'in_start': [0], 'in_end': [8]}), sys.argv[1])",
        unindexable_fasta,
    )

    # Assert
    assert name == "OSError"
    assert unindexable_fasta in message
    assert "samtools faidx" in message


@settings(max_examples=40, deadline=None)
@given(
    placements=st.lists(
        st.tuples(st.sampled_from(sorted(TWO_CONTIGS)), st.integers(0, 10**6)),
        max_size=8,
    ),
    width=st.integers(min_value=0, max_value=80),
)
def test_extract_sequences_should_equal_the_one_hot_any_windows_imply(
    corpus, placements, width
):
    """Test the extraction against the file's residues, over arbitrary windows.

    Given:
        Any windows of one width lying on the module's two contigs, in
        any order, soft-masked and ``N`` runs and each contig's last
        base among what they may cover.
    When:
        Their sequences are extracted.
    Then:
        Row ``i`` should equal the one-hot of the file's residues at
        window ``i``, for every window.
    """

    # Arrange
    residues = parse_fasta(corpus.fasta)
    rows = []
    for chrom, offset in placements:
        start = offset % (TWO_CONTIGS[chrom] - width + 1)
        rows.append((chrom, start, start + width))
    windows = window_frame(rows, ("in_start", "in_end"))

    # Act
    encoded = loci.extract_sequences(windows, corpus.fasta)

    # Assert
    assert encoded.shape == (len(rows), 4, width if rows else 0)
    for index, (chrom, start, end) in enumerate(rows):
        assert numpy.array_equal(
            encoded[index].numpy(), one_hot(residues[chrom][start:end])
        )


@pytest.mark.parametrize("max_jitter", [0, 5])
@pytest.mark.parametrize("chroms", [None, ["chr1"]])
def test_extract_loci_should_match_tangermeme_over_the_same_inputs(
    corpus, engine, max_jitter, chroms
):
    """Test the whole extraction against tangermeme's.

    Given:
        Peaks with windows off either edge of a contig, one ending
        exactly at a contig's length and one inside an excluded
        region; an exclusion list; and one track, given to tangermeme
        as the dense arrays its rows imply.
    When:
        The loci are extracted by this package and by tangermeme, with
        and without jitter and a contig filter.
    Then:
        The sequences, the signal and the kept mask should be
        identical in value, dtype and shape.
    """

    # Arrange
    peaks = pandas.DataFrame(AGREED_PEAKS)
    shared = dict(
        chroms=chroms, max_jitter=max_jitter, exclusion_lists=[corpus.exclusion]
    )
    expected_sequences, expected_signal, expected_kept = tangermeme_extract_loci(
        peaks,
        corpus.fasta,
        signals=[dense(TRACK_ROWS)],
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        return_mask=True,
        **shared,
    )

    # Act
    extracted = loci.extract_loci(
        peaks, corpus.fasta, IN_WINDOW, OUT_WINDOW, signals=corpus.tracks[:1], **shared
    )

    # Assert
    assert extracted.kept.dtype == expected_kept.dtype == torch.bool
    assert extracted.sequences.dtype == expected_sequences.dtype == torch.int8
    assert extracted.signals.dtype == expected_signal.dtype == torch.float32
    assert torch.equal(extracted.kept, expected_kept)
    assert torch.equal(extracted.sequences, expected_sequences)
    assert torch.equal(extracted.signals, expected_signal)
    assert extracted.signals.sum() > 0


def test_extract_loci_should_keep_the_loci_its_three_tests_admit(corpus, engine):
    """Test the kept mask against a hand-derived expectation.

    Given:
        The module's nine peaks, its exclusion list and a jitter of 5.
    When:
        The loci are extracted.
    Then:
        It should drop the two windows off their contig, the one
        inside the excluded region and the one reaching it through a
        bin edge, keep the one ending exactly at the contig's length,
        and answer one row per kept locus.
    """

    # Act
    extracted = loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=corpus.tracks[:1],
        max_jitter=5,
        exclusion_lists=[corpus.exclusion],
    )

    # Assert
    assert extracted.kept.tolist() == [
        True,  # chr2 200-260
        True,  # chr1 100-140
        False,  # chr1 20-40 starts before the contig
        False,  # chr1 960-1000 ends past the contig
        True,  # chr1 955-975 ends exactly at its length
        False,  # chr1 690-720 reaches the region through a bin edge
        True,  # chr1 800-840
        True,  # chr1 930-940
        False,  # chr1 540-560 is inside the region
    ]
    assert extracted.sequences.shape == (5, 4, 70)
    assert extracted.signals.shape == (5, 1, 40)


def test_extract_loci_should_read_and_count_a_coverage_track(corpus, engine, tmp_path):
    """Test a track of coverage deltas as the signal an extraction counts on.

    Given:
        The module's peaks, and one span covering the output window of
        the chr2 peak alone, written as boundary events at a scale of
        two.
    When:
        The loci are extracted with a count floor of one.
    Then:
        It should keep the chr2 peak alone, and its signal should be
        two at every base of its output window.
    """

    # Arrange
    deltas = builders.write_track(
        str(tmp_path / "d.track.parquet"),
        [("chr2", 150, 1.0), ("chr2", 300, -1.0)],
        signal="coverage_deltas",
        scale=2.0,
    )

    # Act
    extracted = loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=[deltas],
        min_counts=1,
    )

    # Assert
    assert extracted.kept.tolist() == [True] + [False] * 8
    assert extracted.signals.tolist() == [[[2.0] * OUT_WINDOW]]


def test_extract_loci_should_answer_sequences_alone_when_no_track_is_given(corpus):
    """Test the extraction that reads no signal.

    Given:
        The module's peaks and no signal or control track.
    When:
        The loci are extracted by this package with no output window,
        and by tangermeme.
    Then:
        It should answer tangermeme's sequences, with no signal and no
        controls.
    """

    # Arrange
    expected = tangermeme_extract_loci(
        corpus.peaks, corpus.fasta, in_window=IN_WINDOW, out_window=OUT_WINDOW
    )

    # Act
    extracted = loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW)

    # Assert
    assert torch.equal(extracted.sequences, expected)
    assert extracted.signals is None
    assert extracted.controls is None
    assert int(extracted.kept.sum()) == len(expected)


def test_extract_loci_should_not_let_an_unread_output_window_decide_the_bounds(
    corpus,
):
    """Test that the output window is without effect when no signal is read.

    Given:
        A locus whose input window fits its contig and whose output
        window of 200 would not.
    When:
        The locus is extracted with that output window and no signal
        track.
    Then:
        It should keep the locus.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 30, 50)])

    # Act
    extracted = loci.extract_loci(locus, corpus.fasta, IN_WINDOW, 200)

    # Assert
    assert extracted.kept.tolist() == [True]


def test_extract_loci_should_drop_the_loci_outside_its_count_thresholds(corpus, engine):
    """Test the count filter against tangermeme's.

    Given:
        The module's peaks and thresholds of 19 and 24, each the exact
        count of some locus, with three loci counting outside them.
    When:
        The loci are extracted by this package and by tangermeme.
    Then:
        The kept mask, the sequences and the signal should be
        identical, with a locus on each threshold kept and five loci
        kept in all.
    """

    # Arrange
    peaks = pandas.DataFrame(PEAKS)
    expected_sequences, expected_signal, expected_kept = tangermeme_extract_loci(
        peaks,
        corpus.fasta,
        signals=[dense(TRACK_ROWS)],
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        min_counts=19,
        max_counts=24,
        return_mask=True,
    )

    # Act
    extracted = loci.extract_loci(
        peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=corpus.tracks[:1],
        min_counts=19,
        max_counts=24,
    )

    # Assert
    counts = extracted.signals.sum(dim=(1, 2)).tolist()
    assert torch.equal(extracted.kept, expected_kept)
    assert torch.equal(extracted.sequences, expected_sequences)
    assert torch.equal(extracted.signals, expected_signal)
    assert 19.0 in counts and 24.0 in counts
    assert int(extracted.kept.sum()) == 5


def test_extract_loci_should_count_on_the_track_target_idx_names(corpus, engine):
    """Test which track the count is taken on.

    Given:
        Two signal tracks and one locus, with a floor above the
        locus's count on the first track and below its count on the
        second.
    When:
        The locus is extracted counting on the first track, on the
        second, and on the last by a negative index.
    Then:
        It should keep the locus on one track and drop it on the
        other, the negative index agreeing with the second.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 955, 975)])
    arguments = dict(signals=corpus.tracks, min_counts=11)

    # Act
    on_first = loci.extract_loci(locus, corpus.fasta, IN_WINDOW, OUT_WINDOW, **arguments)
    on_second = loci.extract_loci(
        locus, corpus.fasta, IN_WINDOW, OUT_WINDOW, target_idx=1, **arguments
    )
    on_last = loci.extract_loci(
        locus, corpus.fasta, IN_WINDOW, OUT_WINDOW, target_idx=-1, **arguments
    )

    # Assert
    assert on_first.kept.tolist() == [False]
    assert on_second.kept.tolist() == [True]
    assert on_last.kept.tolist() == on_second.kept.tolist()


def test_extract_loci_should_read_controls_over_the_input_window(corpus, engine):
    """Test the window a control is read over.

    Given:
        One track given as both signal and control, an input window of
        100 and an output window of 20.
    When:
        The loci are extracted.
    Then:
        It should answer the controls at the input width and the
        signal at the output width, equal where the two windows
        overlap.
    """

    # Arrange
    in_window, out_window = 100, 20
    trim = (in_window - out_window) // 2

    # Act
    extracted = loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        in_window,
        out_window,
        signals=corpus.tracks[:1],
        controls=corpus.tracks[:1],
    )

    # Assert
    rows = extracted.sequences.shape[0]
    assert extracted.signals.shape == (rows, 1, out_window)
    assert extracted.controls.shape == (rows, 1, in_window)
    assert extracted.controls.dtype == torch.float32
    assert torch.equal(
        extracted.controls[:, :, trim : trim + out_window], extracted.signals
    )


def test_extract_loci_should_size_the_mask_to_the_loci_the_filter_leaves(corpus):
    """Test what the kept mask is indexed by.

    Given:
        The module's nine peaks, eight of them on ``chr1``.
    When:
        The loci are extracted keeping ``chr1``.
    Then:
        It should answer a mask of eight entries, whose true entries
        number the rows of the sequences.
    """

    # Act
    extracted = loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW, chroms=["chr1"])

    # Assert
    assert extracted.kept.shape == (8,)
    assert int(extracted.kept.sum()) == extracted.sequences.shape[0]


def test_extract_loci_should_answer_no_rows_for_an_empty_locus_set(corpus):
    """Test the extraction of a locus set with nothing in it.

    Given:
        A locus frame with no rows, a contig filter and a signal
        track.
    When:
        The loci are extracted.
    Then:
        It should answer tensors with no rows and an empty mask,
        raising nothing.
    """

    # Arrange
    empty = pandas.DataFrame({"chrom": [], "start": [], "end": []})

    # Act
    extracted = loci.extract_loci(
        empty,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=corpus.tracks[:1],
        chroms=["chr1"],
    )

    # Assert
    assert len(extracted.sequences) == len(extracted.signals) == len(extracted.kept) == 0


@pytest.mark.parametrize(
    "emptied_by",
    [
        dict(loci=pandas.DataFrame({"chrom": [], "start": [], "end": []})),
        dict(loci=pandas.DataFrame([("chr1", 0, 10), ("chr1", 990, 1000)])),
        dict(exclusion_lists=pandas.DataFrame([("chr1", 0, 1000), ("chr2", 0, 500)])),
        dict(min_counts=10**9),
    ],
    ids=["no-loci", "off-contig", "excluded", "count-filter"],
)
def test_extract_loci_should_answer_an_empty_result_at_the_nominal_widths(
    corpus, engine, emptied_by
):
    """Test the shape of a result with no rows, whatever left it empty.

    Given:
        A signal and a control track, an input window of 60, an output
        window of 30 and a jitter of 5, over loci of which none is
        read: there being none, or all of them off their contig,
        excluded, or under the count floor.
    When:
        The loci are extracted.
    Then:
        It should answer no rows at the widths a full result has: 70
        for the sequences and the controls and 40 for the signals,
        each at its own dtype.
    """

    # Arrange
    arguments = dict(
        loci=corpus.peaks,
        signals=corpus.tracks[:1],
        controls=corpus.tracks,
        max_jitter=5,
    )
    arguments.update(emptied_by)

    # Act
    extracted = loci.extract_loci(
        arguments.pop("loci"), corpus.fasta, IN_WINDOW, OUT_WINDOW, **arguments
    )

    # Assert
    assert (extracted.sequences.dtype, extracted.sequences.shape) == (
        torch.int8,
        (0, 4, 70),
    )
    assert (extracted.signals.dtype, extracted.signals.shape) == (
        torch.float32,
        (0, 1, 40),
    )
    assert (extracted.controls.dtype, extracted.controls.shape) == (
        torch.float32,
        (0, 2, 70),
    )
    assert not extracted.kept.any()


def test_extract_loci_should_read_sequences_alone_without_the_pileup_extra(
    corpus, no_engine
):
    """Test that an extraction of sequences needs no engine.

    Given:
        The module's nine peaks, one of them ending past its contig,
        and no track or exclusion list, in a process where the engine
        will not import.
    When:
        The loci are extracted.
    Then:
        It should answer eight rows of sequence, one per locus in
        bounds.
    """

    # Act
    extracted = loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW)

    # Assert
    assert extracted.sequences.shape == (8, 4, IN_WINDOW)
    assert int(extracted.kept.sum()) == 8


@pytest.mark.packaging
def test_extract_loci_should_raise_os_error_rather_than_scan_an_unindexable_reference(
    unindexable_fasta,
):
    """Test that an extraction of sequences never reaches for the engine.

    Given:
        A reference with no index in a directory that cannot be written
        to, in a fresh interpreter in which neither module of the
        ``pileup`` extra will import.
    When:
        A locus is extracted with no track or exclusion list.
    Then:
        It should raise `OSError` naming the reference and the command
        that indexes it, the engine never being asked for.
    """

    # Act
    name, message = raised_without_engine(
        "loci.extract_loci(pandas.DataFrame([('chr1', 100, 140)]), sys.argv[1], 60)",
        unindexable_fasta,
    )

    # Assert
    assert name == "OSError"
    assert unindexable_fasta in message
    assert "samtools faidx" in message


def test_extract_loci_should_raise_os_error_when_the_reference_is_missing(
    tmp_path, recwarn
):
    """Test the refusal for a reference that is not there.

    Given:
        A locus and a path at which there is no reference.
    When:
        The locus is extracted.
    Then:
        It should raise `OSError` naming the path, without first
        warning of a scan for the reference's lengths.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 100, 140)])

    # Act & assert
    with pytest.raises(OSError, match=r"missing\.fa"):
        loci.extract_loci(locus, str(tmp_path / "missing.fa"), 60)
    assert not [caught for caught in recwarn.list if "scanning" in str(caught.message)]


def test_extract_loci_should_raise_when_signals_come_without_an_output_window(corpus):
    """Test the refusal for signal with no window to read it over.

    Given:
        A signal track and no output window.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming both arguments.
    """

    # Act & assert
    with pytest.raises(ValueError, match="`signals` needs `out_window`"):
        loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW, signals=corpus.tracks)


@pytest.mark.parametrize("threshold", ["min_counts", "max_counts"])
def test_extract_loci_should_raise_when_a_count_threshold_comes_without_signals(
    corpus, threshold
):
    """Test the refusal for a count filter with nothing to count.

    Given:
        A count threshold and no signal track.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` rather than keep every locus
        unfiltered.
    """

    # Act & assert
    with pytest.raises(ValueError, match="min_counts and max_counts need `signals`"):
        loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW, **{threshold: 1})


@pytest.mark.parametrize(
    "threshold", [{"min_counts": 1}, {}], ids=["counted", "uncounted"]
)
@pytest.mark.parametrize("target_idx", [2, -3])
def test_extract_loci_should_raise_when_target_idx_names_no_signal_track(
    corpus, target_idx, threshold
):
    """Test the refusal for an index naming a track that is not there.

    Given:
        Two signal tracks and an index past either end of them, with a
        count threshold or with none.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the index and the number
        of tracks.
    """

    # Act & assert
    with pytest.raises(
        ValueError, match=r"target_idx {} names none of the 2 given".format(target_idx)
    ):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=corpus.tracks,
            target_idx=target_idx,
            **threshold,
        )


def test_extract_loci_should_read_no_channel_from_an_empty_signal_list(corpus, engine):
    """Test that an empty signal list leaves the default index unrefused.

    Given:
        The module's peaks, an empty list of signal tracks, an output
        window and no count threshold.
    When:
        The loci are extracted at the default ``target_idx``.
    Then:
        It should answer a signal tensor with one row per kept locus
        and no channel, raising nothing.
    """

    # Act
    extracted = loci.extract_loci(
        corpus.peaks, corpus.fasta, IN_WINDOW, OUT_WINDOW, signals=[]
    )

    # Assert
    assert extracted.signals.shape == (int(extracted.kept.sum()), 0, OUT_WINDOW)


def test_extract_loci_should_raise_when_a_residue_is_neither_encoded_nor_ignored(
    corpus,
):
    """Test that the residue refusal reaches a caller of the composition.

    Given:
        A locus whose input window covers the reference's ``N`` run,
        and nothing ignored.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` saying a residue is in neither
        the alphabet nor the ignored residues.
    """

    # Arrange
    locus = pandas.DataFrame([("chr1", 100, 140)])

    # Act & assert
    with pytest.raises(ValueError, match="not in `alphabet` or in `ignore`"):
        loci.extract_loci(locus, corpus.fasta, IN_WINDOW, ignore=())


def test_extract_loci_should_raise_when_chroms_names_no_contig_the_loci_use(corpus):
    """Test that the filter refusal reaches a caller of the composition.

    Given:
        The module's peaks, named ``chr1`` and ``chr2``, and a filter
        naming ``1`` and ``2``.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the locus file, the number
        of loci it holds, the contigs the loci name and the contigs the
        filter names, rather than answer an empty result.
    """

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"^none of the 9 loci in \S*peaks\.bed is on a contig `chroms` "
        r"names: the loci name `chr2`, `chr1` and `chroms` names `1`, `2`\.",
    ):
        loci.extract_loci(corpus.peaks, corpus.fasta, IN_WINDOW, chroms=["1", "2"])


def test_extract_loci_should_raise_when_a_locus_is_on_a_contig_the_reference_lacks(
    corpus,
):
    """Test that the placement refusal reaches a caller of the composition.

    Given:
        A locus frame with one locus on ``chr1`` and one on ``chrZ``,
        which the reference does not name.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the locus frame, the
        reference, the contig that is absent and the contigs the
        reference holds, rather than drop the locus it cannot place.
    """

    # Arrange
    frame = pandas.DataFrame([("chr1", 100, 140), ("chrZ", 100, 140)])

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"^1 of the 2 loci in the locus frame given cannot be placed: "
        r"\S*genome\.fa does not name `chrZ`\. The reference names `chr1`, "
        r"`chr2`\.",
    ):
        loci.extract_loci(frame, corpus.fasta, IN_WINDOW)


@pytest.mark.parametrize("label", ["signals", "controls"])
def test_extract_loci_should_raise_type_error_when_given_a_single_track(corpus, label):
    """Test the refusal for one track where a sequence of them is taken.

    Given:
        A single track path as the signals or as the controls.
    When:
        The loci are extracted.
    Then:
        It should raise `TypeError` naming the argument.
    """

    # Act & assert
    with pytest.raises(
        TypeError, match="`{}` must be a sequence of tracks".format(label)
    ):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            **{label: corpus.tracks[0]},
        )


def test_extract_loci_should_raise_type_error_when_a_track_is_no_track(corpus):
    """Test the refusal for something among the tracks that is not one.

    Given:
        A mapping of dense arrays among the signals.
    When:
        The loci are extracted.
    Then:
        It should raise `TypeError` naming the type it was given.
    """

    # Act & assert
    with pytest.raises(TypeError, match="Got dict"):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=[dense(TRACK_ROWS)],
        )


@pytest.mark.parametrize("label", ["signals", "controls"])
def test_extract_loci_should_raise_when_a_track_path_does_not_exist(
    corpus, tmp_path, label
):
    """Test the refusal for a track path with nothing at it.

    Given:
        A single-file track path at which nothing exists, and a locus
        set with no locus in it, so that no window would reach the
        track.
    When:
        The loci are extracted.
    Then:
        It should raise `FileNotFoundError` naming the path, rather
        than answer an empty result for a track it never opened.
    """

    # Arrange
    empty = pandas.DataFrame({"chrom": [], "start": [], "end": []})
    arguments = dict(signals=corpus.tracks[:1])
    arguments[label] = [str(tmp_path / "mistyped.parquet")]

    # Act & assert
    with pytest.raises(FileNotFoundError, match=r"mistyped\.parquet is not a track"):
        loci.extract_loci(empty, corpus.fasta, IN_WINDOW, OUT_WINDOW, **arguments)


def test_extract_loci_should_raise_when_no_signal_shares_a_contig_with_the_windows(
    corpus, engine
):
    """Test the refusal for tracks that would read back as all zeros.

    Given:
        The module's peaks kept to ``chr1``, and a signal track holding
        positions only on ``chr2``.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the contigs the track
        names and the contigs the windows are on.
    """

    # Arrange
    elsewhere = builders.track_rows_frame([("chr2", 220, 3.0)])

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"no `signals` track holds a position.*the tracks name `chr2` "
        r"and the windows name `chr1`",
    ):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=[elsewhere],
            chroms=["chr1"],
        )


def test_extract_loci_should_raise_when_no_control_shares_a_contig_with_the_windows(
    corpus, engine
):
    """Test that controls are held to the rule signals are.

    Given:
        The module's peaks kept to ``chr1``, a signal track that
        matches them and a control track holding positions only on
        ``chr2``.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the controls.
    """

    # Arrange
    elsewhere = builders.track_rows_frame([("chr2", 220, 3.0)])

    # Act & assert
    with pytest.raises(ValueError, match="no `controls` track holds a position"):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=corpus.tracks[:1],
            controls=[elsewhere],
            chroms=["chr1"],
        )


@pytest.mark.parametrize(
    "threshold", [{"min_counts": 1}, {"max_counts": -1}], ids=["floor", "ceiling"]
)
def test_extract_loci_should_raise_before_counting_when_no_signal_shares_a_contig(
    corpus, engine, threshold
):
    """Test that a count filter cannot hide the track refusal.

    Given:
        The module's peaks kept to ``chr1``, a signal track holding
        positions only on ``chr2``, and a count threshold every
        all-zero window falls outside: a floor of 1 or a ceiling of -1.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming the contigs the track
        names and the contigs the windows are on, rather than answer
        the empty result the count filter would leave.
    """

    # Arrange
    elsewhere = builders.track_rows_frame([("chr2", 220, 3.0)])

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"no `signals` track holds a position.*the tracks name `chr2` "
        r"and the windows name `chr1`",
    ):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=[elsewhere],
            chroms=["chr1"],
            **threshold,
        )


def test_extract_loci_should_compare_the_tracks_with_the_loci_still_standing(
    corpus, engine
):
    """Test which windows the tracks are held against.

    Given:
        Peaks on ``chr1`` and ``chr2``, an exclusion list covering all
        of ``chr2``, and a signal track holding positions only on
        ``chr2``.
    When:
        The loci are extracted.
    Then:
        It should raise `ValueError` naming ``chr1`` as the windows'
        contig, the ``chr2`` locus having been excluded before the
        tracks were compared.
    """

    # Arrange
    elsewhere = builders.track_rows_frame([("chr2", 220, 3.0)])
    regions = pandas.DataFrame([("chr2", 0, 500)])

    # Act & assert
    with pytest.raises(ValueError, match=r"the windows name `chr1`\."):
        loci.extract_loci(
            corpus.peaks,
            corpus.fasta,
            IN_WINDOW,
            OUT_WINDOW,
            signals=[elsewhere],
            exclusion_lists=regions,
        )


def test_extract_loci_should_accept_a_window_contig_no_track_holds_a_position_on(
    corpus, engine
):
    """Test that one contig without signal is not a mismatch.

    Given:
        Windows on ``chr1`` and ``chr2``, and a signal track holding
        positions only on ``chr1``.
    When:
        The loci are extracted.
    Then:
        It should raise nothing, answer zeros for the ``chr2`` window
        and signal for the ``chr1`` windows.
    """

    # Arrange
    one = builders.track_rows_frame([row for row in TRACK_ROWS if row[0] == "chr1"])

    # Act
    extracted = loci.extract_loci(
        corpus.peaks, corpus.fasta, IN_WINDOW, OUT_WINDOW, signals=[one]
    )

    # Assert
    assert extracted.kept[0]
    assert not extracted.signals[0].any()
    assert extracted.signals[1:].sum() > 0


def test_extract_loci_should_answer_no_rows_when_the_count_filter_keeps_nothing(
    corpus, engine
):
    """Test that a count filter emptying the set is not a mismatch.

    Given:
        The module's peaks, a signal track that matches them and a
        floor no locus reaches.
    When:
        The loci are extracted.
    Then:
        It should raise nothing and answer tensors with no rows and a
        mask with no true entry.
    """

    # Act
    extracted = loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=corpus.tracks[:1],
        min_counts=10**9,
    )

    # Assert
    assert extracted.signals.shape == (0, 1, OUT_WINDOW)
    assert extracted.sequences.shape == (0, 4, IN_WINDOW)
    assert not extracted.kept.any()


def test_extract_loci_should_answer_no_rows_when_the_exclusion_lists_keep_nothing(
    corpus, engine, tmp_path
):
    """Test that an exclusion list emptying the set is not a mismatch.

    Given:
        The module's peaks, an exclusion list covering both contigs,
        and a signal track sharing no contig with the peaks.
    When:
        The loci are extracted.
    Then:
        It should raise nothing and answer tensors with no rows, there
        being no window left for the track to disagree with.
    """

    # Arrange
    everything = builders.write_bed(
        str(tmp_path / "exclude_all.bed.gz"), [("chr1", 0, 1000), ("chr2", 0, 500)]
    )
    elsewhere = builders.track_rows_frame([("chrUn", 5, 1.0)])

    # Act
    extracted = loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=[elsewhere],
        exclusion_lists=[everything],
    )

    # Assert
    assert extracted.signals.shape == (0, 1, OUT_WINDOW)
    assert extracted.sequences.shape == (0, 4, IN_WINDOW)
    assert not extracted.kept.any()


def test_extract_loci_should_leave_no_table_bound(corpus, engine):
    """Test that nothing an extraction registers outlives it.

    Given:
        The session's tables before an extraction.
    When:
        Loci are extracted with signal, controls and an exclusion list.
    Then:
        It should leave the session with the same tables.
    """

    # Arrange
    before = quadrat_tables(engine)

    # Act
    loci.extract_loci(
        corpus.peaks,
        corpus.fasta,
        IN_WINDOW,
        OUT_WINDOW,
        signals=corpus.tracks[:1],
        controls=corpus.tracks[1:],
        exclusion_lists=[corpus.exclusion],
    )

    # Assert
    assert quadrat_tables(engine) == before


@settings(max_examples=25, deadline=None)
@given(
    rows=locus_rows,
    regions=exclusion_rows,
    min_counts=count_floors,
    signal_positions=signal_tracks,
    control_positions=control_tracks,
    target_idx=target_indices,
    in_window=in_windows,
    out_window=out_windows,
    max_jitter=jitters,
)
def test_extract_loci_should_answer_the_per_window_reads_of_the_loci_it_keeps(
    corpus,
    engine,
    rows,
    regions,
    min_counts,
    signal_positions,
    control_positions,
    target_idx,
    in_window,
    out_window,
    max_jitter,
):
    """Test the rows of an extraction against its kept mask, over arbitrary inputs.

    Given:
        Any loci on the module's two contigs, some reaching off a
        contig's edge; no exclusion list or any regions; no count floor
        or any; two in-memory signal tracks, the count taken on either;
        one or two in-memory control tracks; and any window widths and
        jitter.
    When:
        The loci are extracted.
    Then:
        The mask should have one entry per locus, true exactly where
        the locus's windows lie on its contig, share no bin with a
        region and count at least the floor on the target track; the
        sequences, signals and controls should each have one row per
        true entry; and each row should equal what `extract_sequences`,
        or `extract_signal` over the output window for a signal and
        the input window for a control, reads at that locus's windows
        as `load_windows` answers them.
    """

    # Arrange
    frame = pandas.DataFrame(
        {
            "chrom": [row[0] for row in rows],
            "start": [row[1] for row in rows],
            "end": [row[1] + row[2] for row in rows],
        }
    )
    # A position on each contig in every track, so that no draw is
    # refused for a track sharing no contig with the windows.
    anchors = [(chrom, 0, 1.0) for chrom in sorted(TWO_CONTIGS)]
    signal_rows = [anchors + positions for positions in signal_positions]
    signals = [builders.track_rows_frame(rows_of) for rows_of in signal_rows]
    controls = [
        builders.track_rows_frame(anchors + positions) for positions in control_positions
    ]
    windows = loci.load_windows(
        frame, corpus.fasta, in_window, out_window, max_jitter=max_jitter
    )
    target = dense(signal_rows[target_idx])
    expected_kept = [
        bool(
            window["in_bounds"]
            and not touches(
                (window["chrom"], window["outer_start"], window["outer_end"]),
                regions or [],
            )
            and (
                min_counts is None
                or target[window["chrom"]][window["sig_start"] : window["sig_end"]].sum()
                >= min_counts
            )
        )
        for window in windows.to_dicts()
    ]

    # Act
    extracted = loci.extract_loci(
        frame,
        corpus.fasta,
        in_window,
        out_window,
        signals=signals,
        controls=controls,
        max_jitter=max_jitter,
        min_counts=min_counts,
        target_idx=target_idx,
        exclusion_lists=None if regions is None else pandas.DataFrame(regions),
    )

    # Assert
    kept_windows = windows.filter(polars.Series(expected_kept, dtype=polars.Boolean))
    assert len(extracted.kept) == len(rows)
    assert extracted.kept.tolist() == expected_kept
    tensors = [
        (extracted.sequences, loci.extract_sequences(kept_windows, corpus.fasta)),
        (extracted.signals, loci.extract_signal(kept_windows, signals)),
        (
            extracted.controls,
            loci.extract_signal(
                kept_windows, controls, bound_columns=("in_start", "in_end")
            ),
        ),
    ]
    for answered, read in tensors:
        assert answered.shape[0] == int(extracted.kept.sum())
        for index in range(len(kept_windows)):
            assert torch.equal(answered[index], read[index])
