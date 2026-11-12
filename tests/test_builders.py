"""Tests for the synthetic-input builders the suite is built on.

The builders are test infrastructure, which is exactly why they are
tested: every later assertion about the data path is an assertion about
data these functions produced, so a builder that silently writes the
wrong thing makes the suite agree with itself and with nothing else.

Tests that read an alignment back request the `pysam` fixture, which
owns the skip.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import gzip
import os
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import quadrat
from quadrat import stores
from tests import (
    FILE_BACKED,
    THREE_CONTIGS,
    TWO_CONTIGS,
    builders,
    parse_fasta,
    record_placements,
)

# Contig names over lowercase letters only, so a generated name cannot
# collide with FASTA or BED syntax. Lengths stay small because every
# draw writes a real file, and never zero, because `faidx` refuses a
# contig with no sequence.
contig_names = st.text(
    st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=4
)
contig_sizes = st.dictionaries(
    contig_names, st.integers(min_value=1, max_value=200), min_size=1, max_size=4
)

# Contigs of at least the default read length, which is the floor
# `write_random_bam` refuses to go below: a shorter contig cannot hold
# a 20M record ending before the end its own header declares.
alignment_sizes = st.dictionaries(
    contig_names, st.integers(min_value=20, max_value=200), min_size=1, max_size=4
)

# Rows of a contig, a start and a stop followed by none to seven text
# fields of mixed width, every field printable and free of tabs, so
# that rows in one file disagree on how many fields they carry.
bed_rows = st.lists(
    st.builds(
        lambda chrom, start, stop, rest: (chrom, start, stop, *rest),
        contig_names,
        st.integers(min_value=0, max_value=10**6),
        st.integers(min_value=0, max_value=10**6),
        st.lists(
            st.text(
                st.characters(min_codepoint=32, max_codepoint=126),
                min_size=0,
                max_size=12,
            ),
            max_size=7,
        ),
    ),
    max_size=10,
)


@st.composite
def geometries(draw):
    """Draw a contig mapping with masked intervals that may overrun their contigs.

    The intervals are drawn against the widest contig the mapping
    allows rather than against each contig's own length, and either
    coordinate may fall below zero, so an interval running off either
    end — or lying wholly outside the contig on either side — is a
    case the invariant sees rather than one it is protected from. Soft
    masks and N runs are drawn from one domain because either may
    appear anywhere.
    """

    sizes = draw(contig_sizes)
    intervals = st.lists(
        st.tuples(
            st.sampled_from(sorted(sizes)),
            st.integers(min_value=-50, max_value=250),
            st.integers(min_value=-50, max_value=250),
        ),
        max_size=3,
    )
    return sizes, draw(intervals), draw(intervals)


@st.composite
def alignments(draw, read_length=20):
    """Draw a contig mapping with reads placed on it, some of them unplaced.

    A placement of ``None`` is an unplaced record, which the sort has
    to carry rather than locate. Placed reads sit far enough inside
    their contig to end before it does, so no draw asks the writer to
    produce a record the header contradicts.
    """

    sizes = draw(alignment_sizes)
    names = sorted(sizes)
    placements = []
    for chrom in draw(st.lists(st.sampled_from(names + [None]), max_size=8)):
        if chrom is None:
            placements.append((None, None))
        else:
            stop = sizes[chrom] - read_length
            placements.append((chrom, draw(st.integers(min_value=0, max_value=stop))))
    return sizes, placements


@st.composite
def cigars(draw):
    """Draw a CIGAR whose soft clips sit only at the ends and that consumes query.

    At least one ``M`` is forced, because a CIGAR of pure deletions
    infers a query length of zero and the builder's fallback would then
    disagree with the CIGAR for reasons that are not a defect.
    """

    middle = draw(
        st.lists(
            st.tuples(st.sampled_from("MID"), st.integers(min_value=1, max_value=20)),
            min_size=1,
            max_size=3,
        )
    )
    if not any(operation == "M" for operation, _ in middle):
        middle.append(("M", draw(st.integers(min_value=1, max_value=20))))

    operations = []
    leading = draw(st.integers(min_value=0, max_value=10))
    if leading:
        operations.append(("S", leading))
    operations.extend(middle)
    trailing = draw(st.integers(min_value=0, max_value=10))
    if trailing:
        operations.append(("S", trailing))
    return "".join("{}{}".format(length, op) for op, length in operations)


def sequence_lines(path):
    """Return the sequence lines of a FASTA, headers dropped."""

    with open(path) as handle:
        return [line.strip() for line in handle if not line.startswith(">")]


def read_bed(path):
    """Return the rows read back from a BED, gzipped or not, as tuples of fields.

    The start and stop come back as integers and every further field
    as the text it was written as.
    """

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        rows = []
        for line in handle:
            if not line.strip():
                continue
            chrom, start, stop, *rest = line.rstrip("\n").split("\t")
            rows.append((chrom, int(start), int(stop), *rest))
    return rows


def covered_positions(intervals, sizes):
    """Return the ``(contig, offset)`` pairs a set of intervals covers.

    Clamped to each contig, because an interval is allowed to name
    coordinates past the end of the one it masks.
    """

    return {
        (chrom, offset)
        for chrom, start, stop in intervals
        for offset in range(
            min(max(start, 0), sizes[chrom]), min(max(stop, 0), sizes[chrom])
        )
    }


@pytest.fixture
def two_contig_header(pysam):
    """Return an alignment header over the suite's two-contig geometry."""

    return builders.alignment_header(TWO_CONTIGS)


def test_write_fasta_should_write_every_contig_at_its_declared_length(tmp_path):
    """Test that the FASTA holds exactly the requested bases.

    Given:
        A mapping of two contig names to lengths.
    When:
        The builder renders it.
    Then:
        It should write each contig at its declared length over the
        DNA alphabet.
    """

    # Arrange
    path = str(tmp_path / "genome.fa")

    # Act
    builders.write_fasta(path, TWO_CONTIGS)

    # Assert
    sequences = parse_fasta(path)
    assert {name: len(bases) for name, bases in sequences.items()} == TWO_CONTIGS
    assert set("".join(sequences.values())) <= set("ACGT")


def test_write_fasta_should_mask_only_the_intervals_it_is_given(tmp_path):
    """Test that soft masking and N runs land where they were asked for.

    Given:
        A soft-masked interval and an N run on a known contig, with
        unmasked sequence on both sides of each.
    When:
        The builder renders it.
    Then:
        It should lowercase only the first interval, fill only the
        second with N, and leave every base outside them an uppercase
        ACGT.
    """

    # Arrange
    path = str(tmp_path / "masked.fa")

    # Act
    builders.write_fasta(
        path, TWO_CONTIGS, soft_mask=[("chr1", 100, 120)], n_runs=[("chr1", 200, 210)]
    )

    # Assert
    bases = parse_fasta(path)["chr1"]
    assert bases[100:120] == bases[100:120].lower()
    assert bases[200:210] == "N" * 10
    untouched = bases[:100] + bases[120:200] + bases[210:]
    assert set(untouched) <= set("ACGT")
    assert len(bases) == TWO_CONTIGS["chr1"]


def test_write_fasta_should_produce_identical_bases_when_the_seed_is_reused(tmp_path):
    """Test that one seed renders one sequence.

    Given:
        Two renders of one geometry under the same seed.
    When:
        The bases are compared.
    Then:
        It should produce identical bases, so every later assertion
        about this data is reproducible.
    """

    # Arrange
    first = str(tmp_path / "a.fa")
    second = str(tmp_path / "b.fa")

    # Act
    builders.write_fasta(first, TWO_CONTIGS, seed=7)
    builders.write_fasta(second, TWO_CONTIGS, seed=7)

    # Assert
    assert parse_fasta(first) == parse_fasta(second)


def test_write_fasta_should_produce_different_bases_when_the_seed_differs(tmp_path):
    """Test that the seed actually reaches the generator.

    Given:
        Two renders of one geometry under different seeds.
    When:
        The bases are compared.
    Then:
        It should produce different bases, so a geometry can be
        rendered more than one way.
    """

    # Arrange
    first = str(tmp_path / "a.fa")
    second = str(tmp_path / "b.fa")

    # Act
    builders.write_fasta(first, TWO_CONTIGS, seed=7)
    builders.write_fasta(second, TWO_CONTIGS, seed=8)

    # Assert
    assert parse_fasta(first) != parse_fasta(second)


def test_write_fasta_should_wrap_every_line_at_the_requested_width(tmp_path):
    """Test that the line width reaches the file rather than the default.

    Given:
        A contig longer than the requested line width.
    When:
        The builder renders it.
    Then:
        It should wrap every sequence line but the last at that width,
        which is what makes the file indexable.
    """

    # Arrange
    path = str(tmp_path / "wrapped.fa")

    # Act
    builders.write_fasta(path, {"chr1": 25}, line_width=10)

    # Assert
    assert [len(line) for line in sequence_lines(path)] == [10, 10, 5]


def test_write_fasta_should_preserve_the_length_when_an_n_run_overruns(tmp_path):
    """Test that an out-of-range N run cannot lengthen a contig.

    Given:
        An N run whose stop lies past the end of its contig.
    When:
        The builder renders it.
    Then:
        It should still write the contig at its declared length, with
        the run clipped to the end, because a contig longer than its own
        header would make the suite agree with itself and nothing else.
    """

    # Arrange
    path = str(tmp_path / "overrun.fa")

    # Act
    builders.write_fasta(path, {"chr1": 100}, n_runs=[("chr1", 90, 120)])

    # Assert
    bases = parse_fasta(path)["chr1"]
    assert len(bases) == 100
    assert bases[90:] == "N" * 10


def test_write_fasta_should_mask_from_the_contig_start_when_a_soft_mask_underruns(
    tmp_path,
):
    """Test that a soft mask beginning before a contig still masks it.

    Given:
        A soft mask whose start lies before the beginning of its contig.
    When:
        The builder renders it.
    Then:
        It should lowercase from the contig's first base to the mask's
        stop, because a negative start truncates to an empty slice and
        would otherwise mask nothing at all.
    """

    # Arrange
    path = str(tmp_path / "underrun.fa")

    # Act
    builders.write_fasta(path, {"chr1": 100}, soft_mask=[("chr1", -5, 10)])

    # Assert
    bases = parse_fasta(path)["chr1"]
    assert bases[:10].islower()
    assert bases[10:].isupper()


def test_write_fasta_should_preserve_the_length_when_an_n_run_lies_wholly_before_its_contig(
    tmp_path,
):
    """Test that an N run entirely below zero masks nothing and removes nothing.

    Given:
        An N run whose start and stop both lie before the beginning of
        its contig.
    When:
        The builder renders it.
    Then:
        It should write the contig at its declared length with no base
        masked, because a negative stop counts back from the end of the
        sequence and would otherwise fill a slice of almost the whole
        contig with a run narrow enough to delete the rest of it.
    """

    # Arrange
    path = str(tmp_path / "before.fa")

    # Act
    builders.write_fasta(path, {"chr1": 100}, n_runs=[("chr1", -50, -10)])

    # Assert
    bases = parse_fasta(path)["chr1"]
    assert len(bases) == 100
    assert "N" not in bases


def test_write_fasta_should_mask_nothing_when_a_soft_mask_lies_wholly_before_its_contig(
    tmp_path,
):
    """Test that a soft mask entirely below zero lowercases nothing.

    Given:
        A soft mask whose start and stop both lie before the beginning
        of its contig.
    When:
        The builder renders it.
    Then:
        It should leave every base upper case, because a negative stop
        counts back from the end of the sequence and would otherwise
        lowercase positions the interval never named.
    """

    # Arrange
    path = str(tmp_path / "before.fa")

    # Act
    builders.write_fasta(path, {"chr1": 100}, soft_mask=[("chr1", -50, -10)])

    # Assert
    bases = parse_fasta(path)["chr1"]
    assert len(bases) == 100
    assert bases.isupper()


def test_write_fasta_should_write_an_index_when_indexing_is_requested(tmp_path, pysam):
    """Test that an index is written beside the sequence.

    Given:
        A two-contig geometry with indexing requested.
    When:
        The builder renders it.
    Then:
        It should write an index beside the file from which each
        contig's full sequence can be fetched, which is the path the
        sequence stores read through.
    """

    # Arrange
    path = str(tmp_path / "indexed.fa")

    # Act
    builders.write_fasta(path, TWO_CONTIGS, index=True)

    # Assert
    assert os.path.exists(path + ".fai")
    with pysam.FastaFile(path) as reference:
        lengths = {chrom: len(reference.fetch(chrom)) for chrom in reference.references}
    assert lengths == TWO_CONTIGS


@settings(FILE_BACKED, max_examples=25)
@given(geometry=geometries(), line_width=st.integers(min_value=1, max_value=80))
def test_write_fasta_should_hold_its_declared_lengths_for_any_geometry(
    tmp_path, geometry, line_width
):
    """Test the length invariant across arbitrary geometries.

    Given:
        Any mapping of contig names to positive lengths, any soft
        masks and any N runs over them including intervals past a
        contig's end, at any line width.
    When:
        The builder renders it and the file is parsed back.
    Then:
        It should hold every contig at exactly its declared length,
        within the masked and unmasked DNA alphabet, with no line
        longer than the width.
    """

    # Arrange
    sizes, soft_mask, n_runs = geometry
    path = str(tmp_path / "property.fa")

    # Act
    builders.write_fasta(
        path,
        sizes,
        soft_mask=soft_mask,
        n_runs=n_runs,
        line_width=line_width,
    )

    # Assert
    sequences = parse_fasta(path)
    assert {name: len(bases) for name, bases in sequences.items()} == sizes
    assert set("".join(sequences.values())) <= set("ACGTNacgt")
    assert max(len(line) for line in sequence_lines(path)) <= line_width


@settings(FILE_BACKED, max_examples=25)
@given(geometry=geometries())
def test_write_fasta_should_mask_every_interval_it_is_given(tmp_path, geometry):
    """Test that both masking kinds reach every base they name.

    Given:
        Any contig mapping with any soft masks and N runs over it,
        including intervals reaching past a contig's end.
    When:
        The builder renders it and the file is parsed back.
    Then:
        It should leave no position inside a requested interval
        unmasked, i.e., ``N`` where a run covers it and lower case
        where only a soft mask does.
    """

    # Arrange
    sizes, soft_mask, n_runs = geometry
    path = str(tmp_path / "masked.fa")

    # Act
    builders.write_fasta(path, sizes, soft_mask=soft_mask, n_runs=n_runs)

    # Assert
    sequences = parse_fasta(path)
    runs = covered_positions(n_runs, sizes)
    soft = covered_positions(soft_mask, sizes) - runs
    assert all(sequences[chrom][offset] == "N" for chrom, offset in runs)
    assert all(sequences[chrom][offset].islower() for chrom, offset in soft)


@pytest.mark.parametrize("suffix", [".bed", ".bed.gz"])
def test_write_bed_should_round_trip_a_row_for_either_suffix(tmp_path, suffix):
    """Test that the path suffix decides the compression.

    Given:
        One row and a path whose suffix is plain or gzipped.
    When:
        The builder writes it and the matching reader reads it back.
    Then:
        It should produce the same tab-separated line either way, so a
        caller reads back what it wrote whichever suffix it chose.
    """

    # Arrange
    path = str(tmp_path / ("x" + suffix))

    # Act
    builders.write_bed(path, [("chr1", 500, 600)])

    # Assert
    assert read_bed(path) == [("chr1", 500, 600)]


def test_write_bed_should_write_every_field_a_row_carries(tmp_path):
    """Test that a wider row is written through at its own width.

    Given:
        A six-field peak row, a ten-field row carrying a summit offset
        and a three-field row.
    When:
        The builder writes them.
    Then:
        It should write one line per row in order, each with every
        field its row carried.
    """

    # Arrange
    path = str(tmp_path / "wide.bed")
    rows = [
        ("chr1", 100, 200, "peak", 50, "+"),
        ("chr1", 300, 400, "peak", 0, ".", 0, 0, 0, 40),
        ("chr2", 10, 20),
    ]

    # Act
    builders.write_bed(path, rows)

    # Assert
    assert read_bed(path) == [
        ("chr1", 100, 200, "peak", "50", "+"),
        ("chr1", 300, 400, "peak", "0", ".", "0", "0", "0", "40"),
        ("chr2", 10, 20),
    ]


@settings(FILE_BACKED, max_examples=25)
@given(rows=bed_rows)
def test_write_bed_should_round_trip_any_rows(tmp_path, rows):
    """Test the write-and-read roundtrip over arbitrary rows.

    Given:
        Any list of contig, start and stop rows, each followed by none
        to seven text fields of mixed width, written to both a plain
        and a gzipped path.
    When:
        Each file is read back with its matching reader.
    Then:
        It should return the original rows in order from both, every
        field a row carried and no other.
    """

    # Arrange
    plain = str(tmp_path / "rows.bed")
    packed = str(tmp_path / "rows.bed.gz")

    # Act
    builders.write_bed(plain, rows)
    builders.write_bed(packed, rows)

    # Assert
    assert read_bed(plain) == rows
    assert read_bed(packed) == rows


def test_intervals_should_step_across_the_range_at_a_fixed_width():
    """Test that the interval helper tiles a range as asked.

    Given:
        A range of 400 to 600 at a step of 75 and a width of 20.
    When:
        The builder builds the rows.
    Then:
        It should return every start in the range, each paired with a
        stop one width along.
    """

    # Act
    rows = builders.intervals("chr1", 400, 600, 75, 20)

    # Assert
    assert rows == [("chr1", 400, 420), ("chr1", 475, 495), ("chr1", 550, 570)]


@settings(max_examples=50)
@given(
    start=st.integers(min_value=-1000, max_value=1000),
    stop=st.integers(min_value=-1000, max_value=1000),
    step=st.integers(min_value=1, max_value=100),
    width=st.integers(min_value=1, max_value=100),
)
def test_intervals_should_tile_the_range_for_any_geometry(start, stop, step, width):
    """Test the tiling invariant across the helper's whole domain.

    Given:
        Any start, stop, positive step and positive width.
    When:
        The builder tiles the range.
    Then:
        It should return one row per step across the range, each
        spanning exactly one width, and nothing at all once the start
        has caught the stop.
    """

    # Act
    rows = builders.intervals("chr1", start, stop, step, width)

    # Assert
    assert [row[1] for row in rows] == list(range(start, stop, step))
    assert all(row[2] - row[1] == width for row in rows)
    assert all(row[0] == "chr1" for row in rows)


def test_alignment_header_should_declare_its_contigs_in_order(pysam):
    """Test that the header names every contig at its length.

    Given:
        A mapping of two contig names to lengths.
    When:
        The builder builds a header.
    Then:
        It should mark the order coordinate-sorted and declare each
        contig at its length in insertion order, each resolving to its
        insertion index.
    """

    # Act
    header = builders.alignment_header(TWO_CONTIGS)

    # Assert
    as_dict = header.to_dict()
    assert as_dict["HD"]["SO"] == "coordinate"
    assert [entry["SN"] for entry in as_dict["SQ"]] == list(TWO_CONTIGS)
    assert [entry["LN"] for entry in as_dict["SQ"]] == list(TWO_CONTIGS.values())
    for index, chrom in enumerate(TWO_CONTIGS):
        assert header.get_tid(chrom) == index


@settings(max_examples=25)
@given(sizes=contig_sizes)
def test_alignment_header_should_declare_any_contig_mapping(pysam, sizes):
    """Test that the header echoes whatever mapping it is given.

    Given:
        Any mapping of contig names to positive lengths.
    When:
        The builder builds a header from it.
    Then:
        It should declare every contig at its length in the mapping's
        own order and resolve each name to that position, so a read
        placed by name lands on the contig the caller named.
    """

    # Act
    header = builders.alignment_header(sizes)

    # Assert
    declared = [(entry["SN"], entry["LN"]) for entry in header.to_dict()["SQ"]]
    assert declared == list(sizes.items())
    assert [header.get_tid(chrom) for chrom in sizes] == list(range(len(sizes)))


def test_aligned_read_should_size_the_query_from_the_cigar(two_contig_header):
    """Test that a soft-clipped CIGAR gets a query long enough for it.

    Given:
        A read with five soft-clipped and fifteen matched bases.
    When:
        The builder builds it.
    Then:
        It should give the record a twenty-base query, so the CIGAR and
        the sequence agree.
    """

    # Act
    read = builders.aligned_read(
        two_contig_header, "clipped", "chr1", 300, cigar="5S15M"
    )

    # Assert
    assert read.query_length == 20


def test_aligned_read_should_place_the_read_at_the_requested_position(two_contig_header):
    """Test that the contig and position reach the record.

    Given:
        A read placed at a known position on a known contig.
    When:
        The builder builds it.
    Then:
        It should resolve the contig to its header id and start at that
        position.
    """

    # Act
    read = builders.aligned_read(two_contig_header, "placed", "chr2", 300)

    # Assert
    assert read.reference_id == two_contig_header.get_tid("chr2")
    assert read.reference_start == 300


def test_aligned_read_should_build_an_unplaced_record_when_the_contig_is_none(
    two_contig_header,
):
    """Test that passing no contig yields an unplaced read.

    Given:
        A read whose contig is None.
    When:
        The builder builds it.
    Then:
        It should carry no reference and no start while keeping the
        CIGAR it was given, so a record with no coordinates is
        something a caller can construct rather than only encounter,
        and the query length still follows the CIGAR there.
    """

    # Act
    read = builders.aligned_read(two_contig_header, "unplaced", None, None, cigar="35M")

    # Assert
    assert read.reference_id == -1
    assert read.reference_start == -1
    assert read.cigarstring == "35M"
    assert read.query_length == 35


def test_aligned_read_should_carry_the_flag_and_mapping_quality_given(two_contig_header):
    """Test that the flag and mapping quality are not defaulted away.

    Given:
        A read given the reverse flag and an explicit mapping quality.
    When:
        The builder builds it.
    Then:
        It should mark the record reverse and carry that mapping
        quality.
    """

    # Act
    read = builders.aligned_read(two_contig_header, "rev", "chr1", 100, flag=16, mapq=7)

    # Assert
    assert read.is_reverse
    assert read.mapping_quality == 7


def test_aligned_read_should_exclude_hard_clipped_bases_from_the_query(
    two_contig_header,
):
    """Test that a hard clip is absent from the query it sizes.

    Given:
        A hard-clipped CIGAR over a fifteen-base match.
    When:
        The builder builds it.
    Then:
        It should size the query from the matched span alone, because
        a hard clip names bases the record does not carry.
    """

    # Act
    read = builders.aligned_read(two_contig_header, "hard", "chr1", 100, cigar="5H15M")

    # Assert
    assert read.query_length == 15


def test_aligned_read_should_leave_the_record_unreferenced_when_unknown(
    two_contig_header,
):
    """Test that an undeclared contig gives a half-placed record.

    Given:
        A contig name the header does not declare.
    When:
        The builder builds it.
    Then:
        It should leave the record without a reference while keeping
        its start and CIGAR -- a half-placed record, distinct from the
        unplaced one a None contig produces.
    """

    # Act
    read = builders.aligned_read(two_contig_header, "stray", "chrZ", 250)

    # Assert
    assert read.reference_id == -1
    assert read.reference_start == 250
    assert read.cigarstring == "20M"


@settings(FILE_BACKED, max_examples=25)
@given(cigar=cigars())
def test_aligned_read_should_match_query_and_qualities_to_any_cigar(
    two_contig_header, cigar
):
    """Test that the sequence and qualities always agree with the CIGAR.

    Given:
        Any CIGAR assembled from match, soft-clip, insert and delete
        operations, consuming at least one query base.
    When:
        The builder builds the record.
    Then:
        It should give the sequence and the qualities equal length,
        both matching the length the CIGAR consumes -- an invariant
        pysam raises on rather than tolerates.
    """

    # Act
    read = builders.aligned_read(
        two_contig_header, "generated", "chr1", 100, cigar=cigar
    )

    # Assert
    assert read.query_length == read.infer_query_length()
    assert len(read.query_qualities) == read.query_length


def test_write_bam_should_write_a_sorted_indexed_file(
    tmp_path, pysam, two_contig_header
):
    """Test that reads handed over out of order come back sorted.

    Given:
        Three reads supplied in descending coordinate order.
    When:
        The builder writes them.
    Then:
        It should emit them in ascending coordinate order beside an
        index, and return the path.
    """

    # Arrange
    reads = [
        builders.aligned_read(two_contig_header, "r{}".format(index), "chr1", pos)
        for index, pos in enumerate((400, 200, 100))
    ]
    path = str(tmp_path / "reads.bam")

    # Act
    written = builders.write_bam(path, reads, two_contig_header)

    # Assert
    assert written == path
    assert os.path.exists(path + ".bai")
    assert [start for _, start in record_placements(path, pysam)] == [100, 200, 400]


def test_write_bam_should_remove_its_staging_file(tmp_path, two_contig_header):
    """Test that the unsorted intermediate does not outlive the call.

    Given:
        Reads written through the staging file the builder sorts from.
    When:
        The builder finishes.
    Then:
        It should leave only the BAM and its index, so nothing later
        reaching into the directory can pick up a file whose order
        disagrees with that index.
    """

    # Arrange
    reads = [builders.aligned_read(two_contig_header, "r0", "chr1", 100)]
    path = str(tmp_path / "reads.bam")

    # Act
    builders.write_bam(path, reads, two_contig_header)

    # Assert
    assert sorted(os.listdir(tmp_path)) == ["reads.bam", "reads.bam.bai"]


def test_write_bam_should_sort_by_contig_before_position(
    tmp_path, pysam, two_contig_header
):
    """Test that the sort key is the contig order, then the position.

    Given:
        Four reads across two contigs in scrambled order.
    When:
        The builder writes them.
    Then:
        It should order them by contig as the header declares them,
        then by position within each.
    """

    # Arrange
    placements = [("chr2", 400), ("chr1", 900), ("chr2", 100), ("chr1", 50)]
    reads = [
        builders.aligned_read(two_contig_header, "r{}".format(index), chrom, pos)
        for index, (chrom, pos) in enumerate(placements)
    ]
    path = str(tmp_path / "multi.bam")

    # Act
    builders.write_bam(path, reads, two_contig_header)

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        observed = [(read.reference_name, read.reference_start) for read in bam.fetch()]
    assert observed == [("chr1", 50), ("chr1", 900), ("chr2", 100), ("chr2", 400)]


@settings(FILE_BACKED, max_examples=25)
@given(alignment=alignments())
def test_write_bam_should_sort_and_preserve_any_reads(tmp_path, pysam, alignment):
    """Test that the file holds its input, in coordinate order.

    Given:
        Any reads over any contig mapping, handed over in any order,
        including unplaced ones.
    When:
        The builder writes them and every record is read back.
    Then:
        It should hold exactly those reads with the placed ones in
        contig then position order, so no input can produce a file
        whose order disagrees with its index.
    """

    # Arrange
    sizes, placements = alignment
    header = builders.alignment_header(sizes)
    reads = [
        builders.aligned_read(header, "r{}".format(index), chrom, start)
        for index, (chrom, start) in enumerate(placements)
    ]
    path = str(tmp_path / "property.bam")

    # Act
    builders.write_bam(path, reads, header)

    # Assert
    # Unplaced records sort last, which is the coordinate-sort
    # convention rather than what a plain ``(tid, start)`` key over a
    # reference id of -1 would give.
    placed = sorted(
        [row for row in placements if row[0] is not None],
        key=lambda row: (header.get_tid(row[0]), row[1]),
    )
    unplaced = [(None, -1) for row in placements if row[0] is None]
    assert record_placements(path, pysam, until_eof=True) == placed + unplaced


def test_write_bam_should_keep_an_unplaced_record(tmp_path, pysam, two_contig_header):
    """Test that a record with no contig survives the round trip.

    Given:
        One placed read and one unplaced one.
    When:
        The builder writes them and every record is read back.
    Then:
        It should hold both with the unplaced one last, so a count
        taken over the file counts records written rather than records
        the index can find.
    """

    # Arrange
    reads = [
        builders.aligned_read(two_contig_header, "placed", "chr1", 100),
        builders.aligned_read(two_contig_header, "unplaced", None, None),
    ]
    path = str(tmp_path / "mixed.bam")

    # Act
    builders.write_bam(path, reads, two_contig_header)

    # Assert
    assert record_placements(path, pysam, until_eof=True) == [
        ("chr1", 100),
        (None, -1),
    ]


def test_write_random_bam_should_raise_when_a_contig_is_shorter_than_one_read(
    tmp_path, pysam
):
    """Test that a contig too short to hold a read is refused.

    Given:
        A geometry whose only contig is shorter than the read length.
    When:
        The builder is asked to place reads on it.
    Then:
        It should raise ValueError naming that contig, because a random
        offset on it would put the record past the end the header
        declares.
    """

    # Arrange
    sizes = {"chr1": 10}
    header = builders.alignment_header(sizes)

    # Act & assert
    with pytest.raises(ValueError, match="shorter than the read length of 20: chr1"):
        builders.write_random_bam(str(tmp_path / "short.bam"), header, sizes)


def test_write_random_bam_should_accept_a_contig_as_long_as_one_read(tmp_path, pysam):
    """Test that the shortest usable contig is not refused.

    Given:
        A contig exactly as long as the read length.
    When:
        The builder places one read on it.
    Then:
        It should write that read across the whole contig, because a
        contig that can hold a read is one the builder has no reason to
        turn down.
    """

    # Arrange
    sizes = {"chr1": 20}
    header = builders.alignment_header(sizes)
    path = str(tmp_path / "exact.bam")

    # Act
    builders.write_random_bam(path, header, sizes, n_reads=1)

    # Assert
    assert record_placements(path, pysam) == [("chr1", 0)]


def test_write_random_bam_should_reach_the_last_base_of_a_contig(tmp_path, pysam):
    """Test that no stretch of a contig is out of reach of a read.

    Given:
        A contig twice the read length, and a seed that draws the
        latest start the contig admits.
    When:
        The builder places one read on it.
    Then:
        It should end at the contig's last base, so a suite expecting
        signal near a boundary is not handed a region no read can ever
        occupy.
    """

    # Arrange
    sizes = {"chr1": 40}
    header = builders.alignment_header(sizes)
    path = str(tmp_path / "tail.bam")

    # Act
    builders.write_random_bam(path, header, sizes, n_reads=1, seed=27)

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        (read,) = bam.fetch()
    assert read.reference_end == 40


def test_write_random_bam_should_write_the_requested_number_of_reads(
    tmp_path, pysam, two_contig_header
):
    """Test that the record count is the count that was asked for.

    Given:
        A request for twelve randomly placed reads.
    When:
        The builder writes them.
    Then:
        It should write exactly that many records and return the path.
    """

    # Arrange
    path = str(tmp_path / "random.bam")

    # Act
    written = builders.write_random_bam(path, two_contig_header, TWO_CONTIGS, n_reads=12)

    # Assert
    assert written == path
    assert len(record_placements(path, pysam)) == 12


def test_write_random_bam_should_split_the_reads_evenly_between_strands(
    tmp_path, pysam, two_contig_header
):
    """Test that half the reads come back on the reverse strand.

    Given:
        A request for an even number of randomly placed reads.
    When:
        The builder writes them.
    Then:
        It should put half on each strand, so a suite reading strand
        from the data sees both without arranging for either.
    """

    # Arrange
    path = str(tmp_path / "stranded.bam")

    # Act
    builders.write_random_bam(path, two_contig_header, TWO_CONTIGS, n_reads=12)

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        assert sum(read.is_reverse for read in bam.fetch()) == 6


def test_write_random_bam_should_place_reads_identically_for_one_seed(
    tmp_path, pysam, two_contig_header
):
    """Test that a seeded random BAM is reproducible.

    Given:
        Two renders of one geometry under the same seed.
    When:
        The placements are compared.
    Then:
        It should place every read identically, so a suite that
        depends on where a read landed can depend on the seed instead.
    """

    # Arrange
    first = str(tmp_path / "a.bam")
    second = str(tmp_path / "b.bam")

    # Act
    builders.write_random_bam(first, two_contig_header, TWO_CONTIGS, n_reads=12, seed=3)
    builders.write_random_bam(second, two_contig_header, TWO_CONTIGS, n_reads=12, seed=3)

    # Assert
    assert record_placements(first, pysam) == record_placements(second, pysam)


def test_write_random_bam_should_place_reads_differently_when_the_seed_differs(
    tmp_path, pysam, two_contig_header
):
    """Test that the seed is what decides the placement.

    Given:
        Two renders of one geometry under different seeds.
    When:
        The placements are compared.
    Then:
        It should place them differently, so the seed is a real knob
        rather than a parameter the builder ignores.
    """

    # Arrange
    first = str(tmp_path / "a.bam")
    second = str(tmp_path / "b.bam")

    # Act
    builders.write_random_bam(first, two_contig_header, TWO_CONTIGS, n_reads=12, seed=3)
    builders.write_random_bam(second, two_contig_header, TWO_CONTIGS, n_reads=12, seed=4)

    # Assert
    assert record_placements(first, pysam) != record_placements(second, pysam)


def test_write_random_bam_should_put_reads_on_every_contig(tmp_path, pysam):
    """Test that a contig split is neither empty nor uniform.

    Given:
        A three-contig geometry and forty reads.
    When:
        The builder places them.
    Then:
        It should put at least one read on every contig and more on the
        first than on any other, so a per-contig count is neither zero
        anywhere nor the same number everywhere.
    """

    # Arrange
    header = builders.alignment_header(THREE_CONTIGS)
    path = str(tmp_path / "spread.bam")

    # Act
    builders.write_random_bam(path, header, THREE_CONTIGS, n_reads=40)

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        counts = {chrom: 0 for chrom in THREE_CONTIGS}
        for read in bam.fetch():
            counts[read.reference_name] += 1
    first, *rest = [counts[chrom] for chrom in THREE_CONTIGS]
    assert all(count > 0 for count in counts.values())
    assert all(first > count for count in rest)


@settings(FILE_BACKED, max_examples=25)
@given(sizes=alignment_sizes)
def test_write_random_bam_should_place_every_read_inside_its_contig_for_any_geometry(
    tmp_path, pysam, sizes
):
    """Test that no read is placed where it could not end inside its contig.

    Given:
        Any contig mapping the builder accepts.
    When:
        The builder places reads across it.
    Then:
        It should start every read far enough from the end that a
        record of the read length still finishes inside the contig, so
        the bound the refusal protects holds across geometries rather
        than only at the two sizes an example can pin.
    """

    # Arrange
    header = builders.alignment_header(sizes)
    path = str(tmp_path / "bounds.bam")

    # Act
    builders.write_random_bam(path, header, sizes)

    # Assert
    for chrom, start in record_placements(path, pysam):
        assert 0 <= start <= sizes[chrom] - 20


@settings(FILE_BACKED, max_examples=25)
@given(sizes=alignment_sizes, extra=st.integers(min_value=0, max_value=20))
def test_write_random_bam_should_put_reads_on_every_contig_for_any_geometry(
    tmp_path, pysam, sizes, extra
):
    """Test that contig coverage holds for any geometry.

    Given:
        Any contig mapping, and any read count at or above the contig
        count, i.e., the condition under which the builder promises it.
    When:
        The builder places them.
    Then:
        It should put at least one read on every contig, so no caller
        asking for more reads than contigs is handed an empty one.
    """

    # Arrange
    header = builders.alignment_header(sizes)
    path = str(tmp_path / "property.bam")

    # Act
    builders.write_random_bam(path, header, sizes, n_reads=len(sizes) + extra)

    # Assert
    assert {chrom for chrom, _ in record_placements(path, pysam)} == set(sizes)


@settings(FILE_BACKED, max_examples=25)
@given(sizes=alignment_sizes, extra=st.integers(min_value=1, max_value=20))
def test_write_random_bam_should_skew_toward_the_first_contig_for_any_geometry(
    tmp_path, pysam, sizes, extra
):
    """Test that the contig skew holds wherever it is promised.

    Given:
        Any contig mapping, and any read count above the contig count,
        i.e., the condition under which the builder promises a skew.
    When:
        The builder places them.
    Then:
        It should put more reads on the first contig than on any other,
        so a suite that splits work per contig sees an uneven split
        rather than one it only assumes is uneven.
    """

    # Arrange
    header = builders.alignment_header(sizes)
    path = str(tmp_path / "skew.bam")
    first = next(iter(sizes))

    # Act
    builders.write_random_bam(path, header, sizes, n_reads=len(sizes) + extra)

    # Assert
    counts = Counter(chrom for chrom, _ in record_placements(path, pysam))
    assert all(counts[first] > counts[chrom] for chrom in sizes if chrom != first)


def test_write_random_bam_should_size_its_reads_from_the_read_length(
    tmp_path, pysam, two_contig_header
):
    """Test that the read length reaches every record.

    Given:
        An explicit read length.
    When:
        The builder writes the reads.
    Then:
        It should give every record a CIGAR and a query of that length.
    """

    # Arrange
    path = str(tmp_path / "sized.bam")

    # Act
    builders.write_random_bam(
        path, two_contig_header, TWO_CONTIGS, n_reads=8, read_length=35
    )

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        reads = list(bam.fetch())
    assert [read.cigarstring for read in reads] == ["35M"] * 8
    assert [read.query_length for read in reads] == [35] * 8


def test_track_rows_frame_should_answer_the_three_columns_a_track_declares():
    """Test the schema of a built in-memory track.

    Given:
        Rows over two contigs.
    When:
        The builder turns them into a frame.
    Then:
        It should answer a polars frame with exactly ``chrom``, ``pos``
        and ``value``, as text, 64-bit integers and 64-bit floats.
    """

    # Arrange
    polars = pytest.importorskip("polars")

    # Act
    frame = builders.track_rows_frame([("chr1", 5, 2.0), ("chr2", 7, 3.0)])

    # Assert
    assert isinstance(frame, polars.DataFrame)
    assert dict(frame.schema) == {
        "chrom": polars.Utf8,
        "pos": polars.Int64,
        "value": polars.Float64,
    }


def test_track_rows_frame_should_keep_the_rows_in_the_order_given():
    """Test that the builder neither sorts nor groups.

    Given:
        Rows whose contigs interleave and whose positions descend.
    When:
        The builder turns them into a frame.
    Then:
        It should hold the rows in the order given.
    """

    # Arrange
    pytest.importorskip("polars")
    rows = [("chr2", 50, 1.0), ("chr1", 30, 2.0), ("chr2", 10, 3.0), ("chr1", 20, 4.0)]

    # Act
    frame = builders.track_rows_frame(rows)

    # Assert
    assert frame.rows() == rows


def test_track_rows_frame_should_convert_each_field_to_its_column_type():
    """Test the rows the builder accepts beyond the literal ones.

    Given:
        A row whose contig is an integer, whose position is a string
        of digits and whose value is an integer.
    When:
        The builder turns it into a frame.
    Then:
        It should hold the contig as text, the position as an integer
        and the value as a float.
    """

    # Arrange
    pytest.importorskip("polars")

    # Act
    frame = builders.track_rows_frame([(1, "12", 3)])

    # Assert
    assert frame.rows() == [("1", 12, 3.0)]
    assert isinstance(frame["value"][0], float)


def test_track_rows_frame_should_answer_an_empty_track_for_no_rows():
    """Test the frame built from nothing.

    Given:
        No rows.
    When:
        The builder turns them into a frame.
    Then:
        It should answer a frame of no rows that still carries the
        three columns, so a reader can select them.
    """

    # Arrange
    pytest.importorskip("polars")

    # Act
    frame = builders.track_rows_frame([])

    # Assert
    assert frame.height == 0
    assert frame.columns == ["chrom", "pos", "value"]


@settings(max_examples=50)
@given(
    rows=st.lists(
        st.tuples(
            contig_names,
            st.integers(min_value=0, max_value=2**40),
            st.integers(min_value=-(2**20), max_value=2**20).map(float),
        ),
        max_size=12,
    )
)
def test_track_rows_frame_should_hold_exactly_the_rows_it_was_given(rows):
    """Test the builder over arbitrary rows.

    Given:
        Any rows of a contig name, a position and a whole-number value,
        repeats allowed.
    When:
        The builder turns them into a frame.
    Then:
        The frame's rows should equal the rows given, in order.
    """

    # Arrange
    pytest.importorskip("polars")

    # Act
    frame = builders.track_rows_frame(rows)

    # Assert
    assert frame.rows() == rows


def test_scattered_rows_should_place_distinct_positions_on_every_contig():
    """Test the geometry of the scattered rows.

    Given:
        Two contigs of 1000 and 500 bases.
    When:
        Rows are scattered over them.
    Then:
        It should answer rows on both contigs, each position inside
        its contig and given once, sorted by contig and position, and
        no more of them on a contig than a third of its length.
    """

    # Act
    rows = builders.scattered_rows(TWO_CONTIGS, seed=3)

    # Assert
    placed = [(chrom, pos) for chrom, pos, _ in rows]
    assert placed == sorted(set(placed))
    for chrom, size in TWO_CONTIGS.items():
        positions = [pos for on, pos in placed if on == chrom]
        assert 0 < len(positions) <= size // 3
        assert 0 <= min(positions) and max(positions) < size


def test_scattered_rows_should_give_each_position_a_whole_value_from_one_to_four():
    """Test the values of the scattered rows.

    Given:
        Two contigs.
    When:
        Rows are scattered over them.
    Then:
        Every value should be a float equal to 1, 2, 3 or 4, and more
        than one of those should occur.
    """

    # Act
    rows = builders.scattered_rows(TWO_CONTIGS, seed=3)

    # Assert
    values = [value for _, _, value in rows]
    assert all(isinstance(value, float) for value in values)
    assert set(values) <= {1.0, 2.0, 3.0, 4.0}
    assert len(set(values)) > 1


def test_scattered_rows_should_answer_the_same_rows_for_the_same_seed():
    """Test that the scattering is a function of its seed.

    Given:
        Two contigs and two seeds.
    When:
        Rows are scattered twice under one seed and once under the
        other.
    Then:
        The two under one seed should be equal, and differ from the
        third.
    """

    # Act
    first = builders.scattered_rows(TWO_CONTIGS, seed=3)
    again = builders.scattered_rows(TWO_CONTIGS, seed=3)
    other = builders.scattered_rows(TWO_CONTIGS, seed=4)

    # Assert
    assert first == again
    assert first != other


def test_write_track_should_write_one_parquet_file_per_contig(tmp_path):
    """Test the layout a built track has.

    Given:
        Rows over two contigs.
    When:
        The builder writes them as a track.
    Then:
        It should produce a directory holding one Parquet file per
        contig and nothing else, with the completeness record beside
        the directory rather than inside it.

    Holding Parquet and nothing else is the invariant the format rests
    on: polars refuses to scan a directory of mixed extensions, and so
    does anything else pointed at the dataset.
    """

    # Arrange
    pytest.importorskip("polars")
    root = str(tmp_path / "sample.track.parquet")

    # Act
    builders.write_track(root, [("chr1", 10, 1.0), ("chr2", 5, 2.0)])

    # Assert
    assert sorted(os.listdir(root)) == ["chr1.parquet", "chr2.parquet"]
    assert os.path.exists(root + ".manifest.json")


def test_write_track_should_sort_rows_it_was_given_out_of_order(tmp_path):
    """Test that a hand-written row list needs no ordering.

    Given:
        Rows whose contigs interleave and whose positions descend.
    When:
        The builder writes them as a track.
    Then:
        It should still write one file per contig, with each file's
        positions ascending, so a test may list the rows it means in
        the order it means them.

    An ungrouped, unsorted row list is the one thing a hand-written
    literal reliably is, and the format requires both.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    root = str(tmp_path / "jumbled.track.parquet")
    rows = [("chr2", 50, 1.0), ("chr1", 30, 2.0), ("chr2", 10, 3.0), ("chr1", 20, 4.0)]

    # Act
    builders.write_track(root, rows)

    # Assert
    assert sorted(os.listdir(root)) == ["chr1.parquet", "chr2.parquet"]
    for name, expected in (("chr1", [20, 30]), ("chr2", [10, 50])):
        frame = polars.read_parquet(os.path.join(root, name + ".parquet"))
        assert frame.get_column("pos").to_list() == expected


def test_write_track_should_produce_a_track_the_package_verifies(tmp_path):
    """Test that a built track is the artifact a pileup commits.

    Given:
        Rows written as a track by the builder.
    When:
        The package is asked to verify it complete.
    Then:
        It should pass, and report the contigs the rows named, so every
        later assertion about reading a track is an assertion about the
        same artifact the pileup produces.

    The builder goes through the package's own writer for exactly this
    reason: a suite that built the layout itself could drift from the
    format and still agree with itself.
    """

    # Arrange
    pytest.importorskip("polars")
    root = str(tmp_path / "verified.track.parquet")

    # Act
    builders.write_track(root, [("chr1", 10, 1.0), ("chr1", 20, 2.0), ("chr2", 5, 3.0)])

    # Assert
    assert quadrat.require_complete(root) == root
    assert quadrat.track_contigs(root) == {"chr1", "chr2"}
    assert quadrat.read_manifest(root)["rows"] == 3


def test_write_track_should_record_the_parameters_it_was_given(tmp_path):
    """Test that a track carries the arguments it was built with.

    Given:
        A parameters mapping passed to the builder.
    When:
        The written track's record is read back.
    Then:
        It should carry that mapping, so a test can assert on what a
        track claims about its own provenance.
    """

    # Arrange
    pytest.importorskip("polars")
    root = str(tmp_path / "noted.track.parquet")

    # Act
    builders.write_track(root, [("chr1", 10, 1.0)], parameters={"shift": 4})

    # Assert
    assert quadrat.read_manifest(root)["parameters"] == {"shift": 4}


def test_write_track_should_write_a_track_that_records_nothing(tmp_path):
    """Test the natural spelling of a channel with no rows at all.

    Given:
        An empty row list.
    When:
        A track is written from it.
    Then:
        It should verify complete and report no contigs, because a
        channel that recorded nothing is a channel and not an error --
        and it is what a reader answers a row of zeros for.
    """

    # Arrange
    path = str(tmp_path / "empty.track.parquet")

    # Act
    written = builders.write_track(path, [])

    # Assert
    assert quadrat.require_complete(written) == written
    assert quadrat.track_contigs(written) == set()


def test_write_track_should_write_a_track_of_one_row(tmp_path):
    """Test the smallest track that holds anything.

    Given:
        A single row.
    When:
        A track is written from it.
    Then:
        It should hold one contig whose record counts one row, so the
        boundary between nothing and something is a track either way.
    """

    # Arrange
    path = str(tmp_path / "one.track.parquet")

    # Act
    written = builders.write_track(path, [("chr1", 5, 2.0)])

    # Assert
    assert quadrat.track_contigs(written) == {"chr1"}
    assert quadrat.read_manifest(written)["rows"] == 1


def test_write_track_should_keep_a_position_given_twice(tmp_path):
    """Test what the builder does with a duplicated position.

    Given:
        Two rows at one contig and position, with different values.
    When:
        A track is written from them.
    Then:
        Both should be recorded, because the builder writes the rows a
        test means rather than the rows the format promises -- which is
        what lets a test hand the readers a malformed track on purpose.
    """

    # Arrange
    path = str(tmp_path / "dup.track.parquet")

    # Act
    written = builders.write_track(path, [("chr1", 5, 2.0), ("chr1", 5, 3.0)])

    # Assert
    assert quadrat.read_manifest(written)["rows"] == 2


def test_write_track_should_sort_within_a_contig_as_well_as_across_them(tmp_path):
    """Test the order each contig's own file comes out in.

    Given:
        Rows already grouped by contig but descending by position
        inside each group.
    When:
        A track is written from them.
    Then:
        Each file's positions should ascend, because the format
        promises a position-sorted file and a grouped-but-unsorted
        frame is the shape a sort across contigs alone would pass
        through.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    path = str(tmp_path / "desc.track.parquet")

    # Act
    written = builders.write_track(
        path, [("chr1", 12, 3.0), ("chr1", 8, 1.0), ("chr2", 9, 4.0), ("chr2", 5, 2.0)]
    )

    # Assert
    for chrom in ("chr1", "chr2"):
        rows = polars.read_parquet(quadrat.track_chrom_path(written, chrom))
        assert rows["pos"].to_list() == sorted(rows["pos"].to_list())


def test_write_track_should_round_trip_a_contig_name_needing_quoting(tmp_path):
    """Test a contig whose name carries characters a filesystem reserves.

    Given:
        A contig named with a slash, a space and a percent sign.
    When:
        A track is written and its contigs read back.
    Then:
        It should come back under the name it was given, because the
        per-contig filename is percent-quoted and a test corpus has to
        be able to provoke that.
    """

    # Arrange
    name = "chr1/a b%c"
    path = str(tmp_path / "quoted.track.parquet")

    # Act
    written = builders.write_track(path, [(name, 5, 2.0)])

    # Assert
    assert quadrat.track_contigs(written) == {name}


def test_write_track_should_write_a_track_whose_path_holds_glob_characters(tmp_path):
    """Test a track path a glob would read as a pattern.

    Given:
        A track path carrying a bracket, a star and a question mark.
    When:
        A track is written there and verified.
    Then:
        It should write, verify and list its contigs, because the
        commit sweeps its own siblings by glob and a caller's path is
        not a pattern.
    """

    # Arrange
    path = str(tmp_path / "t[1]*?.track.parquet")

    # Act
    written = builders.write_track(path, [("chr1", 5, 2.0)])

    # Assert
    assert quadrat.require_complete(written) == written
    assert quadrat.track_contigs(written) == {"chr1"}


def test_write_flat_track_should_write_the_rows_in_the_order_given(tmp_path):
    """Test the layout and row order of a single-file track.

    Given:
        Rows over two contigs, interleaved and descending by position.
    When:
        The builder writes them as a single-file track.
    Then:
        It should write one Parquet file with exactly the ``chrom``,
        ``pos`` and ``value`` columns, holding the rows in the order
        given, because this layout exists to hand a reader rows it was
        never promised are sorted.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    path = str(tmp_path / "flat.parquet")
    rows = [("chr2", 50, 1.0), ("chr1", 30, 2.0), ("chr2", 10, 3.0), ("chr1", 20, 4.0)]

    # Act
    written = builders.write_flat_track(path, rows)

    # Assert
    assert written == path
    assert os.listdir(tmp_path) == ["flat.parquet"]
    table = polars.read_parquet(written)
    assert table.columns == ["chrom", "pos", "value"]
    assert table.rows() == rows


def test_write_flat_track_should_write_a_track_the_package_reads(tmp_path):
    """Test that a single-file track is one a reader accepts.

    Given:
        Rows over two contigs written as a single-file track.
    When:
        The package reads it as a whole track.
    Then:
        It should answer every row, because the builder stands in for a
        table this package did not write, and a stand-in the readers
        refuse would test nothing about them.
    """

    # Arrange
    pytest.importorskip("polars")
    rows = [("chr1", 12, 3.0), ("chr2", 5, 2.0), ("chr1", 10, 1.0)]
    written = builders.write_flat_track(str(tmp_path / "read.parquet"), rows)

    # Act
    table = stores.track_frame(written).collect()

    # Assert
    assert sorted(table.select("chrom", "pos", "value").rows()) == sorted(rows)
