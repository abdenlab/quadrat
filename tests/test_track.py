"""Tests for turning alignments into tracks.

The refusal of a second concurrent pileup is pinned through the call
a caller makes rather than through the lock behind it: two threads both
call `pileup_alignments` behind a barrier and exactly one is refused.
The assertion is on the count and never on which thread won.

The two BAMs come from `tests.builders` directly rather than from
``corpus_factory``, because their geometry is the point -- secondary and
duplicate records, a placed-unmapped one, a reference skip, a soft clip
and a contig the reference does not have -- and the factory places
reads by a seeded draw.

Splitting the genome into contig batches is a memory optimisation and
must be nothing else, so the tests for it assert equality with the
ungrouped form rather than anything about the peak: a byte comparison
is checkable in CI, where a memory figure is not.
"""

import dataclasses
import gc
import logging
import os
import pathlib
import random
import tempfile
import threading
import warnings
import weakref

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quadrat import Aggregate, EngineOptionWarning, stores, tables
from quadrat import pileup as pileup_pkg
from quadrat.pileup import track
from quadrat.pileup.aggregate import DEFAULT_EXCLUDE_FLAGS
from tests import (
    FILE_BACKED,
    SPLICED_CONTIGS,
    SPLICED_HEADER,
    TWO_CONTIGS,
    bound,
    builders,
    coverage_from_track,
    digest_track,
    oracle_depth,
    oracle_pileup,
    strand_channels,
)

START_SHIFT, END_SHIFT = 4, -4

# A track of coverage deltas may hold a span's closing event at its
# contig's length, so its rows are densified one base past each contig.
DELTA_SIZES = {chrom: size + 1 for chrom, size in TWO_CONTIGS.items()}

# The two aggregates most tests pile up under: every default but the
# strands pooled, and that with ATAC's Tn5 shift.
UNSTRANDED = Aggregate(unstranded=True)
ATAC = Aggregate(start_shift=START_SHIFT, end_shift=END_SHIFT, unstranded=True)


class EnginePanic(BaseException):
    """Stand-in for the panic polars-bio raises out of Rust.

    The real type is `PanicException`, which pyo3 creates at runtime and
    exports from no importable path -- ``from polars_bio.polars_bio
    import PanicException`` fails. What the suite needs from it is the
    one property that made it a defect: it derives from `BaseException`
    rather than from `Exception`, so an ``except Exception`` does not
    see it.
    """


@pytest.fixture(scope="module")
def reference_fasta(corpus_factory):
    """An indexed reference over two contigs."""

    pytest.importorskip("polars_bio")
    return corpus_factory("track", TWO_CONTIGS, index_fasta=True).fasta


@pytest.fixture(scope="module")
def bam_header():
    """A header naming a contig the reference does not have."""

    # `chrZ` is how a header-versus-reference mismatch reaches a reader.
    return builders.alignment_header(dict(TWO_CONTIGS, chrZ=800))


@pytest.fixture(scope="module")
def bam_pair(tmp_path_factory, bam_header):
    """Two alignments covering every record shape the aggregate must judge."""

    root = tmp_path_factory.mktemp("track-pair")
    h = bam_header
    first = [
        builders.aligned_read(h, "f1", "chr1", 100, "20M", 0),
        builders.aligned_read(h, "f1_secondary", "chr1", 100, "20M", 256),
        builders.aligned_read(h, "f1_duplicate", "chr1", 100, "20M", 1024),
        builders.aligned_read(h, "r1", "chr1", 150, "20M", 16),
        builders.aligned_read(h, "f_del", "chr1", 200, "10M5D10M", 0),
        builders.aligned_read(h, "r_clip", "chr1", 300, "5S15M", 16),
        builders.aligned_read(h, "r_skip", "chr1", 400, "10M100N10M", 16),
        builders.aligned_read(h, "placed_unmapped", "chr1", 600, "20M", 4),
        builders.aligned_read(h, "unmapped", None, None, None, 4),
        builders.aligned_read(h, "on_chrZ", "chrZ", 50, "20M", 0),
        builders.aligned_read(h, "c2_f", "chr2", 10, "20M", 0),
        builders.aligned_read(h, "c2_r", "chr2", 480, "20M", 16),
    ]
    second = [
        builders.aligned_read(h, "f2", "chr1", 700, "20M", 0),
        builders.aligned_read(h, "r2", "chr1", 100, "20M", 16),
        builders.aligned_read(h, "c2_f_again", "chr2", 10, "20M", 0),
    ]
    return [
        builders.write_bam(str(root / "first.bam"), first, h),
        builders.write_bam(str(root / "second.bam"), second, h),
    ]


@pytest.fixture(scope="module")
def bam_edges(tmp_path_factory, bam_header):
    """Reads whose shifted positions fall off either end of a contig."""

    root = tmp_path_factory.mktemp("track-edges")
    h = bam_header
    reads = [
        builders.aligned_read(h, "rev_at_origin", "chr1", 0, "3M", 16),
        builders.aligned_read(h, "fwd_at_end", "chr1", 998, "2M", 0),
        builders.aligned_read(h, "fwd_inside", "chr1", 990, "2M", 0),
    ]
    return builders.write_bam(str(root / "edges.bam"), reads, h)


@pytest.fixture
def shifted_pair(corpus_factory):
    """One forward and one reverse read at positions a test can name."""

    pytest.importorskip("polars_bio")
    return corpus_factory(
        "shifted",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        reads=[
            ("fwd", "chr1", 100, "20M", 0),
            ("rev", "chr1", 300, "20M", 16),
        ],
    )


@pytest.fixture(scope="module")
def flagged(tmp_path_factory, bam_header):
    """One record per filter clause, each at a position of its own on ``chr1``.

    100 is a plain primary record; 110, 120, 130 and 140 carry the
    secondary, QC-failed, duplicate and supplementary flags; 150 is
    placed but unmapped; 160 and 170 are the first and second mates of
    one pair, 250 bases apart by template length; and 180 is a primary
    record of mapping quality 5, the rest being 30.
    """

    h = bam_header
    reads = [
        builders.aligned_read(h, "plain", "chr1", 100, "20M", 0),
        builders.aligned_read(h, "secondary", "chr1", 110, "20M", 0x100),
        builders.aligned_read(h, "qc_failed", "chr1", 120, "20M", 0x200),
        builders.aligned_read(h, "duplicate", "chr1", 130, "20M", 0x400),
        builders.aligned_read(h, "supplementary", "chr1", 140, "20M", 0x800),
        builders.aligned_read(h, "unmapped", "chr1", 150, "20M", 0x4),
        builders.aligned_read(h, "pair", "chr1", 160, "20M", 0x1 | 0x40),
        builders.aligned_read(h, "pair", "chr1", 170, "20M", 0x1 | 0x80),
        builders.aligned_read(h, "low_quality", "chr1", 180, "20M", 0, mapq=5),
    ]
    reads[6].template_length = 250
    reads[7].template_length = -250
    root = tmp_path_factory.mktemp("track-flagged")
    return builders.write_bam(str(root / "flagged.bam"), reads, h)


def mate_pair(header, name, chrom, left, length, positive, flags=(0, 0), mapq=(30, 30)):
    """Return the two records of one pair spanning ``length`` bases from ``left``.

    The forward mate's 5' end is ``left`` and the reverse mate's is the
    fragment's last base, each record 20 bases long, so the template
    length is the distance between the two 5' ends, inclusive.
    ``positive`` names the mate carrying the positive template length,
    ``"forward"`` or ``"reverse"``, whatever their order by position.
    """

    right = left + length - 1
    forward = builders.aligned_read(
        header, name, chrom, left, "20M", 0x1 | 0x40 | flags[0], mapq=mapq[0]
    )
    reverse = builders.aligned_read(
        header,
        name,
        chrom,
        right - 19,
        "20M",
        0x1 | 0x80 | 0x10 | flags[1],
        mapq=mapq[1],
    )
    sign = 1 if positive == "forward" else -1
    forward.template_length = sign * length
    reverse.template_length = -sign * length
    return [forward, reverse]


@pytest.fixture(scope="module")
def mixed_pairs(tmp_path_factory):
    """Pairs and unpaired records of every shape the filter and the rules judge.

    Sixty pairs over both contigs, of seeded fragment lengths from 20
    to 199, with the positive template length on a seeded mate, a
    seeded flag from the default exclusion mask or none on each mate,
    and a seeded mapping quality; and ten unpaired records with no
    template length. Every record is unspliced, so its span is the
    bases it aligns.
    """

    rng = random.Random(11)
    header = builders.alignment_header(TWO_CONTIGS)
    extras = [0, 0, 0, 0, 0x100, 0x200, 0x400, 0x800]
    reads = []
    for index in range(60):
        chrom = "chr1" if index % 3 else "chr2"
        length = rng.randrange(20, 200)
        reads += mate_pair(
            header,
            "pair{}".format(index),
            chrom,
            rng.randrange(0, TWO_CONTIGS[chrom] - length + 1),
            length,
            rng.choice(["forward", "reverse"]),
            flags=(rng.choice(extras), rng.choice(extras)),
            mapq=(rng.choice([0, 5, 20, 60]), rng.choice([0, 5, 20, 60])),
        )
    for index in range(10):
        reads.append(
            builders.aligned_read(
                header,
                "single{}".format(index),
                "chr1",
                rng.randrange(0, 980),
                "20M",
                rng.choice([0, 0x10]),
            )
        )
    root = tmp_path_factory.mktemp("track-mixed")
    return builders.write_bam(str(root / "mixed.bam"), reads, header)


@pytest.fixture(scope="module")
def long_reference(corpus_factory):
    """An indexed reference with one 2 kb contig, room for a 300-base fragment at 1000."""

    pytest.importorskip("polars_bio")
    return corpus_factory("track-long", {"chr1": 2000}, index_fasta=True).fasta


def samtools_depth(bam_paths, chrom_sizes, pysam):
    """Return ``samtools depth -a -J`` per contig, summed over the alignments.

    Run by pysam over every file at once, with each base past its
    contig's length in ``chrom_sizes`` dropped. ``pysam`` is passed in,
    as it is to `tests.record_placements`.
    """

    depth = {c: numpy.zeros(n) for c, n in chrom_sizes.items()}
    for line in pysam.depth("-a", "-J", *bam_paths).splitlines():
        chrom, pos, *counts = line.split("\t")
        if int(pos) <= chrom_sizes[chrom]:
            depth[chrom][int(pos) - 1] = sum(int(count) for count in counts)
    return depth


def dense_from_parquet(path, chrom_sizes):
    """Return a track as a dense array per contig, and the frame itself."""

    polars = pytest.importorskip("polars")
    frame = polars.read_parquet(path)
    dense = {c: numpy.zeros(n, dtype=numpy.float64) for c, n in chrom_sizes.items()}
    for chrom, pos, value in frame.iter_rows():
        dense[chrom][pos] = value
    return dense, frame


def positions(path):
    """The positions a single-contig track holds, in order."""

    polars = pytest.importorskip("polars")
    return polars.read_parquet(path)["pos"].to_list()


@pytest.fixture(scope="module")
def track_paths(tmp_path_factory, bam_pair, reference_fasta):
    """The tables one unstranded pileup of the pair writes."""

    root = tmp_path_factory.mktemp("track-out")
    return pileup_pkg.pileup_alignments(
        bam_pair,
        reference_fasta,
        str(root / "pair"),
        aggregate=ATAC,
    )


def test_pileup_alignments_should_accept_path_objects(
    tmp_path, bam_pair, reference_fasta
):
    """Test the path type every sibling in this surface already takes.

    Given:
        The alignments, the reference and the output name all as
        `pathlib.Path` objects.
    When:
        They are piled up.
    Then:
        It should write the same tables a string call writes.

    `pathlib.Path` is admitted by `chrom_lengths`,
    `register_alignments`, `bam_contigs` and the three readers in
    `quadrat.tables`, all of them through one normalizer and each with
    a test pinning it -- and the package's headline entry point was the
    one that refused it. Both
    failures were tracebacks from the middle of the call, raised after
    the alignment tables had been registered into the process-global
    session: ``", ".join(alignments)`` from inside a logging call, and
    ``name + TRACK_SUFFIX`` from the path builder.
    """

    # Arrange
    name = tmp_path / "paths"

    # Act
    written = pileup_pkg.pileup_alignments(
        [pathlib.Path(path) for path in bam_pair],
        pathlib.Path(reference_fasta),
        name,
        aggregate=UNSTRANDED,
    )

    # Assert
    assert written == [str(name) + tables.TRACK_SUFFIX]
    assert os.path.isdir(written[0])


def test_pileup_alignments_should_accept_bytes_paths(
    tmp_path, bam_pair, reference_fasta
):
    """Test the path type this entry point admitted and never honoured.

    Given:
        The alignments, the reference and the output name all as
        ``bytes`` paths.
    When:
        They are piled up.
    Then:
        It should write the same tables a string call writes.

    The boundary probe already counted ``bytes`` as one path -- it is
    what let a bare ``bytes`` argument be wrapped as a single-alignment
    pool -- and nothing downstream converted it: the value survived to
    `register_alignments`, which refused it with a message about file
    format, *after* the process-global registration window had opened,
    and inside a worker under ``pileup_many``. No test exercised a
    ``bytes`` path anywhere in the suite.
    """

    # Arrange
    name = tmp_path / "bytes"

    # Act
    written = pileup_pkg.pileup_alignments(
        [os.fsencode(path) for path in bam_pair],
        os.fsencode(str(reference_fasta)),
        os.fsencode(str(name)),
        aggregate=UNSTRANDED,
    )

    # Assert
    assert written == [str(name) + tables.TRACK_SUFFIX]
    assert os.path.isdir(written[0])


def test_pileup_alignments_should_refuse_a_name_that_is_not_a_path(
    bam_pair, reference_fasta, mocker
):
    """Test that a wrong-typed name is refused before anything registers.

    Given:
        An integer where the output name belongs, with the registration
        watched.
    When:
        A pileup is attempted.
    Then:
        It should raise `TypeError` naming the parameter, and
        `register_alignments` should never have been called.

    The registration is what this is about rather than the exception
    class. ``name`` used to reach `track_paths` and raise there, which
    is also a `TypeError` naming ``name`` -- so an assertion on the
    exception alone passes either way. What differs is where: the path
    builder runs inside the session window, after the alignment tables
    are bound, which is the failure the boundary normalization exists
    to pre-empt and the reason `_normalize` normalizes ``name`` for the
    fan-out rather than leaving it to a worker.
    """

    # Arrange
    registering = mocker.patch.object(
        track, "register_alignments", wraps=track.register_alignments
    )

    # Act & assert
    with pytest.raises(TypeError, match="name"):
        pileup_pkg.pileup_alignments(bam_pair, reference_fasta, 42, aggregate=UNSTRANDED)
    registering.assert_not_called()


def test_pileup_alignments_should_accept_one_alignment_as_a_bare_path(
    tmp_path, bam_pair, reference_fasta
):
    """Test the single-alignment call written without a list.

    Given:
        One alignment given as a bare path rather than as a list of
        one.
    When:
        It is piled up.
    Then:
        It should write the table that one alignment produces.

    The plural name invites the mistake, and the bare string passed the
    emptiness check and was then iterated character-wise -- so the
    failure named the single character the engine was asked to register
    rather than the shape of the argument.
    """

    # Arrange
    name = str(tmp_path / "bare")

    # Act
    written = pileup_pkg.pileup_alignments(
        bam_pair[0], reference_fasta, name, aggregate=UNSTRANDED
    )

    # Assert
    assert written == [name + tables.TRACK_SUFFIX]
    assert tables.track_contigs(written[0]) <= set(TWO_CONTIGS)


def test_pileup_alignments_should_count_the_shifted_five_prime_end_of_each_kept_record(
    track_paths, bam_pair, pysam
):
    """Test the position every record contributes under the default filter.

    Given:
        Two BAMs carrying secondary, duplicate, unmapped and off-contig
        reads.
    When:
        They are piled up with the ATAC +4/-4 shift, unstranded.
    Then:
        It should match a pysam re-implementation of bam2bw's loop at
        every position once the same filter is applied, pooling all
        files and skipping the secondary, duplicate, unmapped and
        unknown-contig reads.
    """

    # Arrange
    expected = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, ATAC)

    # Act
    dense, frame = dense_from_parquet(track_paths[0], TWO_CONTIGS)

    # Assert
    assert track_paths == [track_paths[0]]
    assert track_paths[0].endswith(tables.TRACK_SUFFIX)
    assert frame.schema["pos"].is_integer()
    assert frame.schema["value"].is_float()
    assert frame["chrom"].to_list() == sorted(frame["chrom"].to_list())
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(dense[chrom], expected[chrom])

    # Spot checks that pin the semantics independently of the oracle.
    assert dense["chr1"][100 + START_SHIFT] == 1  # secondary and duplicate skipped
    assert dense["chr1"][170 - 1 + END_SHIFT] == 1  # reverse read cuts at end - 1
    assert dense["chr1"][520 - 1 + END_SHIFT] == 1  # 10M100N10M spans the skip
    assert dense["chr1"][315 - 1 + END_SHIFT] == 1  # 5S15M: clip is not reference
    assert dense["chr2"][10 + START_SHIFT] == 2  # pooled across files
    assert dense["chr1"][600 + START_SHIFT] == 0  # placed unmapped skipped
    assert frame["value"].sum() == 10  # 7 records in file one + 3 in file two


def test_pileup_alignments_should_apply_the_scale_factor(
    tmp_path, bam_pair, reference_fasta, pysam
):
    """Test the one multiplication the aggregate applies.

    Given:
        A scale factor of one half.
    When:
        The same BAMs are piled up.
    Then:
        It should halve every value the oracle counts under the same
        aggregate.
    """

    # Arrange
    aggregate = Aggregate(
        start_shift=START_SHIFT,
        end_shift=END_SHIFT,
        unstranded=True,
        scale_factor=0.5,
    )

    # Act
    paths = pileup_pkg.pileup_alignments(
        bam_pair, reference_fasta, str(tmp_path / "scaled"), aggregate=aggregate
    )

    # Assert
    expected = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, aggregate)
    dense, _ = dense_from_parquet(paths[0], TWO_CONTIGS)
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(dense[chrom], expected[chrom])


def test_pileup_alignments_should_write_the_columns_at_the_formats_dtypes(
    track_paths,
):
    """Test the dtypes a written track carries.

    Given:
        The pair piled up unstranded.
    When:
        A contig's file is read.
    Then:
        It should hold ``chrom`` as ``Utf8``, ``pos`` as ``Int64`` and
        ``value`` as ``Float64``, as the track format declares.
    """

    # Arrange
    polars = pytest.importorskip("polars")

    # Act
    schema = polars.read_parquet(os.path.join(track_paths[0], "chr1.parquet")).schema

    # Assert
    assert dict(schema) == {
        "chrom": polars.Utf8,
        "pos": polars.Int64,
        "value": polars.Float64,
    }


def test_pileup_alignments_should_split_the_strands(
    tmp_path, bam_pair, reference_fasta, pysam
):
    """Test the two channels a stranded pileup writes.

    Given:
        Stranded output.
    When:
        The BAMs are piled up.
    Then:
        It should put forward positions in the plus table and reverse
        ones in the minus table, and the two should sum to the
        unstranded pileup.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        bam_pair,
        reference_fasta,
        str(tmp_path / "stranded"),
        aggregate=Aggregate(start_shift=START_SHIFT, end_shift=END_SHIFT),
    )

    # Assert
    assert paths == [
        str(tmp_path / ("stranded.+" + tables.TRACK_SUFFIX)),
        str(tmp_path / ("stranded.-" + tables.TRACK_SUFFIX)),
    ]

    plus, _ = dense_from_parquet(paths[0], TWO_CONTIGS)
    minus, _ = dense_from_parquet(paths[1], TWO_CONTIGS)
    both = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, ATAC)
    assert plus["chr1"][100 + START_SHIFT] == 1 and minus["chr1"][104] == 0
    assert minus["chr1"][170 - 1 + END_SHIFT] == 1 and plus["chr1"][165] == 0
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(plus[chrom] + minus[chrom], both[chrom])


def test_pileup_alignments_should_drop_sites_outside_their_contig(
    tmp_path, bam_edges, reference_fasta
):
    """Test the bound the aggregate puts on a shifted position.

    Given:
        Reads whose shifted positions fall below zero or past the
        contig's end.
    When:
        They are piled up.
    Then:
        It should keep only the site that is in range.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        [bam_edges],
        reference_fasta,
        str(tmp_path / "edges"),
        aggregate=ATAC,
    )

    # Assert
    dense, frame = dense_from_parquet(paths[0], TWO_CONTIGS)
    assert frame.height == 1
    assert dense["chr1"][990 + START_SHIFT] == 1


def test_pileup_alignments_should_shift_only_the_forward_site_by_start_shift(
    tmp_path, shifted_pair
):
    """Test that the forward shift moves the forward site alone.

    Given:
        One forward and one reverse read at known positions, and a
        non-zero ``start_shift`` with ``end_shift`` left at its default.
    When:
        The alignments are piled up unstranded.
    Then:
        It should move the forward site by that shift and leave the
        reverse site where no shift puts it.

    Every call in the suite this came from passed the two together, so a
    defect that applied one shift to both strands, or swapped them,
    would have passed.
    """

    # Arrange
    unshifted = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "plain"),
        aggregate=UNSTRANDED,
    )

    # Act
    shifted = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "fwd"),
        aggregate=Aggregate(unstranded=True, start_shift=7),
    )

    # Assert
    assert positions(os.path.join(unshifted[0], "chr1.parquet")) == [100, 319]
    assert positions(os.path.join(shifted[0], "chr1.parquet")) == [107, 319]


def test_pileup_alignments_should_shift_only_the_reverse_site_by_end_shift(
    tmp_path, shifted_pair
):
    """Test that the reverse shift moves the reverse site alone.

    Given:
        The same reads, and a non-zero ``end_shift`` with ``start_shift``
        left at its default.
    When:
        The alignments are piled up unstranded.
    Then:
        It should move the reverse site by that shift and leave the
        forward site unmoved.
    """

    # Act
    shifted = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "rev"),
        aggregate=Aggregate(unstranded=True, end_shift=-9),
    )

    # Assert
    assert positions(os.path.join(shifted[0], "chr1.parquet")) == [100, 310]


@pytest.mark.parametrize(
    "aggregate, kept",
    [
        (Aggregate(), [100, 160, 170, 180]),
        (Aggregate(exclude_flags=0), [100, 110, 120, 130, 140, 160, 170, 180]),
        (Aggregate(min_mapq=10), [100, 160, 170]),
        (Aggregate(require_flags=0x40), [160]),
        (Aggregate(min_template_length=1), [160, 170]),
        (Aggregate(max_template_length=100), [100, 180]),
    ],
    ids=[
        "default",
        "no_mask",
        "mapq_floor",
        "first_mate",
        "template_floor",
        "template_ceiling",
    ],
)
def test_pileup_alignments_should_count_only_the_records_the_filter_keeps(
    tmp_path, flagged, reference_fasta, aggregate, kept
):
    """Test each clause of the record filter on a record of its own.

    Given:
        One record carrying each flag the default excludes, a placed
        unmapped one, the two mates of a pair and one of low mapping
        quality, each at a position of its own.
    When:
        They are piled up under the default filter, with no exclusion
        mask, with a quality floor, keeping first mates alone, and with
        a template-length floor and ceiling.
    Then:
        It should count exactly the 5' ends of the records that filter
        keeps, and never the unmapped one.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        [flagged],
        reference_fasta,
        str(tmp_path / "flagged"),
        aggregate=dataclasses.replace(aggregate, unstranded=True),
    )

    # Assert
    assert positions(os.path.join(paths[0], "chr1.parquet")) == kept


@pytest.mark.parametrize("positive", ["forward", "reverse"])
def test_pileup_alignments_should_count_a_fragments_midpoint_whichever_mate_carries_it(
    tmp_path, long_reference, positive
):
    """Test the midpoint of a fragment of known span.

    Given:
        One pair spanning bases 1000 to 1299, the positive template
        length on the forward, leftmost mate or on the reverse,
        rightmost one.
    When:
        It is piled up under the midpoint rule.
    Then:
        It should count one fragment, at base 1149: the lower of the
        two middle bases of a 300-base fragment.
    """

    # Arrange
    header = builders.alignment_header({"chr1": 2000})
    bam = builders.write_bam(
        str(tmp_path / "pair.bam"),
        mate_pair(header, "p", "chr1", 1000, 300, positive),
        header,
    )

    # Act
    paths = pileup_pkg.pileup_alignments(
        [bam],
        long_reference,
        str(tmp_path / "mid"),
        aggregate=Aggregate(position="midpoint", unstranded=True),
    )

    # Assert
    dense, frame = dense_from_parquet(paths[0], {"chr1": 2000})
    assert frame.height == 1
    assert dense["chr1"][1149] == 1


def test_pileup_alignments_should_count_three_prime_ends_with_each_shift_on_its_coordinate(
    tmp_path, shifted_pair
):
    """Test the 3' rule and which coordinate each shift moves under it.

    Given:
        A forward read on 100-119 and a reverse read on 300-319.
    When:
        They are piled up under the 3' rule, with a start shift of 2
        and an end shift of -3.
    Then:
        It should count the forward read at its last base moved by the
        end shift, 116, and the reverse read at its first base moved by
        the start shift, 302 -- each shift moving the coordinate it is
        named for, whichever end of the read that is.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "three"),
        aggregate=Aggregate(
            position="three_prime", start_shift=2, end_shift=-3, unstranded=True
        ),
    )

    # Assert
    assert positions(os.path.join(paths[0], "chr1.parquet")) == [116, 302]


@settings(FILE_BACKED, max_examples=60)
@given(
    position=st.sampled_from(["five_prime", "three_prime", "midpoint", "span"]),
    exclude=st.sets(st.sampled_from([0x100, 0x200, 0x400, 0x800, 0x10])),
    require=st.sampled_from([0, 0x1, 0x40, 0x80]),
    min_mapq=st.sampled_from([0, 5, 20, 60]),
    template=st.sampled_from([(0, None), (1, None), (0, 120), (50, 150)]),
    shifts=st.tuples(st.integers(-5, 5), st.integers(-5, 5)),
    unstranded=st.booleans(),
    spliced=st.booleans(),
)
def test_pileup_alignments_should_agree_with_pysam_under_any_aggregate(
    tmp_path,
    mixed_pairs,
    reference_fasta,
    pysam,
    position,
    exclude,
    require,
    min_mapq,
    template,
    shifts,
    unstranded,
    spliced,
):
    """Test every rule and every filter clause against pysam.

    Given:
        Pairs and unpaired records of seeded flags, qualities and
        fragment lengths, and any combination of an exclusion mask, a
        required flag, a quality floor, template-length bounds, a
        position rule, for the 5' and 3' rules shifts, for ``"span"``
        ``spliced`` or not, and the strands pooled or, except under the
        midpoint rule, kept apart.
    When:
        They are piled up.
    Then:
        Each channel should equal, at every base, what pysam implies
        for the same records on its strand, or on both when pooled: a
        count at each kept record's position for the point rules, the
        midpoint found from the two mates' 5' ends, and per-base depth,
        after the read-time prefix sum, for ``"span"``.
    """

    # Arrange
    point = position in ("five_prime", "three_prime")
    aggregate = Aggregate(
        exclude_flags=sum(exclude) | 0x4,
        require_flags=require,
        min_mapq=min_mapq,
        min_template_length=template[0],
        max_template_length=template[1],
        position=position,
        start_shift=shifts[0] if point else 0,
        end_shift=shifts[1] if point else 0,
        unstranded=unstranded or position == "midpoint",
        spliced=spliced and position == "span",
    )
    oracle = oracle_depth if position == "span" else oracle_pileup
    channels = strand_channels(oracle, [mixed_pairs], TWO_CONTIGS, pysam, aggregate)

    # Act
    paths = pileup_pkg.pileup_alignments(
        [mixed_pairs], reference_fasta, str(tmp_path / "any"), aggregate=aggregate
    )

    # Assert
    actual = coverage_from_track(paths, TWO_CONTIGS)
    for chrom in TWO_CONTIGS:
        assert len(actual[chrom]) == len(channels)
        for row, expected in zip(actual[chrom], channels):
            numpy.testing.assert_array_equal(row, expected[chrom])


def test_pileup_alignments_should_write_coverage_that_sums_to_per_base_depth(
    tmp_path, span_corpus, pysam
):
    """Test span coverage against pysam's depth, strand by strand.

    Given:
        Two alignments of unspliced records on both strands.
    When:
        They are piled up stranded under ``"span"`` and each channel is
        read back through the store's prefix sum.
    Then:
        The plus channel should equal pysam's per-base depth over the
        forward records and the minus channel its depth over the
        reverse ones, at every base.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        span_corpus.bams,
        span_corpus.fasta,
        str(tmp_path / "span"),
        aggregate=Aggregate(position="span"),
    )

    # Assert
    actual = coverage_from_track(paths, TWO_CONTIGS)
    plus = oracle_depth(
        span_corpus.bams,
        TWO_CONTIGS,
        pysam,
        Aggregate(exclude_flags=DEFAULT_EXCLUDE_FLAGS | 0x10, position="span"),
    )
    minus = oracle_depth(
        span_corpus.bams,
        TWO_CONTIGS,
        pysam,
        Aggregate(require_flags=0x10, position="span"),
    )
    assert plus["chr1"].any() and minus["chr1"].any()
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(actual[chrom][0], plus[chrom])
        numpy.testing.assert_array_equal(actual[chrom][1], minus[chrom])


@pytest.fixture(scope="module", params=[0.1, 1 / 3], ids=["tenth", "third"])
def scaled_spans(request, tmp_path_factory, span_corpus):
    """`span_corpus` piled up unstranded under ``"span"`` at a scale of a tenth or a third.

    Neither scale is a power of two. Returns ``(aggregate, paths)``.
    """

    aggregate = Aggregate(position="span", unstranded=True, scale_factor=request.param)
    paths = pileup_pkg.pileup_alignments(
        span_corpus.bams,
        span_corpus.fasta,
        str(tmp_path_factory.mktemp("track-scaled") / "scaled"),
        aggregate=aggregate,
    )
    return aggregate, paths


def test_pileup_alignments_should_write_span_deltas_unscaled_beside_their_scale(
    scaled_spans,
):
    """Test where a span track's scale goes.

    Given:
        Two alignments of unspliced records piled up unstranded under
        ``"span"`` at a scale factor of a tenth or a third.
    When:
        The written rows and the manifest are read.
    Then:
        Every row should hold an integer, the delta unscaled, and the
        manifest should record the scale.
    """

    # Arrange
    aggregate, paths = scaled_spans

    # Act
    record = tables.read_manifest(paths[0])
    rows, _ = dense_from_parquet(paths[0], DELTA_SIZES)

    # Assert
    assert record["scale"] == aggregate.scale_factor
    assert all((values == numpy.round(values)).all() for values in rows.values())


def test_pileup_alignments_should_write_span_coverage_that_scales_without_residue(
    scaled_spans, span_corpus, pysam
):
    """Test span coverage under a scale factor that is not a power of two.

    Given:
        Two alignments of unspliced records piled up unstranded under
        ``"span"`` at a scale factor of a tenth or a third.
    When:
        The track is read back through the store.
    Then:
        Every base past the last record's end should read exactly
        ``0.0``, and every base should equal pysam's depth times the
        scale, to ``float32`` precision.
    """

    # Arrange
    aggregate, paths = scaled_spans
    expected = oracle_depth(span_corpus.bams, TWO_CONTIGS, pysam, aggregate)

    # Act
    actual = coverage_from_track(paths, TWO_CONTIGS)

    # Assert
    for chrom in TWO_CONTIGS:
        last = int(numpy.flatnonzero(expected[chrom])[-1])
        assert (actual[chrom][0][last + 1 :] == 0.0).all()
        numpy.testing.assert_allclose(actual[chrom][0], expected[chrom], rtol=1e-6)


@pytest.mark.parametrize("unstranded", [True, False], ids=["unstranded", "stranded"])
def test_pileup_alignments_should_close_a_span_that_ends_on_its_contigs_last_base(
    tmp_path, last_base_corpus, unstranded
):
    """Test the closing event of a span that reaches its contig's end.

    Given:
        A forward record ending on chr1's last base, a reverse one
        ending on chr2's, and a forward record ending inside chr1.
    When:
        They are piled up under ``"span"``, pooled and split by strand.
    Then:
        Every track should hold a ``-1`` at the contig's length for
        each record ending there, and each contig's deltas should sum
        to zero in every track, so no coverage runs past a contig.
    """

    # Arrange
    polars = pytest.importorskip("polars")

    # Act
    paths = pileup_pkg.pileup_alignments(
        last_base_corpus.bams,
        last_base_corpus.fasta,
        str(tmp_path / "span"),
        aggregate=Aggregate(position="span", unstranded=unstranded),
    )

    # Assert
    frame = polars.concat([polars.read_parquet(path) for path in paths])
    for chrom, length in TWO_CONTIGS.items():
        rows = frame.filter(polars.col("chrom") == chrom)
        assert rows.filter(polars.col("pos") == length)["value"].to_list() == [-1.0]
    for path in paths:
        sums = polars.read_parquet(path).group_by("chrom").agg(polars.col("value").sum())
        assert sums["value"].to_list() == [0.0] * sums.height


@pytest.fixture(scope="module")
def overhang_corpus(corpus_factory):
    """One alignment whose header declares both contigs longer than the reference.

    On each strand, records end inside a contig, on its last base and
    past it, and start on its length and past it, so a ``"span"``
    pileup meets every way a span can leave its contig.
    """

    pytest.importorskip("polars_bio")
    return corpus_factory(
        "overhang",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        header_contigs={"chr1": 1100, "chr2": 600},
        reads=[
            ("inside", "chr1", 500, "20M", 0),
            ("last", "chr1", 980, "20M", 0),
            ("past", "chr1", 990, "30M", 0),
            ("past_r", "chr1", 985, "40M", 16),
            ("at", "chr1", 1000, "20M", 0),
            ("at_r", "chr1", 1000, "20M", 16),
            ("beyond", "chr1", 1050, "20M", 0),
            ("past2", "chr2", 495, "20M", 16),
            ("at2", "chr2", 500, "10M", 0),
        ],
    )


@pytest.mark.parametrize("unstranded", [True, False], ids=["unstranded", "stranded"])
def test_pileup_alignments_should_clip_a_span_to_its_contig_under_a_longer_header(
    tmp_path, overhang_corpus, unstranded
):
    """Test the spans of an alignment whose header outruns the reference.

    Given:
        Records ending on and past each contig's end and starting on
        and past its length, under a header declaring both contigs
        longer than the reference.
    When:
        They are piled up under ``"span"``, pooled and split by strand.
    Then:
        Each contig's deltas should sum to zero in every track, with no
        row past the contig's length.
    """

    # Arrange
    polars = pytest.importorskip("polars")

    # Act
    paths = pileup_pkg.pileup_alignments(
        overhang_corpus.bams,
        overhang_corpus.fasta,
        str(tmp_path / "span"),
        aggregate=Aggregate(position="span", unstranded=unstranded),
    )

    # Assert
    for path in paths:
        frame = polars.read_parquet(path)
        for chrom, length in TWO_CONTIGS.items():
            rows = frame.filter(polars.col("chrom") == chrom)
            assert rows["value"].sum() == 0.0
            assert (rows["pos"] <= length).all()


def test_pileup_alignments_should_read_a_clipped_span_as_depth_inside_its_contig(
    tmp_path, overhang_corpus, pysam
):
    """Test the coverage spans past a contig's end leave inside it.

    Given:
        The same records under a header longer than the reference.
    When:
        They are piled up unstranded under ``"span"`` and read back
        through the store over each contig and 30 bases past it.
    Then:
        It should equal pysam's depth inside each contig and read zero
        past it.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        overhang_corpus.bams,
        overhang_corpus.fasta,
        str(tmp_path / "span"),
        aggregate=Aggregate(position="span", unstranded=True),
    )

    # Assert
    store = stores.TrackStore(paths)
    depth = oracle_depth(
        overhang_corpus.bams,
        TWO_CONTIGS,
        pysam,
        Aggregate(position="span", unstranded=True),
    )
    for chrom, length in TWO_CONTIGS.items():
        expected = numpy.concatenate((depth[chrom], numpy.zeros(30)))
        numpy.testing.assert_array_equal(
            store.window(chrom, 0, length + 30)[0].numpy(), expected
        )


def test_pileup_alignments_should_cover_a_deletion_and_not_a_skip_under_spliced_span(
    tmp_path, bam_pair, reference_fasta
):
    """Test what a spliced span covers, where it differs from its record's span.

    Given:
        A record with a 5-base deletion at 210-214 and one with a
        100-base reference skip at 410-509.
    When:
        They are piled up under a spliced ``"span"``.
    Then:
        Each base inside the deletion should be covered and none inside
        the skip, and the skipping record should be written as four
        boundary events, two for its span and two for its skip, rather
        than one row per base.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        bam_pair,
        reference_fasta,
        str(tmp_path / "span"),
        aggregate=Aggregate(position="span", unstranded=True, spliced=True),
    )

    # Assert
    covered = coverage_from_track(paths, TWO_CONTIGS)["chr1"][0]
    assert covered[210:215].tolist() == [1.0] * 5
    assert covered[400:410].tolist() == covered[510:520].tolist() == [1.0] * 10
    assert covered[410:510].tolist() == [0.0] * 100
    rows, _ = dense_from_parquet(paths[0], DELTA_SIZES)
    assert rows["chr1"][[400, 410, 510, 520]].tolist() == [1, -1, 1, -1]
    assert numpy.count_nonzero(rows["chr1"][401:520]) == 2


@pytest.fixture(scope="module")
def skip_on_chr2(corpus_factory):
    """One alignment of unspliced records on chr1 and one skipping record on chr2.

    Split in two contig batches, the skipping record is in the second,
    so a pileup meets it once the first is already staged.
    """

    pytest.importorskip("polars_bio")
    return corpus_factory(
        "skip-chr2",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        reads=[
            ("plain", "chr1", 100, "20M", 0),
            ("plain_r", "chr1", 300, "20M", 16),
            ("skip", "chr2", 100, "10M50N10M", 16),
        ],
    )


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("unstranded", [True, False], ids=["unstranded", "stranded"])
def test_pileup_alignments_should_refuse_a_skipping_record_under_unspliced_span(
    tmp_path, skip_on_chr2, unstranded, groups
):
    """Test that an unspliced span never covers a skipped run.

    Given:
        Unspliced records on chr1 and a record skipping 50 bases of
        chr2, run in one contig batch or in two with the skip in the
        second.
    When:
        They are piled up under ``"span"`` without ``spliced``, pooled
        and split by strand.
    Then:
        It should raise `ValueError` naming ``spliced=True`` and the
        contig, and leave nothing in the output directory: no table,
        no record and no staging.
    """

    # Arrange
    out = tmp_path / "out"
    out.mkdir()

    # Act & assert
    with pytest.raises(
        ValueError, match=r"At least 1 record\(s\).* on chr2.*spliced=True"
    ):
        pileup_pkg.pileup_alignments(
            skip_on_chr2.bams,
            skip_on_chr2.fasta,
            str(out / "span"),
            aggregate=Aggregate(position="span", unstranded=unstranded),
            contig_groups=groups,
        )
    assert os.listdir(out) == []


def test_pileup_alignments_should_keep_no_grouped_result_alive_in_a_caught_refusal(
    tmp_path, spliced_corpus, mocker
):
    """Test that a caught refusal keeps no grouped result alive.

    Given:
        Alignments of records that skip reference, every frame a lazy
        query collects watched through a weak reference, and one frame
        collected and held by the test.
    When:
        They are piled up under ``"span"`` without ``spliced``, and the
        refusal is caught and held.
    Then:
        Every frame the pileup collected should be dead while the
        refusal is still held, and the one the test holds alive.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    collect = polars.LazyFrame.collect
    watched = []

    def watch(self, *args, **kwargs):
        frame = collect(self, *args, **kwargs)
        watched.append(weakref.ref(frame))
        return frame

    mocker.patch.object(polars.LazyFrame, "collect", watch)
    held = polars.LazyFrame({"pos": [0]}).collect()

    # Act
    with pytest.raises(ValueError, match="spliced=True") as caught:
        pileup_pkg.pileup_alignments(
            spliced_corpus.bams,
            spliced_corpus.fasta,
            str(tmp_path / "span"),
            aggregate=Aggregate(position="span"),
        )
    gc.collect()

    # Assert
    assert caught.value.__traceback__ is not None
    assert len(watched) > 1, "the pileup collected nothing"
    alive = [frame for frame in (ref() for ref in watched) if frame is not None]
    assert [frame is held for frame in alive] == [True]


def assert_spliced_writes_the_same_table(corpus, tmp_path):
    """Assert a corpus piles up alike under ``"span"`` with and without ``spliced``.

    Both are stranded and written under ``tmp_path``, and every Parquet
    file must be byte-identical between the two, at least one written.
    """

    plain, spliced = (
        pileup_pkg.pileup_alignments(
            corpus.bams,
            corpus.fasta,
            str(tmp_path / name),
            aggregate=Aggregate(position="span", spliced=flag),
        )
        for name, flag in (("plain", False), ("spliced", True))
    )
    for left, right in zip(plain, spliced, strict=True):
        assert digest_track(left), "table wrote no Parquet files"
        assert digest_track(left) == digest_track(right)


def test_pileup_alignments_should_write_a_skipping_record_past_its_contig_under_unspliced_span(
    tmp_path, corpus_factory
):
    """Test that a skipping record the clip leaves nothing of is not refused.

    Given:
        Unspliced records inside both contigs, one of them deleting
        five bases, and records skipping reference that start at and
        past chr2's length, under a header declaring chr2 longer than
        the reference.
    When:
        They are piled up stranded under ``"span"`` without
        ``spliced``, and with it.
    Then:
        It should write the table rather than refuse, every Parquet
        file byte-identical to the spliced one.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    corpus = corpus_factory(
        "skip-past",
        SPLICED_CONTIGS,
        n_bams=1,
        index_fasta=True,
        header_contigs=SPLICED_HEADER,
        reads=[
            ("plain", "chr1", 100, "20M", 0),
            ("deleted", "chr1", 300, "10M5D10M", 0),
            ("plain_r", "chr2", 300, "20M", 16),
            ("at_end", "chr2", 500, "5M5N5M", 16),
            ("beyond", "chr2", 520, "5M5N5M", 0),
        ],
    )

    # Act & assert
    assert_spliced_writes_the_same_table(corpus, tmp_path)


def test_pileup_alignments_should_drop_a_count_shifted_below_zero_rather_than_refuse(
    tmp_path, corpus_factory
):
    """Test that a counts pileup reads no shifted position as a refusal.

    Given:
        Forward records at 0-based positions 0, 4 and 100 of chr1.
    When:
        Their 5' ends are counted with a ``start_shift`` of ``-5``, which
        moves the first two to ``-5`` and ``-1``.
    Then:
        It should write the one position inside the contig, ``95``,
        rather than refuse or write a negative one.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    polars = pytest.importorskip("polars")
    corpus = corpus_factory(
        "shifted-below",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        reads=[
            ("at_zero", "chr1", 0, "20M", 0),
            ("at_four", "chr1", 4, "20M", 0),
            ("inside", "chr1", 100, "20M", 0),
        ],
    )

    # Act
    (path,) = pileup_pkg.pileup_alignments(
        corpus.bams,
        corpus.fasta,
        str(tmp_path / "shifted"),
        aggregate=Aggregate(start_shift=-5, unstranded=True),
    )

    # Assert
    assert polars.read_parquet(path).select("pos", "value").rows() == [(95, 1.0)]


def test_pileup_alignments_should_ignore_a_skipping_record_its_filter_drops(
    tmp_path, corpus_factory, pysam
):
    """Test that only a record the aggregate keeps is refused for skipping.

    Given:
        Unspliced records, and skipping records the aggregate drops: a
        duplicate, a secondary and one below its quality floor.
    When:
        They are piled up under ``"span"`` without ``spliced``.
    Then:
        It should write the coverage pysam's depth gives over the kept
        records, rather than refuse.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    corpus = corpus_factory(
        "skip-dropped",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        reads=[
            ("plain", "chr1", 100, "20M", 0),
            ("duplicate", "chr1", 200, "10M50N10M", 0x400),
            ("secondary", "chr1", 300, "10M50N10M", 0x100),
            ("plain_r", "chr2", 100, "20M", 16),
        ],
    )
    low = builders.aligned_read(
        builders.alignment_header(TWO_CONTIGS), "low", "chr2", 200, "5M5N5M", mapq=2
    )
    extra = builders.write_bam(
        str(tmp_path / "low.bam"), [low], builders.alignment_header(TWO_CONTIGS)
    )
    aggregate = Aggregate(position="span", unstranded=True, min_mapq=10)

    # Act
    paths = pileup_pkg.pileup_alignments(
        corpus.bams + [extra], corpus.fasta, str(tmp_path / "span"), aggregate=aggregate
    )

    # Assert
    actual = coverage_from_track(paths, TWO_CONTIGS)
    expected = oracle_depth(corpus.bams + [extra], TWO_CONTIGS, pysam, aggregate)
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(actual[chrom][0], expected[chrom])


@pytest.fixture(scope="module")
def spliced_tracks(tmp_path_factory, spliced_corpus):
    """`spliced_corpus` piled up under a spliced ``"span"``, pooled and by strand.

    Returns ``{unstranded: paths}``.
    """

    root = tmp_path_factory.mktemp("track-spliced")
    return {
        unstranded: pileup_pkg.pileup_alignments(
            spliced_corpus.bams,
            spliced_corpus.fasta,
            str(root / ("pooled" if unstranded else "split")),
            aggregate=Aggregate(position="span", unstranded=unstranded, spliced=True),
        )
        for unstranded in (True, False)
    }


@pytest.mark.parametrize("unstranded", [True, False], ids=["unstranded", "stranded"])
def test_pileup_alignments_should_write_spliced_coverage_that_sums_to_per_base_depth(
    spliced_tracks, spliced_corpus, pysam, unstranded
):
    """Test spliced coverage against a CIGAR walk, strand by strand.

    Given:
        Two alignments of the same spliced and unspliced records:
        repeated skip lengths in one record, identical records, a
        deletion beside a skip on either side, clips, insertions,
        ``=`` and ``X``, and runs past each contig's end under a header
        longer than the reference.
    When:
        They are piled up under a spliced ``"span"``, pooled and split
        by strand, and each channel is read back through the store.
    Then:
        Each channel should equal, at every base, the depth a walk of
        each kept record's CIGAR gives over its records, skipped runs
        uncovered and deletions covered.
    """

    # Arrange
    paths = spliced_tracks[unstranded]
    expected = strand_channels(
        oracle_depth,
        spliced_corpus.bams,
        SPLICED_CONTIGS,
        pysam,
        Aggregate(position="span", unstranded=unstranded, spliced=True),
    )

    # Act
    actual = coverage_from_track(paths, SPLICED_CONTIGS)

    # Assert
    for chrom in SPLICED_CONTIGS:
        assert len(actual[chrom]) == len(expected)
        for row, depth in zip(actual[chrom], expected):
            assert depth["chr1"].any()
            numpy.testing.assert_array_equal(row, depth[chrom])


def test_pileup_alignments_should_write_spliced_coverage_samtools_depth_agrees_with(
    spliced_tracks, spliced_corpus, pysam
):
    """Test pooled spliced coverage against ``samtools depth -a -J``.

    Given:
        The same two alignments piled up pooled under a spliced
        ``"span"``.
    When:
        The track is read back through the store, and ``samtools depth
        -a -J`` is run over both alignments by pysam.
    Then:
        Every base of every contig should read what samtools reports
        summed over the two files.
    """

    # Arrange
    depth = samtools_depth(spliced_corpus.bams, SPLICED_CONTIGS, pysam)

    # Act
    actual = coverage_from_track(spliced_tracks[True], SPLICED_CONTIGS)

    # Assert
    assert depth["chr1"].any()
    for chrom in SPLICED_CONTIGS:
        numpy.testing.assert_array_equal(actual[chrom][0], depth[chrom])


def long_cigar(n_ops, offset):
    """A CIGAR of ``n_ops`` operations, aligned blocks joined by skips and indels.

    Blocks cycle through ``M``, ``=`` and ``X`` and joins through skips
    of three lengths, deletions and insertions, so a record holds many
    skips of each length. ``offset`` shifts both cycles, so two calls
    with different offsets give different records. ``n_ops`` is odd, so
    the record ends on a block.
    """

    joins = ["4N", "2D", "9N", "1I", "30N", "3D", "2I", "9N", "4N"]
    parts = []
    for index in range(n_ops // 2):
        step = index + offset
        parts.append("{}{}".format(1 + step * 7 % 9, "M=X"[step % 3]))
        parts.append(joins[step % len(joins)])
    parts.append("5M")
    return "".join(parts)


def test_pileup_alignments_should_cover_records_of_a_thousand_operations_under_spliced_span(
    tmp_path, corpus_factory, pysam
):
    """Test spliced coverage over records whose CIGARs hold about a thousand operations.

    Given:
        Long-read-shaped records of 999 to 1,001 operations each, on
        both strands, one of them twice, one starting behind a soft
        clip, and one running past chr2's end under a header longer
        than the reference.
    When:
        They are piled up pooled under a spliced ``"span"`` and read
        back through the store.
    Then:
        Every base of every contig should read both the depth a walk of
        each record's CIGAR gives and what ``samtools depth -a -J``
        reports.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    contigs = {"chr1": 12000, "chr2": 3000}
    corpus = corpus_factory(
        "spliced-long",
        contigs,
        n_bams=1,
        index_fasta=True,
        header_contigs={"chr1": 12000, "chr2": 9000},
        reads=[
            ("long1", "chr1", 100, long_cigar(1001, 0), 0),
            ("long1_copy", "chr1", 100, long_cigar(1001, 0), 0),
            ("long2", "chr1", 300, "3S" + long_cigar(999, 4), 16),
            ("long3", "chr2", 200, long_cigar(1001, 2), 0),
        ],
    )
    aggregate = Aggregate(position="span", unstranded=True, spliced=True)
    walked = oracle_depth(corpus.bams, contigs, pysam, aggregate)
    reported = samtools_depth(corpus.bams, contigs, pysam)

    # Act
    (path,) = pileup_pkg.pileup_alignments(
        corpus.bams, corpus.fasta, str(tmp_path / "long"), aggregate=aggregate
    )

    # Assert
    actual = coverage_from_track([path], contigs)
    assert walked["chr1"].max() == 3.0 and walked["chr2"][-100:].any()
    for chrom in contigs:
        numpy.testing.assert_array_equal(actual[chrom][0], walked[chrom])
        numpy.testing.assert_array_equal(actual[chrom][0], reported[chrom])


@pytest.mark.parametrize("unstranded", [True, False], ids=["unstranded", "stranded"])
def test_pileup_alignments_should_clip_spliced_coverage_to_its_contig(
    spliced_tracks, unstranded
):
    """Test the deltas of spliced records that leave their contig.

    Given:
        Spliced records whose skipped runs and spans close past each
        contig's end, and one starting past chr2's length.
    When:
        They are piled up under a spliced ``"span"``, pooled and split
        by strand.
    Then:
        Each contig's deltas should sum to zero in every track, with no
        row past the contig's length.
    """

    # Arrange
    polars = pytest.importorskip("polars")

    # Act
    frames = [polars.read_parquet(path) for path in spliced_tracks[unstranded]]

    # Assert
    for frame in frames:
        for chrom, length in SPLICED_CONTIGS.items():
            rows = frame.filter(polars.col("chrom") == chrom)
            assert rows["value"].sum() == 0.0
            assert (rows["pos"] <= length).all()


@pytest.mark.parametrize(
    "reads, rows",
    [
        (
            [("issue", "chr1", 2000, "5M10N5M", 0)],
            [(2000, 1.0), (2005, -1.0), (2015, 1.0), (2020, -1.0)],
        ),
        (
            [("repeat", "chr1", 100, "23M112N20M102N32M102N25M", 0)],
            [
                (100, 1.0),
                (123, -1.0),
                (235, 1.0),
                (255, -1.0),
                (357, 1.0),
                (389, -1.0),
                (491, 1.0),
                (516, -1.0),
            ],
        ),
        (
            [("same{}".format(n), "chr1", 600, "10M50N10M", 0) for n in range(3)],
            [(600, 3.0), (610, -3.0), (660, 3.0), (670, -3.0)],
        ),
    ],
    ids=["one-skip", "repeated-length", "identical-records"],
)
def test_pileup_alignments_should_write_each_skipped_run_as_two_events(
    tmp_path, corpus_factory, reads, rows
):
    """Test the rows a spliced record writes.

    Given:
        A record skipping ten bases from 0-based 2005, the case
        ``samtools depth -a`` reports as zero across 2006-2015; a record
        skipping three runs, two of them of one length; and three
        identical skipping records.
    When:
        Each is piled up under a spliced ``"span"``.
    Then:
        It should write ``+1`` at each record's start and ``-1`` at its
        end, and ``-1`` at the first base of each skipped run and
        ``+1`` one past its last, each times the number of identical
        records, and nothing else.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    polars = pytest.importorskip("polars")
    corpus = corpus_factory(
        "spliced-rows", SPLICED_CONTIGS, n_bams=1, index_fasta=True, reads=reads
    )

    # Act
    (path,) = pileup_pkg.pileup_alignments(
        corpus.bams,
        corpus.fasta,
        str(tmp_path / "rows"),
        aggregate=Aggregate(position="span", unstranded=True, spliced=True),
    )

    # Assert
    written = polars.read_parquet(path).select("pos", "value").rows()
    assert written == rows


def test_pileup_alignments_should_write_the_same_table_spliced_over_unspliced_records(
    tmp_path, span_corpus
):
    """Test that ``spliced`` changes nothing where no record skips.

    Given:
        Two alignments of unspliced records on both strands.
    When:
        They are piled up stranded under ``"span"`` with and without
        ``spliced``.
    Then:
        Every Parquet file should be byte-identical between the two.
    """

    # Act & assert
    assert_spliced_writes_the_same_table(span_corpus, tmp_path)


@st.composite
def cigars(draw, joins="NDI"):
    """A CIGAR of aligned blocks joined by skips, deletions and insertions.

    Each block is ``M``, ``=`` or ``X``, each join one or two of the
    operations ``joins`` names, ``N``, ``D`` and ``I`` by default, in
    any order, so a skip may sit beside a deletion, and either end may
    carry a soft clip. Skip lengths come from a set of three, so two
    skips of one length in one record are common.
    """

    def block():
        return "{}{}".format(draw(st.integers(1, 15)), draw(st.sampled_from("M=X")))

    def join():
        kind = draw(st.sampled_from(joins))
        if kind == "N":
            return "{}N".format(draw(st.sampled_from([4, 9, 30])))
        return "{}{}".format(draw(st.integers(1, 4)), kind)

    clip = "{}S".format(draw(st.integers(1, 3))) if draw(st.booleans()) else ""
    parts = [clip, block()]
    for _ in range(draw(st.integers(0, 4))):
        parts += [join() for _ in range(draw(st.integers(1, 2)))]
        parts.append(block())
    if draw(st.booleans()):
        parts.append("{}S".format(draw(st.integers(1, 3))))
    return "".join(parts)


def write_drawn_records(tmp_path, records):
    """Write drawn records to one indexed BAM in a fresh directory under ``tmp_path``.

    Each record is ``(chrom, pos, cigar, flag, copies)``, written
    ``copies`` times under `SPLICED_HEADER`. The directory is fresh per
    call because Hypothesis draws every example of a test into one
    ``tmp_path``.

    :returns: ``(root, bam)``, the directory as a `pathlib.Path`, which
        a pileup's output may share, and the BAM's path.
    """

    root = pathlib.Path(tempfile.mkdtemp(dir=tmp_path))
    header = builders.alignment_header(SPLICED_HEADER)
    reads = [
        builders.aligned_read(
            header, "r{}_{}".format(index, copy), chrom, pos, cigar, flag
        )
        for index, (chrom, pos, cigar, flag, copies) in enumerate(records)
        for copy in range(copies)
    ]
    return root, builders.write_bam(str(root / "any.bam"), reads, header)


@settings(FILE_BACKED, max_examples=50)
@given(
    records=st.lists(
        st.tuples(
            st.sampled_from(["chr1", "chr2"]),
            st.integers(0, 590),
            cigars(),
            st.sampled_from([0, 0x10]),
            st.integers(1, 3),
        ),
        min_size=1,
        max_size=10,
    ),
    unstranded=st.booleans(),
)
def test_pileup_alignments_should_agree_with_a_cigar_walk_under_spliced_span(
    tmp_path, spliced_corpus, pysam, records, unstranded
):
    """Test spliced coverage against a CIGAR walk over any spliced records.

    Given:
        Any handful of records over two contigs, each placed up to 90
        bases past chr2's end, with any CIGAR of aligned blocks joined
        by skips, deletions and insertions, on either strand, each
        repeated up to three times, under a header longer than the
        reference.
    When:
        They are piled up under a spliced ``"span"``, pooled or split
        by strand, and read back through the store.
    Then:
        Each channel should equal, at every base, the depth a walk of
        each record's CIGAR gives over the records on its strand, or on
        both when pooled.
    """

    # Arrange
    root, bam = write_drawn_records(tmp_path, records)
    aggregate = Aggregate(position="span", unstranded=unstranded, spliced=True)
    expected = strand_channels(oracle_depth, [bam], SPLICED_CONTIGS, pysam, aggregate)

    # Act
    paths = pileup_pkg.pileup_alignments(
        [bam], spliced_corpus.fasta, str(root / "any"), aggregate=aggregate
    )

    # Assert
    actual = coverage_from_track(paths, SPLICED_CONTIGS)
    for chrom in SPLICED_CONTIGS:
        for row, depth in zip(actual[chrom], expected, strict=True):
            numpy.testing.assert_array_equal(row, depth[chrom])


@settings(FILE_BACKED, max_examples=50)
@given(
    records=st.lists(
        st.tuples(
            st.sampled_from(["chr1", "chr2"]),
            st.integers(0, 590) | st.integers(500, 590),
            cigars(joins="DI") | cigars(),
            st.sampled_from([0, 0x10, 0x400]),
            st.integers(1, 2),
        ),
        min_size=1,
        max_size=4,
    ),
    unstranded=st.booleans(),
)
def test_pileup_alignments_should_refuse_under_unspliced_span_exactly_when_a_kept_record_skips_inside_its_contig(
    tmp_path, spliced_corpus, records, unstranded
):
    """Test which records an unspliced span refuses.

    Given:
        Any handful of records over two contigs, each placed up to 90
        bases past chr2's end and often past it, with any CIGAR of
        aligned blocks joined by deletions and insertions, or by those
        and skips, on either strand or flagged duplicate, each repeated
        up to twice, under a header longer than the reference.
    When:
        They are piled up under ``"span"`` without ``spliced``, pooled
        or split by strand.
    Then:
        It should raise `ValueError` naming ``spliced=True`` if and only
        if a record the aggregate keeps holds an ``N`` and starts inside
        its contig.
    """

    # Arrange
    root, bam = write_drawn_records(tmp_path, records)
    aggregate = Aggregate(position="span", unstranded=unstranded)
    refused = any(
        "N" in cigar
        and not flag & aggregate.exclude_flags
        and pos < SPLICED_CONTIGS[chrom]
        for chrom, pos, cigar, flag, _ in records
    )

    # Act
    try:
        pileup_pkg.pileup_alignments(
            [bam], spliced_corpus.fasta, str(root / "any"), aggregate=aggregate
        )
    except ValueError as error:
        raised = "spliced=True" in str(error)
    else:
        raised = False

    # Assert
    assert raised == refused


@settings(FILE_BACKED, max_examples=50)
@given(
    records=st.lists(
        st.tuples(
            st.sampled_from(["chr1", "chr2"]),
            st.integers(0, 590),
            cigars(joins="DI"),
            st.sampled_from([0, 0x10, 0x400]),
            st.integers(1, 2),
        )
        | st.tuples(
            st.just("chr2"),
            st.integers(500, 590),
            cigars(),
            st.sampled_from([0, 0x10, 0x400]),
            st.integers(1, 2),
        )
        | st.tuples(
            st.sampled_from(["chr1", "chr2"]),
            st.integers(0, 590),
            cigars(),
            st.just(0x400),
            st.integers(1, 2),
        ),
        min_size=1,
        max_size=4,
    ),
    unstranded=st.booleans(),
)
def test_pileup_alignments_should_agree_with_a_cigar_walk_and_with_spliced_under_unspliced_span(
    tmp_path, spliced_corpus, pysam, records, unstranded
):
    """Test what an unspliced span covers wherever it refuses nothing.

    Given:
        Any handful of records over two contigs that an unspliced span
        does not refuse, under a header longer than the reference: on
        either strand or flagged duplicate, with any CIGAR of aligned
        blocks joined by deletions and insertions placed anywhere up to
        90 bases past chr2's end, one joined by skips as well placed
        past chr2's end, or one of those anywhere flagged duplicate,
        each repeated up to twice.
    When:
        They are piled up under ``"span"`` without ``spliced``, pooled
        or split by strand, and again with ``spliced``.
    Then:
        Each channel should equal, at every base, the depth a walk of
        each record's CIGAR gives over the records on its strand, or on
        both when pooled, and every Parquet file should be
        byte-identical to the spliced one.
    """

    # Arrange
    root, bam = write_drawn_records(tmp_path, records)
    aggregate = Aggregate(position="span", unstranded=unstranded)
    expected = strand_channels(oracle_depth, [bam], SPLICED_CONTIGS, pysam, aggregate)

    # Act
    paths = pileup_pkg.pileup_alignments(
        [bam], spliced_corpus.fasta, str(root / "plain"), aggregate=aggregate
    )
    spliced = pileup_pkg.pileup_alignments(
        [bam],
        spliced_corpus.fasta,
        str(root / "spliced"),
        aggregate=dataclasses.replace(aggregate, spliced=True),
    )

    # Assert
    actual = coverage_from_track(paths, SPLICED_CONTIGS)
    for chrom in SPLICED_CONTIGS:
        for row, depth in zip(actual[chrom], expected, strict=True):
            numpy.testing.assert_array_equal(row, depth[chrom])
    for left, right in zip(paths, spliced, strict=True):
        assert digest_track(left) == digest_track(right)


@pytest.fixture(
    scope="module",
    params=[
        (Aggregate(unstranded=True), "counts", 1),
        (Aggregate(position="midpoint", unstranded=True), "counts", 1),
        (Aggregate(position="span", unstranded=True, min_mapq=3), "coverage_deltas", 2),
        (
            Aggregate(position="span", unstranded=True, spliced=True),
            "coverage_deltas",
            2,
        ),
    ],
    ids=["five_prime", "midpoint", "span", "spliced"],
)
def recorded(request, corpus_factory):
    """One alignment piled up under a point rule, the midpoint rule or a ``"span"``.

    Returns ``(aggregate, signal, version, record)``: the aggregate, the
    signal and manifest version a track of it should declare, and the
    record its track was committed with.
    """

    pytest.importorskip("polars_bio")
    aggregate, signal, version = request.param
    corpus = corpus_factory(
        "recorded",
        TWO_CONTIGS,
        n_bams=1,
        index_fasta=True,
        reads=[("fwd", "chr1", 100, "20M", 0), ("rev", "chr1", 300, "20M", 16)],
    )
    paths = pileup_pkg.pileup_alignments(
        corpus.bams, corpus.fasta, corpus.path("m"), aggregate=aggregate
    )
    return aggregate, signal, version, tables.read_manifest(paths[0])


def test_pileup_alignments_should_record_what_the_values_mean(recorded):
    """Test the manifest's signal and the version it implies.

    Given:
        A point rule, the midpoint rule, ``"span"`` and a spliced
        ``"span"``.
    When:
        Alignments are piled up under each and the manifest is read.
    Then:
        It should declare counts at version 1 for the first two and
        coverage deltas at version 2 for both spans.
    """

    # Arrange
    _, signal, version, record = recorded

    # Act
    declared = (record["signal"], record["version"])

    # Assert
    assert declared == (signal, version)


def test_pileup_alignments_should_record_the_aggregate_it_counted_by(recorded):
    """Test the aggregate's provenance in the manifest.

    Given:
        A point rule, the midpoint rule, ``"span"`` and a spliced
        ``"span"``.
    When:
        Alignments are piled up under each and the manifest is read.
    Then:
        It should record every field of the aggregate under
        ``parameters``, ``spliced`` among them, as
        `dataclasses.asdict` gives them.
    """

    # Arrange
    aggregate, _, _, record = recorded

    # Act
    provenance = record["parameters"]["aggregate"]

    # Assert
    assert provenance == dataclasses.asdict(aggregate)
    assert provenance["spliced"] is aggregate.spliced


@pytest.fixture(scope="module")
def eight_contigs(corpus_factory):
    """One alignment over eight contigs of equal length, a record on each.

    Eight is the planner's batch ceiling, so every batch count it can
    answer splits these into that many queries.
    """

    pytest.importorskip("polars_bio")
    sizes = {"chr{}".format(index): 200 for index in range(1, 9)}
    return corpus_factory(
        "eight",
        sizes,
        n_bams=1,
        index_fasta=True,
        reads=[("r_" + chrom, chrom, 50, "20M", 0) for chrom in sizes],
    )


@pytest.mark.parametrize(
    "position, spliced, queries",
    [("five_prime", False, 1), ("span", False, 2), ("span", True, 8)],
    ids=["five_prime", "span", "spliced"],
)
def test_pileup_alignments_should_split_by_the_scan_its_aggregate_implies(
    tmp_path, eight_contigs, machine, sized, mocker, position, spliced, queries
):
    """Test that a direct call sizes its batches by its record weight.

    Given:
        An alignment over eight contigs, reported as 6 GiB on a machine
        with memory to spare, piled up under the 5' rule, under
        ``"span"`` and under a spliced ``"span"``.
    When:
        The batch count is left to the planner.
    Then:
        The 5' pileup should run as one batch, its bytes under the scan
        cap, the span pileup as two, its record weight of two taking it
        to 12 GiB, and the spliced one as eight, its weight of ten taking
        it to 60 GiB: each should ask the engine as many queries as the
        same pileup run at that batch count.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    machine(64, 1024 * 1024**3)
    sized(6 * 1024**3)
    spy = mocker.spy(pb, "sql")
    aggregate = Aggregate(position=position, unstranded=True, spliced=spliced)

    # Act
    pileup_pkg.pileup_alignments(
        eight_contigs.bams,
        eight_contigs.fasta,
        str(tmp_path / "planned"),
        aggregate=aggregate,
        target_partitions=1,
    )
    planned = spy.call_count
    spy.reset_mock()
    pileup_pkg.pileup_alignments(
        eight_contigs.bams,
        eight_contigs.fasta,
        str(tmp_path / "named"),
        aggregate=aggregate,
        target_partitions=1,
        contig_groups=queries,
    )

    # Assert
    assert planned == spy.call_count


@pytest.mark.parametrize("aggregate", [None, {"unstranded": True}, "span"])
def test_pileup_alignments_should_refuse_an_aggregate_that_is_not_one(
    tmp_path, shifted_pair, mocker, aggregate
):
    """Test the type the aggregate is held to.

    Given:
        ``None``, a mapping of the record's fields, or a rule's name in
        place of an `Aggregate`.
    When:
        A pileup is called with it.
    Then:
        It should raise `TypeError`, and `register_alignments` should
        never have been called.
    """

    # Arrange
    registering = mocker.patch.object(
        track, "register_alignments", wraps=track.register_alignments
    )

    # Act & assert
    with pytest.raises(TypeError, match="takes what to count as an Aggregate"):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "x"),
            aggregate=aggregate,
        )
    registering.assert_not_called()


def test_pileup_alignments_should_write_the_same_table_at_a_named_partition_count(
    tmp_path, shifted_pair, mocker
):
    """Test that the partition count bounds threads and nothing else.

    Given:
        One pileup run with ``target_partitions`` left to the package
        and again at an explicit count of one.
    When:
        The written tables are hashed.
    Then:
        It should write byte-identical Parquet, the setting being a
        concurrency bound rather than part of the answer, and should
        have asked the engine for the count it was given.

    The second half is not redundant. Output equality was the whole of
    this test, and a setting that never reached the engine satisfies it
    just as well as one that did -- which is what let an
    ``except Exception: pass`` around the call go unnoticed. The spy is
    what distinguishes an applied setting from a discarded one.

    Membership rather than the last call, because the pileup restores
    the option it set and the restore is therefore the last call. That
    the restore happens is a different claim with a test of its own.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    spy = mocker.spy(pb, "set_option")

    # Act
    chosen = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "chosen"),
        aggregate=UNSTRANDED,
    )
    named = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "named"),
        aggregate=UNSTRANDED,
        target_partitions=1,
    )

    # Assert
    assert digest_track(chosen[0]) == digest_track(named[0])
    assert (
        mocker.call("datafusion.execution.target_partitions", "1") in spy.call_args_list
    )


def test_pileup_alignments_should_warn_when_the_engine_rejects_the_partition_count(
    tmp_path, shifted_pair, mocker
):
    """Test what a refused engine setting costs the caller.

    Given:
        An engine that panics out of Rust when asked to set the
        partition count, as one whose config keys have been renamed
        would.
    When:
        A pileup runs.
    Then:
        It should warn and write the same table it writes when the
        setting is accepted.

    The panic derives from `BaseException`, so the ``except Exception``
    this replaced caught every failure except the one it was written
    for. A polars-bio release renaming
    ``datafusion.execution.target_partitions`` ended the pileup it was
    supposed to protect.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    accepted = pileup_pkg.pileup_alignments(
        shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "ok"), aggregate=UNSTRANDED
    )
    mocker.patch.object(
        pb, "set_option", side_effect=EnginePanic("not found on ExecutionOptions")
    )

    # Act
    with pytest.warns(EngineOptionWarning, match="target_partitions"):
        refused = pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "refused"),
            aggregate=UNSTRANDED,
        )

    # Assert
    assert digest_track(refused[0]) == digest_track(accepted[0])


def test_pileup_alignments_should_not_warn_when_the_engine_takes_the_setting(
    tmp_path, shifted_pair
):
    """Test that the quiet path stays quiet.

    Given:
        An engine that accepts the partition count, which is every
        engine this package is tested against.
    When:
        A pileup runs.
    Then:
        It should emit no warning.

    The other half of the warning contract. A guard that warned
    whatever happened would satisfy the test above and make the
    warning worthless.
    """

    # Act & assert
    with warnings.catch_warnings():
        warnings.simplefilter("error", EngineOptionWarning)
        pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "quiet"),
            aggregate=UNSTRANDED,
        )


def test_pileup_alignments_should_propagate_an_interrupt_from_the_engine(
    tmp_path, shifted_pair, mocker
):
    """Test that the guard does not swallow a keyboard interrupt.

    Given:
        A caller who interrupts the process while the engine is being
        configured.
    When:
        A pileup runs.
    Then:
        It should raise `KeyboardInterrupt` rather than warning and
        carrying on.

    Catching `BaseException` to see a Rust panic also puts
    `KeyboardInterrupt` and `SystemExit` in reach, and converting those
    into a warning would make the pileup ignore a request to stop. They
    are re-raised ahead of the warning for that reason, which is the
    thing this pins.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    mocker.patch.object(pb, "set_option", side_effect=KeyboardInterrupt)

    # Act & assert
    with pytest.raises(KeyboardInterrupt):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "interrupted"),
            aggregate=UNSTRANDED,
        )


def test_pileup_alignments_should_raise_when_the_partition_count_is_not_a_number(
    tmp_path, shifted_pair
):
    """Test that a bad partition count reaches the caller.

    Given:
        A ``target_partitions`` that is not a number.
    When:
        A pileup runs.
    Then:
        It should raise, rather than warning and running at whatever
        parallelism the session already had.

    The conversion used to sit inside the guard, so a caller's
    `TypeError` was caught by a block meant for the backend and their
    argument was silently dropped. Two different failures with two
    different owners were being handled as one.
    """

    # Act & assert
    with pytest.raises((TypeError, ValueError)):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "bad"),
            aggregate=UNSTRANDED,
            target_partitions="all of them",
        )


def test_pileup_alignments_should_report_each_channel_when_verbose(
    tmp_path, shifted_pair, capsys
):
    """Test the progress output of one pileup.

    Given:
        An alignment and ``verbose`` set.
    When:
        It is piled up.
    Then:
        It should name the alignments it pooled and report each written
        table with the number of positions it holds.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "loud"),
        aggregate=UNSTRANDED,
        verbose=True,
    )

    # Assert
    reported = capsys.readouterr().out
    assert shifted_pair.bams[0] in reported
    assert "wrote 2 positions to {}".format(paths[0]) in reported


def test_pileup_alignments_should_print_nothing_when_verbose_is_defaulted(
    tmp_path, shifted_pair, capsys
):
    """Test that the default pileup is silent.

    Given:
        An alignment and ``verbose`` left at its default.
    When:
        It is piled up.
    Then:
        It should write its table and print nothing, a library being
        quiet unless asked.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "quiet"),
        aggregate=UNSTRANDED,
    )

    # Assert
    assert capsys.readouterr().out == ""
    assert digest_track(paths[0])


def test_pileup_alignments_should_refuse_an_empty_alignment_list(
    reference_fasta, tmp_path
):
    """Test the refusal of a pileup over nothing.

    Given:
        No alignments at all.
    When:
        A pileup is requested.
    Then:
        It should refuse rather than write a table that reads as valid.

    An empty pool reaches none of the guards downstream: there is no
    header to check against the reference and no contig batch to
    aggregate, and the writer still commits a manifest listing no files,
    which `require_complete` passes because every file it lists does
    exist.
    """

    # Act & assert
    with pytest.raises(ValueError, match="no alignments"):
        pileup_pkg.pileup_alignments([], reference_fasta, str(tmp_path / "nothing"))

    assert not list(tmp_path.glob("nothing*"))


def test_pileup_alignments_should_refuse_one_of_two_concurrent_callers(
    bam_pair, reference_fasta, tmp_path
):
    """Test that two pileups in one process cannot overlap.

    Given:
        Two threads entering `pileup_alignments` behind a barrier.
    When:
        Both run to completion.
    Then:
        It should let exactly one write its table and refuse the other.

    The polars-bio session is process-global and the table names are
    fixed, so two overlapping pileups read each other's reads -- and the
    damage is invisible afterwards, because each writes a manifest
    describing the files it wrote and both pass `require_complete`. The
    refusal is non-blocking on purpose: a second pileup in one process
    is a caller error, and queueing it would turn that into a hang.
    """

    # Arrange
    ready = threading.Barrier(2)
    outcomes = []

    def attempt(label):
        ready.wait(10)
        try:
            pileup_pkg.pileup_alignments(
                bam_pair, reference_fasta, str(tmp_path / label), aggregate=UNSTRANDED
            )
            outcomes.append("wrote")
        except RuntimeError as error:
            assert "already running in this process" in str(error)
            outcomes.append("refused")

    # Act
    threads = [threading.Thread(target=attempt, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    # Assert
    assert sorted(outcomes) == ["refused", "wrote"]


def test_pileup_alignments_should_succeed_after_a_refused_overlapping_call(
    bam_pair, reference_fasta, tmp_path
):
    """Test that the refusal is about overlap and not about history.

    Given:
        A process in which an overlapping pileup has already been
        refused and the holder has returned.
    When:
        A further pileup is requested.
    Then:
        It should succeed, the slot being released rather than spent.
    """

    # Act & assert
    assert pileup_pkg.pileup_alignments(
        bam_pair, reference_fasta, str(tmp_path / "after"), aggregate=UNSTRANDED
    )


def test_pileup_alignments_should_refuse_an_alignment_that_is_neither_bam_nor_sam(
    tmp_path, reference_fasta
):
    """Test the suffix an alignment is held to.

    Given:
        An alignment path ending in ``.txt``.
    When:
        A pileup is requested.
    Then:
        It should raise `ValueError` quoting the path.
    """

    # Arrange
    path = tmp_path / "reads.txt"
    path.write_text("not an alignment")

    # Act & assert
    with pytest.raises(ValueError, match="reads.txt"):
        pileup_pkg.pileup_alignments([str(path)], reference_fasta, str(tmp_path / "t"))


@pytest.mark.parametrize(
    "value, error", [("two", ValueError), ([2], TypeError)], ids=["string", "list"]
)
def test_pileup_alignments_should_raise_when_the_batch_count_is_not_a_number(
    tmp_path, shifted_pair, value, error
):
    """Test the batch count's refusals.

    Given:
        A ``contig_groups`` that is a string ``int`` cannot parse, or a
        list.
    When:
        A pileup runs.
    Then:
        It should raise `ValueError` for the string and `TypeError` for
        the list.
    """

    # Act & assert
    with pytest.raises(error):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams,
            shifted_pair.fasta,
            str(tmp_path / "bad"),
            aggregate=UNSTRANDED,
            contig_groups=value,
        )


@pytest.mark.xfail(
    strict=True,
    reason="an all-unmapped BAM still commits an empty blessed table; the "
    "refusal for it has to be taken per pileup rather than per writer, "
    "which the empty-argument-list guard does not reach",
)
def test_pileup_alignments_should_refuse_a_pool_that_contributes_no_record(
    bam_header, reference_fasta, tmp_path
):
    """Test the one route into an empty blessed table that is still open.

    Given:
        Alignments whose every record is unmapped.
    When:
        A pileup is requested.
    Then:
        It should refuse rather than commit an empty blessed table.

    This is the third route into the state two docstrings in this
    package say they exist to prevent, and the only one still open. The
    header names reference contigs, so the contig-naming refusal passes
    it; the batches are non-empty, so the split passes it; and the
    writers stage nothing, so the manifest records no files and zero
    rows and `require_complete` returns the path. Measured: it does.

    Closing it needs the refusal taken across the channels of one
    pileup -- no channel staged anything -- and not per writer, which
    breaks an ordinary one-strand pileup, because a strand with no reads
    is a legitimate empty channel and the first channel commits before
    the second could raise.
    """

    # Arrange
    unmapped = [
        builders.aligned_read(bam_header, "u{}".format(i), None, 0, None, 4)
        for i in range(10)
    ]
    path = builders.write_bam(str(tmp_path / "unmapped.bam"), unmapped, bam_header)

    # Act & assert
    with pytest.raises(ValueError):
        pileup_pkg.pileup_alignments([path], reference_fasta, str(tmp_path / "quiet"))


def test_pileup_alignments_should_write_no_file_for_a_contig_with_no_rows(
    track_paths,
):
    """Test which contigs a written track turns out to hold.

    Given:
        A pileup whose BAM header named a contig the reference does not
        have.
    When:
        The written track's contigs are listed.
    Then:
        It should hold the contigs that kept rows and no file at all for
        the one that did not.

    ``chrZ`` is in the header and absent from the reference, so every
    record on it was dropped by the join against the chrom sizes. A file
    written for it would read as a contig with no signal rather than as
    a contig that is not in this genome.
    """

    # Act & assert
    assert tables.track_contigs(track_paths[0]) == {"chr1", "chr2"}


def test_pileup_alignments_should_write_fixed_size_row_groups(corpus_factory, tmp_path):
    """Test how a contig's rows are grouped inside its Parquet file.

    Given:
        A pileup over one contig long enough to need several row groups.
    When:
        The written file's Parquet metadata is read.
    Then:
        It should hold 10,000 rows in every group but the last, which
        holds the remainder.

    The row-group size is the only thing standing between a windowed
    read and a scan of the whole contig, and nothing in the suite
    asserted it: a grep for ``row_group`` matched two lines of source
    and no test. The size that shipped had been measured against a
    single-file track and was never re-derived after the per-contig
    split, which is the regression this pins -- not a wrong number, but
    a number nothing was watching.

    The size is spelled out here rather than read from
    ``tables``. Reading it back would make this a tautology that passes
    for any value the writer happens to hold, and the number is part of
    the format's contract -- `TrackWriter` records the measurement
    behind it -- so a deliberate change to it should have to come here
    and say so.

    Asserted as a partition rather than as a count so the remainder is
    covered too. A writer that passed no size at all would leave one
    group per file and still satisfy a count assertion.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    pq = pytest.importorskip("pyarrow.parquet")
    corpus = corpus_factory(
        "row-groups", {"chr1": 60_000}, n_bams=1, n_reads=30_000, index_fasta=True
    )

    # Act
    written = pileup_pkg.pileup_alignments(
        corpus.bams, corpus.fasta, str(tmp_path / "grouped"), aggregate=UNSTRANDED
    )
    metadata = pq.ParquetFile(tables.track_chrom_path(written[0], "chr1")).metadata
    groups = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]

    # Assert
    assert metadata.num_rows > 10_000, "the corpus must outgrow one group to test this"
    full, remainder = divmod(metadata.num_rows, 10_000)
    assert groups == [10_000] * full + ([remainder] if remainder else [])


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("spliced", [False, True], ids=["atac", "spliced"])
def test_pileup_alignments_should_write_the_same_table_at_any_batch_count(
    tmp_path, bam_pair, reference_fasta, spliced_corpus, spliced, groups
):
    """Test that contig batching changes nothing it writes.

    Given:
        An ATAC pileup, and a spliced ``"span"`` over records skipping
        reference on both contigs, each run as one aggregate and as
        several contig batches.
    When:
        The written tables are hashed.
    Then:
        It should write byte-for-byte identical Parquet, with the same
        row totals.

    It holds because batches partition the contigs, and a position
    belongs to exactly one contig, so no count can straddle a batch
    boundary.
    """

    # Arrange
    if spliced:
        bams, fasta = spliced_corpus.bams, spliced_corpus.fasta
        aggregate = Aggregate(position="span", spliced=True)
    else:
        bams, fasta, aggregate = [bam_pair[0]], reference_fasta, ATAC

    # Act
    one = pileup_pkg.pileup_alignments(
        bams, fasta, str(tmp_path / "whole"), aggregate=aggregate, contig_groups=1
    )
    many = pileup_pkg.pileup_alignments(
        bams,
        fasta,
        str(tmp_path / "split"),
        aggregate=aggregate,
        contig_groups=groups,
    )

    # Assert
    for left, right in zip(one, many):
        assert digest_track(left) == digest_track(right)
        assert digest_track(left), "table wrote no Parquet files"
        assert tables.read_manifest(left)["rows"] == tables.read_manifest(right)["rows"]


def test_pileup_alignments_should_tolerate_a_reference_contig_the_bam_omits(
    tmp_path, bam_pair, corpus_factory
):
    """Test a reference with more contigs than the alignment.

    Given:
        A reference carrying a contig absent from the BAM header.
    When:
        The pileup is split into contig batches.
    Then:
        It should complete, match the ungrouped run, and write no file
        for the contig that has no rows.

    polars-bio raises rather than returning nothing when asked for a
    region its header does not name, so batches have to be built from
    the intersection. A reference with more contigs than the alignment
    is the normal case, not an edge case -- hg38 has 455 where an ATAC
    BAM has 195.
    """

    # Arrange
    extra = corpus_factory(
        "track-extra", dict(TWO_CONTIGS, chrUnplaced=300), index_fasta=True
    ).fasta

    # Act
    whole = pileup_pkg.pileup_alignments(
        [bam_pair[0]], extra, str(tmp_path / "w"), aggregate=UNSTRANDED, contig_groups=1
    )
    split = pileup_pkg.pileup_alignments(
        [bam_pair[0]], extra, str(tmp_path / "s"), aggregate=UNSTRANDED, contig_groups=3
    )

    # Assert
    assert digest_track(whole[0]) == digest_track(split[0])
    assert "chrUnplaced.parquet" not in digest_track(split[0])


def test_pileup_alignments_should_keep_every_record_when_headers_differ(
    tmp_path, bam_pair, bam_header, reference_fasta
):
    """Test pooling alignments whose headers name different contigs.

    Given:
        One channel pooling a two-contig BAM and a one-contig BAM.
    When:
        The pileup is split into contig batches.
    Then:
        It should match the ungrouped run and keep the contig only one
        of them has.

    Each branch is restricted to the contigs its own table has. Taking
    the intersection across the pooled alignments instead would drop
    chr2 entirely and still look like a successful run.
    """

    # Arrange
    solo_header = builders.alignment_header({"chr1": TWO_CONTIGS["chr1"]})
    solo = builders.write_bam(
        str(tmp_path / "solo.bam"),
        [
            builders.aligned_read(solo_header, "s{}".format(i), "chr1", 37 * i + 1)
            for i in range(20)
        ],
        solo_header,
    )
    pooled = [bam_pair[0], solo]

    # Act
    whole = pileup_pkg.pileup_alignments(
        pooled,
        reference_fasta,
        str(tmp_path / "w"),
        aggregate=UNSTRANDED,
        contig_groups=1,
    )
    split = pileup_pkg.pileup_alignments(
        pooled,
        reference_fasta,
        str(tmp_path / "s"),
        aggregate=UNSTRANDED,
        contig_groups=2,
    )

    # Assert
    assert digest_track(whole[0]) == digest_track(split[0])
    # chr2 exists only in the first alignment; a naive intersection
    # across the pool would have discarded it.
    assert "chr2.parquet" in digest_track(split[0])


def test_pileup_alignments_should_emit_progress_to_the_package_logger(
    tmp_path, shifted_pair, caplog
):
    """Test that the caller's logging configuration can see the progress.

    Given:
        A caller capturing ``quadrat.pileup`` at INFO, with ``verbose``
        left alone.
    When:
        A pileup runs.
    Then:
        It should have emitted the progress records, so the caller's
        own configuration decides where they go.

    Progress used to go to stdout through `print`, switched by a
    boolean -- which cannot be filtered, redirected per caller or
    captured by the logging a training script already has. `verbose`
    survives as the convenience that wires an output for a caller who
    has configured nothing; it is no longer what decides whether a
    record exists.
    """

    # Arrange
    caplog.set_level("INFO", logger="quadrat.pileup")

    # Act
    pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "logged"),
        aggregate=UNSTRANDED,
    )

    # Assert
    messages = [record.getMessage() for record in caplog.records]
    assert any("piling up from" in message for message in messages)
    assert any("positions to" in message for message in messages)


def test_pileup_alignments_should_leave_the_root_logger_unconfigured(
    tmp_path, shifted_pair
):
    """Test that wiring an output for `verbose` does not reach the root.

    Given:
        A pileup asked to report verbosely.
    When:
        It finishes.
    Then:
        The root logger should carry no handler this package added, and
        the package logger should be back at the level it started at.

    This is the hazard that made `polars_bio.set_loglevel` unusable:
    installing a handler or a level on the root logger on a library's
    own initiative reconfigures the logging of whatever application
    imported it.
    """

    # Arrange
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    package = logging.getLogger("quadrat.pileup")
    package_level = package.level
    package_handlers = list(package.handlers)

    # Act
    pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "verbose-root"),
        aggregate=UNSTRANDED,
        verbose=True,
    )

    # Assert
    assert (root.level, list(root.handlers)) == before
    assert package.level == package_level
    assert list(package.handlers) == package_handlers


def test_pileup_alignments_should_refuse_an_optional_argument_given_positionally(
    tmp_path, shifted_pair
):
    """Test that the optional arguments are keyword-only.

    Given:
        A pileup called with a fourth positional argument.
    When:
        It is invoked.
    Then:
        It should raise `TypeError`.

    Without the ``*`` the order of ten optional parameters becomes part
    of the contract permanently, and the library is pre-1.0 -- the
    asymmetry §0 names, where breaking rework is cheap now and
    permanent later. It is also what made the removal of two refusing
    parameters a free change rather than a breaking one.
    """

    # Act & assert
    with pytest.raises(TypeError):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "positional"), 4
        )


def test_pileup_alignments_should_refuse_a_frame_that_is_not_grouped_by_strand(
    tmp_path, shifted_pair, mocker
):
    """Test that an ungrouped aggregate is refused rather than truncated.

    Given:
        An engine that returns a frame whose strands interleave.
    When:
        A stranded pileup is requested.
    Then:
        It should refuse, naming the run count against the strand
        count.

    Each channel is taken as one contiguous slice, so a frame with more
    than one run per strand would be silently truncated to its last
    run -- with the row count and the manifest both computed from the
    truncated slices, so they agree and `require_complete` passes. The
    precondition holds today, `pileup_sql` ordering by strand first,
    which is why the engine has to be stood in for to reach this at
    all; the sibling writer checks the same precondition fifteen lines
    away and this did not.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    polars = pytest.importorskip("polars")
    interleaved = polars.DataFrame(
        {
            "strand": ["+", "-", "+"],
            "chrom": ["chr1", "chr1", "chr1"],
            "pos": [10, 20, 30],
            "value": [1.0, 1.0, 1.0],
        }
    )
    mocker.patch.object(pb, "sql", return_value=mocker.Mock(collect=lambda: interleaved))

    # Act & assert
    with pytest.raises(ValueError, match="not grouped by strand"):
        pileup_pkg.pileup_alignments(
            shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "ungrouped")
        )


def test_pileup_alignments_should_release_its_tables_when_it_finishes(
    tmp_path, shifted_pair
):
    """Test that a finished pileup leaves nothing registered.

    Given:
        A pooled pileup over two alignments.
    When:
        It finishes.
    Then:
        Neither alignment table nor the chrom-sizes table should still
        be bound in the shared session.

    The names are fixed -- ``quadrat_bam_0`` upward -- so a worker that
    runs a twenty-BAM pooled job and then a one-BAM job would otherwise
    keep nineteen tables bound to the first job's files, and `loky`
    reuses workers across `Parallel` calls.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")

    # Act
    pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "done"),
        aggregate=UNSTRANDED,
    )

    # Assert
    assert not bound(pb, "quadrat_bam_0")
    assert not bound(pb, "quadrat_bam_1")
    assert not bound(pb, "quadrat_chrom_sizes")


def foreign_alignment(path):
    """One alignment whose header names no contig the reference has.

    The route into the contig refusal, which `pileup_alignments` raises
    after the registration -- which is what makes it the case the
    hoisted cleanup exists for.
    """

    header = builders.alignment_header({"chrZ": 800})
    reads = [
        builders.aligned_read(header, "z{}".format(i), "chrZ", 100 + i, "20M", 0)
        for i in range(4)
    ]
    return builders.write_bam(path, reads, header)


def test_pileup_alignments_should_release_its_tables_when_it_refuses(
    reference_fasta, tmp_path
):
    """Test that a pileup refused partway through leaves nothing registered.

    Given:
        An alignment whose header names no reference contig, which the
        pileup refuses only after registering it.
    When:
        The refusal is raised.
    Then:
        The alignment table should not still be bound.

    This is the case the hoist exists for. The refusal happens after
    the registration, so a cleanup that opened below it would release
    nothing on exactly the path that leaks.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    path = foreign_alignment(str(tmp_path / "leak.bam"))

    # Act
    with pytest.raises(ValueError):
        pileup_pkg.pileup_alignments([path], reference_fasta, str(tmp_path / "refused"))

    # Assert
    assert not bound(pb, "quadrat_bam_0")


def test_pileup_alignments_should_restore_the_partition_count_it_found(
    tmp_path, shifted_pair
):
    """Test that the one session option a pileup sets is put back.

    Given:
        A session whose ``target_partitions`` has a known value.
    When:
        A pileup runs at a different count.
    Then:
        The option should hold its original value afterwards.

    The option is process-global and the session is shared, which
    the engine module makes its central hazard and enforces
    for table names. A pileup that left this set would hand the next
    caller in the process whatever parallelism it chose -- the whole
    machine, when the count was defaulted.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    pb.set_option("datafusion.execution.target_partitions", "3")

    # Act
    pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "restored"),
        aggregate=UNSTRANDED,
        target_partitions=2,
    )

    # Assert
    assert str(pb.get_option("datafusion.execution.target_partitions")) == "3"


def test_pileup_alignments_should_restore_the_partition_count_when_it_raises(
    reference_fasta, tmp_path
):
    """Test that a refused pileup still puts the option back.

    Given:
        A session whose ``target_partitions`` has a known value, and a
        pileup that will refuse after setting it.
    When:
        The refusal is raised.
    Then:
        The option should hold its original value afterwards.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    pb.set_option("datafusion.execution.target_partitions", "3")
    path = foreign_alignment(str(tmp_path / "restore.bam"))

    # Act
    with pytest.raises(ValueError):
        pileup_pkg.pileup_alignments(
            [path], reference_fasta, str(tmp_path / "raised"), target_partitions=2
        )

    # Assert
    assert str(pb.get_option("datafusion.execution.target_partitions")) == "3"
