"""Tests for reading alignments and writing cut-site tables.

Ported from cherimoya's ``tests/test_giql_io.py``, which held the whole data path in
one module. The names it reached for as ``giql_io.x`` and
``pileup.x`` are spread over several modules here, so the references
are rewritten and nothing else is: the assertions are the evidence
that the split preserved behaviour, and rewriting them in the same
change that moves them would destroy that evidence.

Style follows cherimoya rather than this project's test conventions --
natural-language names, no Arrange-Act-Assert phases. Converting them
is a separate change, tracked on the epic.
"""

import gzip
import os
import random

import numpy
import pytest

pytest.importorskip("polars_bio")
pytest.importorskip("polars")
pysam = pytest.importorskip("pysam")


from quadrat import _engine, alignments, reference, tables
from quadrat import pileup as pileup_pkg
from quadrat.pileup import sql

os.environ.setdefault("TQDM_DISABLE", "1")

CHROM_SIZES = {"chr1": 1000, "chr2": 500}
POS_SHIFT, NEG_SHIFT = 4, -4
IGNORE = list("QWERYUIOPSDFHJKLZXVBNM")


###
# Fixtures
###


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("giql_io")


@pytest.fixture(scope="module")
def fasta(data_dir):
    rng = random.Random(0)
    path = str(data_dir / "genome.fa")
    with open(path, "w") as f:
        for chrom, size in CHROM_SIZES.items():
            seq = list(rng.choice("ACGT") for _ in range(size))
            # Both stretches sit inside windows the tests keep: the
            # lowercase run under the chr1 800-840 locus, the N run under
            # chr1 100-140 (and inside every wide window over chr2 200-260).
            seq[790:850] = [c.lower() for c in seq[790:850]]
            seq[110:120] = ["N"] * 10
            seq[220:230] = ["N"] * 10
            f.write(">{}\n".format(chrom))
            for i in range(0, size, 60):
                f.write("".join(seq[i : i + 60]) + "\n")
    return path


def _read(name, chrom, pos, cigar, flag, header):
    """Build one aligned segment with a dummy query and qualities."""

    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.flag = flag
    if chrom is not None:
        read.reference_id = header.get_tid(chrom)
        read.reference_start = pos
        read.cigarstring = cigar
    else:
        read.reference_id = -1
        read.reference_start = -1

    qlen = read.infer_query_length() or 20
    read.query_sequence = "A" * qlen
    read.query_qualities = pysam.qualitystring_to_array("I" * qlen)
    read.mapping_quality = 30
    return read


def _write_bam(path, reads, header):
    unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.sort("-o", path, unsorted)
    pysam.index(path)
    os.remove(unsorted)
    return path


@pytest.fixture(scope="module")
def bam_header():
    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [
                {"SN": "chr1", "LN": 1000},
                {"SN": "chr2", "LN": 500},
                {"SN": "chrZ", "LN": 800},
            ],
        }
    )


@pytest.fixture(scope="module")
def bam_pair(data_dir, bam_header):
    h = bam_header
    first = [
        _read("f1", "chr1", 100, "20M", 0, h),
        _read("f1_secondary", "chr1", 100, "20M", 256, h),
        _read("f1_duplicate", "chr1", 100, "20M", 1024, h),
        _read("r1", "chr1", 150, "20M", 16, h),
        _read("f_del", "chr1", 200, "10M5D10M", 0, h),
        _read("r_clip", "chr1", 300, "5S15M", 16, h),
        _read("r_skip", "chr1", 400, "10M100N10M", 16, h),
        _read("placed_unmapped", "chr1", 600, "20M", 4, h),
        _read("unmapped", None, None, None, 4, h),
        _read("on_chrZ", "chrZ", 50, "20M", 0, h),
        _read("c2_f", "chr2", 10, "20M", 0, h),
        _read("c2_r", "chr2", 480, "20M", 16, h),
    ]
    second = [
        _read("f2", "chr1", 700, "20M", 0, h),
        _read("r2", "chr1", 100, "20M", 16, h),
        _read("c2_f_again", "chr2", 10, "20M", 0, h),
    ]
    return [
        _write_bam(str(data_dir / "first.bam"), first, h),
        _write_bam(str(data_dir / "second.bam"), second, h),
    ]


@pytest.fixture(scope="module")
def bam_edges(data_dir, bam_header):
    h = bam_header
    reads = [
        _read("rev_at_origin", "chr1", 0, "3M", 16, h),
        _read("fwd_at_end", "chr1", 998, "2M", 0, h),
        _read("fwd_inside", "chr1", 990, "2M", 0, h),
    ]
    return _write_bam(str(data_dir / "edges.bam"), reads, h)


@pytest.fixture(scope="module")
def peaks_bed(data_dir):
    rows = [
        ("chr2", 200, 260),  # ordinary locus on chr2 (also exercises `chroms`)
        ("chr1", 100, 140),  # ordinary
        ("chr1", 20, 40),  # window start < 0 -> dropped
        ("chr1", 960, 1000),  # window end > chrom length -> dropped
        ("chr1", 955, 975),  # window end == chrom length -> kept (tangermeme >= 1.4)
        ("chr1", 690, 720),  # touches excluded bin 6 only through the +1 quirk
        ("chr1", 800, 840),  # clear of the exclusion zone
        ("chr1", 930, 940),  # ordinary, near the end
        ("chr1", 540, 560),  # inside the exclusion zone
    ]
    path = str(data_dir / "peaks.bed")
    with open(path, "w") as f:
        for chrom, start, end in rows:
            f.write("{}\t{}\t{}\tpeak\t0\t.\n".format(chrom, start, end))
    return path


@pytest.fixture(scope="module")
def exclusion_bed(data_dir):
    path = str(data_dir / "exclusion.bed.gz")
    with gzip.open(path, "wt") as f:
        f.write("chr1\t500\t600\n")
    return path


def _oracle_pileup(bam_paths, chrom_sizes, pos_shift, neg_shift, scale=1.0):
    """bam2bw's loop, densified: every mapped record on a known contig
    contributes its shifted 5' end; out-of-range sites are dropped."""

    dense = {c: numpy.zeros(n, dtype=numpy.float64) for c, n in chrom_sizes.items()}
    for path in bam_paths:
        with pysam.AlignmentFile(path, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.reference_name not in dense:
                    continue
                start = read.reference_start + pos_shift
                end = read.reference_end + neg_shift
                pos = start if read.is_forward else end - 1
                if 0 <= pos < chrom_sizes[read.reference_name]:
                    dense[read.reference_name][pos] += 1
    return {c: v * scale for c, v in dense.items()}


def _dense_from_parquet(path, chrom_sizes):
    import polars

    frame = polars.read_parquet(path)
    dense = {c: numpy.zeros(n, dtype=numpy.float64) for c, n in chrom_sizes.items()}
    for chrom, pos, value in frame.iter_rows():
        dense[chrom][pos] = value
    return dense, frame


@pytest.fixture(scope="module")
def cut_sites(data_dir, bam_pair, fasta):
    _engine.clear_cache()
    return pileup_pkg.pileup_alignments(
        bam_pair,
        fasta,
        str(data_dir / "pair"),
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
    )


###
# The polars-bio BAM provider
###


def test_bam_provider_coordinates_should_match_pysam(bam_pair):
    """GIVEN a BAM with deletions, soft clips and reference skips
    WHEN it is read through polars-bio's BAM provider
    THEN start/end equal pysam's 0-based reference_start/reference_end,
    unmapped records never reach the pileup, and the pinned offset holds."""

    import polars_bio as pb

    pb.register_bam(bam_pair[0], "test_pin")
    frame = (
        pb.sql('SELECT name, chrom, start, "end", flags FROM test_pin')
        .collect()
        .to_pandas()
        .set_index("name")
    )

    with pysam.AlignmentFile(bam_pair[0], "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            row = frame.loc[read.query_name]
            assert row["chrom"] == read.reference_name
            assert (
                int(row["start"]) + alignments._BAM_START_OFFSET == read.reference_start
            ), read.query_name
            assert (
                int(row["end"]) + alignments._BAM_START_OFFSET == read.reference_end
            ), read.query_name

    unmapped = frame[(frame["flags"] & 4) != 0]
    if len(unmapped):
        # The provider surfaces unmapped rows; the pileup SQL must skip them.
        assert {"placed_unmapped", "unmapped"} >= set(unmapped.index)


###
# Cut-site pileup
###


def test_pileup_alignments_should_match_bam2bw_semantics(cut_sites, bam_pair):
    """GIVEN two BAMs with secondary, duplicate, unmapped and off-contig reads
    WHEN they are piled up with the ATAC +4/-4 shift, unstranded
    THEN every position equals a pysam re-implementation of bam2bw's loop
    (all files pooled, only unmapped and unknown-contig reads skipped)."""

    expected = _oracle_pileup(bam_pair, CHROM_SIZES, POS_SHIFT, NEG_SHIFT)
    dense, frame = _dense_from_parquet(cut_sites[0], CHROM_SIZES)

    assert cut_sites == [cut_sites[0]]
    assert cut_sites[0].endswith(tables.CUT_SITE_SUFFIX)
    assert frame.schema["pos"].is_integer()
    assert frame.schema["value"].is_float()
    assert frame["chrom"].to_list() == sorted(frame["chrom"].to_list())
    for chrom in CHROM_SIZES:
        numpy.testing.assert_array_equal(dense[chrom], expected[chrom])

    # Spot checks that pin the semantics independently of the oracle.
    assert dense["chr1"][100 + POS_SHIFT] == 3  # primary + secondary + duplicate
    assert dense["chr1"][170 - 1 + NEG_SHIFT] == 1  # reverse read cuts at end - 1
    assert dense["chr1"][520 - 1 + NEG_SHIFT] == 1  # 10M100N10M spans the skip
    assert dense["chr1"][315 - 1 + NEG_SHIFT] == 1  # 5S15M: clip is not reference
    assert dense["chr2"][10 + POS_SHIFT] == 2  # pooled across files
    assert dense["chr1"][600 + POS_SHIFT] == 0  # placed unmapped skipped
    assert frame["value"].sum() == 12  # 9 records in file one + 3 in file two


def test_pileup_alignments_should_apply_scale_factor(data_dir, bam_pair, fasta):
    """GIVEN a scale factor of 0.5
    WHEN the same BAMs are piled up
    THEN every value is halved and stays a float."""

    paths = pileup_pkg.pileup_alignments(
        bam_pair,
        fasta,
        str(data_dir / "scaled"),
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
        scale_factor=0.5,
    )
    expected = _oracle_pileup(bam_pair, CHROM_SIZES, POS_SHIFT, NEG_SHIFT, 0.5)
    dense, _ = _dense_from_parquet(paths[0], CHROM_SIZES)
    for chrom in CHROM_SIZES:
        numpy.testing.assert_array_equal(dense[chrom], expected[chrom])


def test_pileup_alignments_should_split_strands(data_dir, bam_pair, fasta):
    """GIVEN stranded output
    WHEN the BAMs are piled up
    THEN the + table holds forward cut sites, the - table reverse ones,
    and together they sum to the unstranded pileup."""

    paths = pileup_pkg.pileup_alignments(
        bam_pair,
        fasta,
        str(data_dir / "stranded"),
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=False,
    )
    assert paths == [
        str(data_dir / "stranded.+.cuts.parquet"),
        str(data_dir / "stranded.-.cuts.parquet"),
    ]

    plus, _ = _dense_from_parquet(paths[0], CHROM_SIZES)
    minus, _ = _dense_from_parquet(paths[1], CHROM_SIZES)
    both = _oracle_pileup(bam_pair, CHROM_SIZES, POS_SHIFT, NEG_SHIFT)
    assert plus["chr1"][100 + POS_SHIFT] == 3 and minus["chr1"][104] == 0
    assert minus["chr1"][170 - 1 + NEG_SHIFT] == 1 and plus["chr1"][165] == 0
    for chrom in CHROM_SIZES:
        numpy.testing.assert_array_equal(plus[chrom] + minus[chrom], both[chrom])


def test_pileup_alignments_should_drop_out_of_range_sites(data_dir, bam_edges, fasta):
    """GIVEN reads whose shifted cut sites fall below 0 or past the end
    WHEN they are piled up
    THEN only the in-range site survives."""

    paths = pileup_pkg.pileup_alignments(
        [bam_edges],
        fasta,
        str(data_dir / "edges"),
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
    )
    dense, frame = _dense_from_parquet(paths[0], CHROM_SIZES)
    assert frame.height == 1
    assert dense["chr1"][990 + POS_SHIFT] == 1


def test_pileup_alignments_should_reject_unsupported_modes(bam_pair, fasta, data_dir):
    """GIVEN fragments or read-depth normalization
    WHEN a pileup is requested
    THEN the engine refuses rather than silently diverging from bam2bw."""

    with pytest.raises(NotImplementedError):
        pileup_pkg.pileup_alignments(
            bam_pair, fasta, str(data_dir / "x"), fragments=True
        )
    with pytest.raises(NotImplementedError):
        pileup_pkg.pileup_alignments(
            bam_pair, fasta, str(data_dir / "x"), read_depth=True
        )


###
# Locus extraction
###


def _reference_extract(fasta, bam_pair, **kwargs):
    """tangermeme's extract_loci over the oracle pileup (dict signals)."""

    from tangermeme.io import extract_loci

    dense = _oracle_pileup(bam_pair, CHROM_SIZES, POS_SHIFT, NEG_SHIFT)
    dense = {c: v.astype(numpy.float32) for c, v in dense.items()}
    return extract_loci(sequences=fasta, signals=[dense], **kwargs)


def test_cut_site_contigs_lists_a_directory_from_its_filenames(cut_sites):
    """GIVEN a written cut-site table
    WHEN its contigs are listed
    THEN they are the contigs it holds rows for, and no others.

    `chrZ` is in the BAM header but not in the reference, so no row
    survived the pileup and no file was written for it."""

    assert tables.cut_site_contigs(cut_sites[0]) == {"chr1", "chr2"}


def _plan_ancestor(plan, parent, child):
    """True when every ``child`` node sits under a ``parent`` node.

    DataFusion prints a physical plan depth first, parents before
    children and two spaces of indent per level, so one node is an
    ancestor of another exactly when it appears earlier and is less
    indented."""

    def nodes(marker):
        return [
            (i, len(line) - len(line.lstrip(" ")))
            for i, line in enumerate(plan.splitlines())
            if marker in line
        ]

    parents, children = nodes(parent), nodes(child)
    if not parents or not children:
        return False
    return all(any(pi < ci and pd < cd for pi, pd in parents) for ci, cd in children)


def test_pileup_should_bound_positions_above_the_aggregate(bam_pair, fasta):
    """GIVEN the cut-site pileup query
    WHEN polars-bio explains it
    THEN the chromosome-sizes join takes the grouped result as its input
    rather than one row per read."""

    import polars
    import polars_bio as pb

    tables = alignments.register_alignments(bam_pair)
    lengths = reference.chrom_lengths(fasta)
    _engine._register_frame(
        pb,
        _engine._SIZES_TABLE,
        polars.DataFrame(
            {
                "chrom": list(lengths.keys()),
                "length": list(lengths.values()),
            },
            schema={"chrom": polars.Utf8, "length": polars.Int64},
        ),
    )

    plan = pb.sql("EXPLAIN " + sql.pileup_sql(tables)).collect()
    physical = next(
        row["plan"]
        for row in plan.iter_rows(named=True)
        if row["plan_type"] == "physical_plan"
    )

    assert "JoinExec" in physical and "AggregateExec" in physical

    # Above, not below: the join probes the grouped rows rather than one
    # per read. Below is what overflowed Arrow's 32-bit string offsets at
    # 291,063,148 reads, because the join collects one side whole and
    # that side was the reads.
    assert _plan_ancestor(physical, "JoinExec", "AggregateExec")
    assert not _plan_ancestor(physical, "AggregateExec", "JoinExec")
