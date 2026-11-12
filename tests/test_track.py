"""Tests for turning alignments into tracks.

Relocated from the ported ``tests/test_pileup_io.py`` and
``tests/test_pileup.py``, assertions unchanged.

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

import logging
import os
import pathlib
import threading
import warnings

import numpy
import pytest

from quadrat import EngineOptionWarning, tables
from quadrat import pileup as pileup_pkg
from quadrat.pileup import track
from tests import TWO_CONTIGS, bound, builders, digest_track

POS_SHIFT, NEG_SHIFT = 4, -4


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


def oracle_pileup(bam_paths, chrom_sizes, pysam, pos_shift, neg_shift, scale=1.0):
    """bam2bw's loop, densified.

    Every mapped record on a known contig contributes its shifted 5'
    end; out-of-range sites are dropped.
    """

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
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
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
        unstranded=True,
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
        unstranded=True,
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
        pileup_pkg.pileup_alignments(bam_pair, reference_fasta, 42, unstranded=True)
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
        bam_pair[0], reference_fasta, name, unstranded=True
    )

    # Assert
    assert written == [name + tables.TRACK_SUFFIX]
    assert tables.track_contigs(written[0]) <= set(TWO_CONTIGS)


def test_pileup_alignments_should_match_bam2bw_semantics(track_paths, bam_pair, pysam):
    """Test the position every record contributes.

    Given:
        Two BAMs carrying secondary, duplicate, unmapped and off-contig
        reads.
    When:
        They are piled up with the ATAC +4/-4 shift, unstranded.
    Then:
        It should match a pysam re-implementation of bam2bw's loop at
        every position, pooling all files and skipping only the unmapped
        and unknown-contig reads.
    """

    # Arrange
    expected = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, POS_SHIFT, NEG_SHIFT)

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
    assert dense["chr1"][100 + POS_SHIFT] == 3  # primary + secondary + duplicate
    assert dense["chr1"][170 - 1 + NEG_SHIFT] == 1  # reverse read cuts at end - 1
    assert dense["chr1"][520 - 1 + NEG_SHIFT] == 1  # 10M100N10M spans the skip
    assert dense["chr1"][315 - 1 + NEG_SHIFT] == 1  # 5S15M: clip is not reference
    assert dense["chr2"][10 + POS_SHIFT] == 2  # pooled across files
    assert dense["chr1"][600 + POS_SHIFT] == 0  # placed unmapped skipped
    assert frame["value"].sum() == 12  # 9 records in file one + 3 in file two


def test_pileup_alignments_should_apply_the_scale_factor(
    tmp_path, bam_pair, reference_fasta, pysam
):
    """Test the one multiplication the aggregate applies.

    Given:
        A scale factor of one half.
    When:
        The same BAMs are piled up.
    Then:
        It should halve every value and keep it a float.
    """

    # Act
    paths = pileup_pkg.pileup_alignments(
        bam_pair,
        reference_fasta,
        str(tmp_path / "scaled"),
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
        scale_factor=0.5,
    )

    # Assert
    expected = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, POS_SHIFT, NEG_SHIFT, 0.5)
    dense, _ = dense_from_parquet(paths[0], TWO_CONTIGS)
    for chrom in TWO_CONTIGS:
        numpy.testing.assert_array_equal(dense[chrom], expected[chrom])


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
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=False,
    )

    # Assert
    assert paths == [
        str(tmp_path / ("stranded.+" + tables.TRACK_SUFFIX)),
        str(tmp_path / ("stranded.-" + tables.TRACK_SUFFIX)),
    ]

    plus, _ = dense_from_parquet(paths[0], TWO_CONTIGS)
    minus, _ = dense_from_parquet(paths[1], TWO_CONTIGS)
    both = oracle_pileup(bam_pair, TWO_CONTIGS, pysam, POS_SHIFT, NEG_SHIFT)
    assert plus["chr1"][100 + POS_SHIFT] == 3 and minus["chr1"][104] == 0
    assert minus["chr1"][170 - 1 + NEG_SHIFT] == 1 and plus["chr1"][165] == 0
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
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        unstranded=True,
    )

    # Assert
    dense, frame = dense_from_parquet(paths[0], TWO_CONTIGS)
    assert frame.height == 1
    assert dense["chr1"][990 + POS_SHIFT] == 1


def test_pileup_alignments_should_shift_only_the_forward_site_by_pos_shift(
    tmp_path, shifted_pair
):
    """Test that the forward shift moves the forward site alone.

    Given:
        One forward and one reverse read at known positions, and a
        non-zero ``pos_shift`` with ``neg_shift`` left at its default.
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
        unstranded=True,
    )

    # Act
    shifted = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "fwd"),
        unstranded=True,
        pos_shift=7,
    )

    # Assert
    assert positions(os.path.join(unshifted[0], "chr1.parquet")) == [100, 319]
    assert positions(os.path.join(shifted[0], "chr1.parquet")) == [107, 319]


def test_pileup_alignments_should_shift_only_the_reverse_site_by_neg_shift(
    tmp_path, shifted_pair
):
    """Test that the reverse shift moves the reverse site alone.

    Given:
        The same reads, and a non-zero ``neg_shift`` with ``pos_shift``
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
        unstranded=True,
        neg_shift=-9,
    )

    # Assert
    assert positions(os.path.join(shifted[0], "chr1.parquet")) == [100, 310]


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
        shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "chosen"), unstranded=True
    )
    named = pileup_pkg.pileup_alignments(
        shifted_pair.bams,
        shifted_pair.fasta,
        str(tmp_path / "named"),
        unstranded=True,
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
        shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "ok"), unstranded=True
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
            unstranded=True,
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
            unstranded=True,
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
            unstranded=True,
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
            unstranded=True,
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
        unstranded=True,
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
        unstranded=True,
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
                bam_pair, reference_fasta, str(tmp_path / label), unstranded=True
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
        bam_pair, reference_fasta, str(tmp_path / "after"), unstranded=True
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
        corpus.bams, corpus.fasta, str(tmp_path / "grouped"), unstranded=True
    )
    metadata = pq.ParquetFile(tables.track_chrom_path(written[0], "chr1")).metadata
    groups = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]

    # Assert
    assert metadata.num_rows > 10_000, "the corpus must outgrow one group to test this"
    full, remainder = divmod(metadata.num_rows, 10_000)
    assert groups == [10_000] * full + ([remainder] if remainder else [])


@pytest.mark.parametrize("groups", [2, 4])
def test_pileup_alignments_should_write_the_same_table_at_any_batch_count(
    tmp_path, bam_pair, reference_fasta, groups
):
    """Test that contig batching changes nothing it writes.

    Given:
        A pileup run as one aggregate and as several contig batches.
    When:
        The written tables are hashed.
    Then:
        It should write byte-for-byte identical Parquet, with the same
        row totals.

    It holds because batches partition the contigs, and a position
    belongs to exactly one contig, so no count can straddle a batch
    boundary.
    """

    # Act
    one = pileup_pkg.pileup_alignments(
        [bam_pair[0]],
        reference_fasta,
        str(tmp_path / "whole"),
        unstranded=True,
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
        contig_groups=1,
    )
    many = pileup_pkg.pileup_alignments(
        [bam_pair[0]],
        reference_fasta,
        str(tmp_path / "split"),
        unstranded=True,
        pos_shift=POS_SHIFT,
        neg_shift=NEG_SHIFT,
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
        [bam_pair[0]], extra, str(tmp_path / "w"), unstranded=True, contig_groups=1
    )
    split = pileup_pkg.pileup_alignments(
        [bam_pair[0]], extra, str(tmp_path / "s"), unstranded=True, contig_groups=3
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
        pooled, reference_fasta, str(tmp_path / "w"), unstranded=True, contig_groups=1
    )
    split = pileup_pkg.pileup_alignments(
        pooled, reference_fasta, str(tmp_path / "s"), unstranded=True, contig_groups=2
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
        shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "logged"), unstranded=True
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
        unstranded=True,
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
        shifted_pair.bams, shifted_pair.fasta, str(tmp_path / "done"), unstranded=True
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
        unstranded=True,
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
