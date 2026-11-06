"""Tests for reading alignments: contigs, indexes, and the refusals.

Relocated from the ported ``tests/test_pileup.py`` and
``tests/test_pileup_io.py``, assertions unchanged, split where one body
held two behaviours. `register_alignments` is new: it had no test at
all, although it is the function that puts a BAM into the session under
the ``quadrat_`` prefix that keeps this package's tables off
cherimoya's.

Two reseats. The provider-parity test compared polars-bio's coordinates
against pysam's through the private ``BAM_START_OFFSET``, which made
the constant the subject; it now compares the two conventions directly,
so a provider that changed its base still fails. And the index-naming
test called the private ``_has_bam_index``; it now asks whether the
``samtools index`` warning fires, which is the behaviour that helper
exists to drive and which also pins the reason -- polars-bio cannot
restrict a scan with a CSI, so treating one as an index would promise a
pushdown that does not exist.

The contig-naming refusals are found against a corpus of deliberately
defective files rather than against the synthetic BAMs the rest of this
module uses, which are all indexed and all named to match their
reference. Each of those failures used to be silent, so the tests for
them are about the noise.
"""

import logging
import os
import pathlib
import shutil
import warnings

import pytest

from quadrat import alignments, tables
from quadrat import pileup as pileup_pkg
from tests import TWO_CONTIGS, builders, digest_track


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Two small alignments over two contigs, with an indexed reference."""

    pytest.importorskip("polars_bio")
    return corpus_factory("alignments", TWO_CONTIGS, n_bams=2, index_fasta=True)


def one_contig_bam(path, chrom="chr1", size=1000, reads=20):
    """A coordinate-sorted BAM declaring one contig, named as asked."""

    header = builders.alignment_header({chrom: size})
    return builders.write_bam(
        path,
        [
            builders.aligned_read(
                header,
                "solo{}".format(i),
                chrom,
                (i * 37) % (size - 40),
                flag=16 if i % 2 else 0,
            )
            for i in range(reads)
        ],
        header,
    )


def test_bam_contigs_should_report_the_header_pysam_reads_in_header_order(corpus, pysam):
    """Test that the stdlib header read agrees with pysam's.

    Given:
        A BAM and the header pysam reads from it.
    When:
        Its contigs are read without pysam.
    Then:
        It should report the same names at the same lengths, in the
        header's own order.

    The order matters rather than only the mapping: contig batches are
    built by walking what this returns, so two readers disagreeing about
    the order would disagree about which contigs land in which batch.
    """

    # Arrange
    with pysam.AlignmentFile(corpus.bams[0], "rb") as handle:
        expected = dict(zip(handle.references, handle.lengths))

    # Act
    got = alignments.bam_contigs(corpus.bams[0])

    # Assert
    assert got == expected
    assert list(got.items()) == list(expected.items())


def test_bam_contigs_should_accept_a_bytes_path(corpus):
    """Test that a bytes alignment is read rather than declined.

    Given:
        A BAM named by a ``bytes`` path.
    When:
        Its contigs are read.
    Then:
        It should report what the equivalent string reports.

    The suffix test ran on ``str(path)``, and ``str(b"/data/x.bam")``
    is ``"b'/data/x.bam'"`` -- which does not end in ``.bam``, so this
    answered ``None`` as though the header were unreadable. Both pileup
    entry points admit a ``bytes`` path, so the shape reaches here, and
    `require_reference_contigs` reads the ``None`` as a header to
    exempt: criterion #4's refusal was waived for it.
    """

    # Act & assert
    assert alignments.bam_contigs(os.fsencode(corpus.bams[0])) == (
        alignments.bam_contigs(corpus.bams[0])
    )


def test_bam_contigs_should_raise_type_error_for_something_that_is_not_a_path():
    """Test that a wrong type is a caller error rather than a shrug.

    Given:
        An integer where a BAM path belongs.
    When:
        Its contigs are read.
    Then:
        It should raise `TypeError` naming the function and the
        parameter.

    ``None`` is this function's answer for a header it could not read,
    and `require_reference_contigs` acts on it by exempting that
    alignment -- so answering ``None`` to a wrong type waived
    criterion #4's refusal for a caller mistake as well as for an
    unreadable file. The documented causes for ``None`` are all "there
    is no header here to read"; an integer is not one of them.
    """

    # Act & assert
    with pytest.raises(TypeError, match="bam_contigs"):
        alignments.bam_contigs(42)


def test_bam_contigs_should_accept_a_path_object(corpus):
    """Test that a path-like alignment is read rather than declined.

    Given:
        A BAM named by a `pathlib.Path`.
    When:
        Its contigs are read.
    Then:
        It should report what the equivalent string reports.

    Pinned because the recorded finding says this returns ``None``, and
    it does not -- the suffix test runs on ``str(path)``. A refactor to
    a bare ``path.endswith``, which `register_alignments` did have,
    would regress it silently into a BAM that reads as having no header.
    """

    # Act & assert
    assert alignments.bam_contigs(pathlib.Path(corpus.bams[0])) == (
        alignments.bam_contigs(corpus.bams[0])
    )


def test_bam_contigs_should_return_none_when_the_path_is_not_a_bam(tmp_path):
    """Test the decline for a format with no index to restrict.

    Given:
        A SAM path.
    When:
        Its contigs are read.
    Then:
        It should return ``None``, because SAM carries no index for a
        contig predicate to reach.
    """

    # Act & assert
    assert alignments.bam_contigs(str(tmp_path / "reads.sam")) is None


def test_bam_contigs_should_return_none_when_the_bam_is_missing(tmp_path):
    """Test the decline for a file that is not there.

    Given:
        A ``.bam`` path naming no file.
    When:
        Its contigs are read.
    Then:
        It should return ``None`` rather than raise.

    Reading the header is an optimisation, not a validation step. The
    engine opens the same file moments later and reports the problem
    better than a gzip traceback would, and a header that cannot be read
    only means the pileup is not split.
    """

    # Act & assert
    assert alignments.bam_contigs(str(tmp_path / "missing.bam")) is None


def test_bam_contigs_should_return_none_when_the_file_is_not_gzip(tmp_path):
    """Test the decline for bytes that are no kind of BAM.

    Given:
        A ``.bam`` holding bytes that are not gzip-framed at all.
    When:
        Its contigs are read.
    Then:
        It should return ``None`` rather than raise.
    """

    # Arrange
    junk = tmp_path / "junk.bam"
    junk.write_bytes(b"not actually a bam")

    # Act & assert
    assert alignments.bam_contigs(str(junk)) is None


def test_register_alignments_should_name_each_table_from_the_quadrat_prefix(corpus):
    """Test the naming contract the session depends on.

    Given:
        Two BAM paths.
    When:
        They are registered.
    Then:
        It should return one prefixed name per path, in order.

    polars-bio holds a single process-global session, so the prefix is
    the whole of what keeps these tables from colliding with cherimoya's
    ``cherimoya_``-named ones in a process that imports both.
    """

    # Act
    names = alignments.register_alignments(corpus.bams)

    # Assert
    assert names == ["quadrat_bam_0", "quadrat_bam_1"]
    assert all(name.startswith("quadrat_") for name in names)


def test_register_alignments_should_make_each_name_a_queryable_table(corpus, pysam):
    """Test that registration is what the returned names describe.

    Given:
        Two BAMs of a known record count.
    When:
        They are registered and each returned name is queried.
    Then:
        It should make every name a table holding that BAM's records.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    expected = []
    for path in corpus.bams:
        with pysam.AlignmentFile(path, "rb") as handle:
            expected.append(sum(1 for _ in handle.fetch(until_eof=True)))

    # Act
    names = alignments.register_alignments(corpus.bams)

    # Assert
    for name, count in zip(names, expected):
        rows = pb.sql("SELECT count(*) AS n FROM {}".format(name)).collect()
        assert rows["n"].to_list() == [count]


def test_register_alignments_should_not_let_a_caller_name_the_tables(corpus):
    """Test that the table-name prefix is not the caller's to choose.

    Given:
        A BAM and a prefix of the caller's choosing.
    When:
        It is passed.
    Then:
        It should be refused as an unknown argument.

    This replaces a test that asserted the opposite. `quadrat._engine`
    states the ``quadrat_`` prefix as a property of the package -- the
    mitigation for a process-global session shared with cherimoya's
    ``cherimoya_``-prefixed shapes -- and a ``prefix`` parameter made
    that a default a caller could undo, registering exactly the
    colliding names it exists to prevent. No caller ever passed it.
    """

    # Act & assert
    with pytest.raises(TypeError, match="prefix"):
        alignments.register_alignments(corpus.bams[:1], prefix="cherimoya_bam")


def test_register_alignments_should_register_a_sam_as_text(corpus, pysam):
    """Test the text branch of the register call.

    Given:
        A SAM rendering of a BAM.
    When:
        It is registered and queried.
    Then:
        It should be queryable, which the BAM registration could not
        have made it.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    sam = os.path.join(os.path.dirname(corpus.bams[0]), "as_text.sam")
    with open(sam, "w") as handle:
        handle.write(pysam.view("-h", corpus.bams[0]))

    # Act
    names = alignments.register_alignments([sam])

    # Assert
    rows = pb.sql("SELECT count(*) AS n FROM {}".format(names[0])).collect()
    assert rows["n"].to_list()[0] > 0


def test_register_alignments_should_raise_when_a_path_is_not_bam_or_sam():
    """Test the refusal of a format the engine does not convert.

    Given:
        A path whose suffix is neither ``.bam`` nor ``.sam``.
    When:
        It is registered.
    Then:
        It should raise naming the path, before any file is opened --
        which is why this needs no file to exist.
    """

    # Act & assert
    with pytest.raises(ValueError, match="BAM/SAM"):
        alignments.register_alignments(["reads.cram"])


def test_register_alignments_should_bind_nothing_when_one_suffix_is_wrong(corpus, pysam):
    """Test that a refusal leaves the shared session as it found it.

    Given:
        A list whose first path is a real BAM and whose second has a
        suffix the engine does not convert.
    When:
        They are registered.
    Then:
        It should refuse without binding the first, so the table name
        the first would have taken is unqueryable afterwards.

    The names are fixed and the session is process-global, and this
    function returns the names its caller installs the cleanup
    callback with -- so a refusal that returned nothing left the first
    path bound with nothing to unwind it. `loky` reuses workers across
    calls, which makes that the life of the worker: a stranded provider
    holding a file handle for every arity ever attempted.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    name = "{}_0".format(alignments.BAM_TABLE_PREFIX)
    alignments.deregister_alignments([name])

    # Act
    with pytest.raises(ValueError, match="BAM/SAM"):
        alignments.register_alignments([corpus.bams[0], "reads.cram"])

    # Assert
    with pytest.raises(Exception):
        pb.sql("SELECT * FROM {} LIMIT 1".format(name)).collect()


def test_register_alignments_should_unwind_when_the_provider_refuses_a_file(
    corpus, mocker, pysam
):
    """Test the window the first pass cannot cover.

    Given:
        Two real BAM paths, with the provider raising on the second.
    When:
        They are registered.
    Then:
        It should release the first before propagating, so no name is
        left bound.

    A wrong suffix is a caller mistake the vetting pass refuses up
    front; a missing or unreadable file is refused by the provider,
    in the middle of the registration loop, after earlier names are
    bound. The second case is the one that needs the unwind.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    released = []
    mocker.patch.object(
        alignments, "deregister_alignments", lambda names: released.extend(names)
    )
    real = alignments.engine()
    mocker.patch.object(
        real, "register_bam", side_effect=[None, OSError("no such file")]
    )

    # Act
    with pytest.raises(OSError, match="no such file"):
        alignments.register_alignments([corpus.bams[0], corpus.bams[0]])

    # Assert
    assert released == ["{}_0".format(alignments.BAM_TABLE_PREFIX)]


def test_register_alignments_should_return_no_names_when_given_no_paths():
    """Test the empty case.

    Given:
        No paths at all.
    When:
        They are registered.
    Then:
        It should return no names and register nothing.
    """

    # Act & assert
    assert alignments.register_alignments([]) == []


def test_register_alignments_should_accept_a_path_object(corpus):
    """Test that a path-like alignment is registered rather than crashing.

    Given:
        A BAM path given as a `pathlib.Path`.
    When:
        It is registered.
    Then:
        It should return the same names the equivalent string returns.

    The suffix test ran before any normalization, so a `pathlib.Path`
    raised `AttributeError` from inside the loop -- neither accepting
    the path nor refusing it, and naming no path in the message.
    """

    # Act & assert
    assert alignments.register_alignments([pathlib.Path(corpus.bams[0])]) == [
        "quadrat_bam_0"
    ]


def test_register_alignments_should_accept_a_bytes_path(corpus):
    """Test that a bytes alignment is registered rather than refused.

    Given:
        A BAM named by a ``bytes`` path.
    When:
        It is registered.
    Then:
        It should bind the first table name.

    An ``isinstance(str)`` guard sat beside the suffix test, so a
    ``bytes`` path was refused with a message about file format -- "The
    GIQL engine converts BAM/SAM alignments" -- for what was a path
    type question. The suffix test carries the refusal on its own now,
    because the normalizer has already answered ``None`` for
    everything that is not a path.
    """

    # Act & assert
    assert alignments.register_alignments([os.fsencode(corpus.bams[0])]) == [
        "quadrat_bam_0"
    ]


@pytest.fixture
def root_logger():
    """Hold the root logger at a level of its own, and put it back."""

    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)

    # ERROR because no argument `polars_bio.set_loglevel` accepts
    # produces it. It takes `debug`, `info` and `warn`/`warning`, which
    # set 10, 20 and 30, so a library that reached for any of them
    # lowers this and the assertion sees it. Under pytest the root
    # logger already sits at WARNING, so a test that only recorded
    # whatever was there would watch `set_loglevel("warn")` change
    # nothing and pass.
    root.setLevel(logging.ERROR)
    yield root
    root.setLevel(level)
    root.handlers[:] = handlers


def test_register_alignments_should_leave_the_root_logger_alone(corpus, root_logger):
    """Test that reaching the engine does not reconfigure host logging.

    Given:
        An application whose root logger is configured its own way.
    When:
        An alignment is registered, which is what first imports
        polars-bio.
    Then:
        It should leave the root logger's level and handlers exactly as
        it found them.

    `register_alignments` is the cheapest public route to the engine
    accessor, which is where this package sets its one quiet default.

    This passes against the code that preceded it, and deliberately so
    -- the point is the fix it forbids. That accessor used to call
    ``polars_bio.set_loglevel("error")`` behind a bare ``except
    Exception``, and ``"error"`` is not a level that function accepts,
    so the call raised and was swallowed and the root logger was spared
    by accident. The obvious repair is to pass a level it does accept,
    and that is the regression this watches for: ``set_loglevel`` sets
    the root logger and calls `logging.basicConfig`, so a library
    calling it takes over the logging of the application that imported
    it.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    handlers = list(root_logger.handlers)

    # Act
    alignments.register_alignments(corpus.bams)

    # Assert
    assert root_logger.level == logging.ERROR
    assert list(root_logger.handlers) == handlers


def test_bam_provider_should_report_the_coordinates_pysam_reports(corpus, pysam):
    """Test the coordinate convention the pileup is built on.

    Given:
        A BAM with deletions, soft clips and reference skips.
    When:
        It is read through polars-bio's BAM provider.
    Then:
        It should give each record the 0-based start and exclusive end
        pysam gives it, and surface unmapped records for the pileup's
        own SQL to skip.

    polars-bio's docstring claims 1-based coordinates and the default
    session does not deliver them. This compares the two conventions
    directly, so a release that changes the base fails here rather than
    silently shifting every position by one.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    reads = [
        ("f_del", "chr1", 200, "10M5D10M", 0),
        ("r_clip", "chr1", 300, "5S15M", 16),
        ("r_skip", "chr1", 400, "10M100N10M", 16),
        ("placed_unmapped", "chr1", 600, "20M", 4),
    ]
    header = builders.alignment_header(dict(TWO_CONTIGS))
    path = builders.write_bam(
        os.path.join(os.path.dirname(corpus.bams[0]), "provider.bam"),
        [builders.aligned_read(header, *read) for read in reads],
        header,
    )
    pb.register_bam(path, "quadrat_provider_probe")

    # Act
    frame = (
        pb.sql('SELECT name, chrom, start, "end", flags FROM quadrat_provider_probe')
        .collect()
        .to_pandas()
        .set_index("name")
    )

    # Assert
    with pysam.AlignmentFile(path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                continue
            row = frame.loc[read.query_name]
            assert row["chrom"] == read.reference_name
            assert int(row["start"]) == read.reference_start, read.query_name
            assert int(row["end"]) == read.reference_end, read.query_name

    unmapped = frame[(frame["flags"] & 4) != 0]
    assert set(unmapped.index) == {"placed_unmapped"}


@pytest.mark.parametrize("groups", [1, 4])
def test_pileup_alignments_should_raise_when_the_reference_shares_no_contig(
    corpus, tmp_path, groups
):
    """Test the refusal of a reference the alignment cannot match.

    Given:
        An Ensembl-headered alignment and a UCSC-named reference.
    When:
        A pileup is requested.
    Then:
        It should raise rather than write an empty table, at one contig
        batch as well as several.

    Every record would be dropped by the join against the chrom sizes,
    leaving a table with no rows that `require_complete` passes, since a
    manifest listing no files is one whose files all exist. Training
    then reads a genome with no signal in it.
    """

    # Arrange
    solo = one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")
    name = str(tmp_path / "out")

    # Act & assert
    with pytest.raises(ValueError, match="would write an empty table"):
        pileup_pkg.pileup_alignments(
            [solo], corpus.fasta, name, unstranded=True, contig_groups=groups
        )

    assert not os.path.exists(name + tables.TRACK_SUFFIX)


def test_pileup_alignments_should_name_the_alignment_and_both_conventions(
    corpus, tmp_path
):
    """Test what the contig-mismatch refusal says.

    Given:
        A contig naming mismatch.
    When:
        The pileup refuses.
    Then:
        It should name the file and a contig from either side.

    The cause is almost never a wrong genome, it is a convention, and
    the two are told apart by seeing ``1`` beside ``chr1``.
    """

    # Arrange
    solo = one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")

    # Act
    with pytest.raises(ValueError) as error:
        pileup_pkg.pileup_alignments(
            [solo], corpus.fasta, str(tmp_path / "out"), unstranded=True
        )

    # Assert
    assert "ensembl.bam" in str(error.value)
    assert "chr1" in str(error.value)


def test_pileup_alignments_should_raise_when_one_pooled_alignment_cannot_match(
    corpus, tmp_path
):
    """Test the quiet half of the contig check.

    Given:
        A channel pooling a matching alignment with a mismatched one.
    When:
        A pileup is requested.
    Then:
        It should raise rather than count only the matching one.

    This is the case the check exists for. The other branch keeps the
    batches non-empty, so nothing looks wrong: the run succeeds, the
    table verifies, and one alignment's records are simply absent.
    """

    # Arrange
    solo = one_contig_bam(str(tmp_path / "ensembl.bam"), chrom="1")

    # Act & assert
    with pytest.raises(ValueError, match="ensembl.bam"):
        pileup_pkg.pileup_alignments(
            [corpus.bams[0], solo],
            corpus.fasta,
            str(tmp_path / "out"),
            unstranded=True,
            contig_groups=2,
        )


def test_pileup_alignments_should_let_the_engine_report_an_unreadable_header(
    corpus, tmp_path
):
    """Test that an unopenable file is not diagnosed as a mismatch.

    Given:
        An alignment that does not exist.
    When:
        A pileup is requested.
    Then:
        It should report the open failure rather than the contig check.

    Diagnosing an unreadable file belongs to the component that opens
    it. Claiming a naming mismatch for a missing BAM would be worse than
    saying nothing, and it is what pins `bam_contigs` returning ``None``
    rather than raising.
    """

    # Act
    with pytest.raises(Exception) as error:
        pileup_pkg.pileup_alignments(
            [str(tmp_path / "missing.bam")],
            corpus.fasta,
            str(tmp_path / "out"),
            unstranded=True,
        )

    # Assert
    assert "would write an empty table" not in str(error.value)


def test_pileup_alignments_should_warn_and_still_match_without_an_index(
    corpus, tmp_path
):
    """Test what an unindexed split costs and does not change.

    Given:
        An alignment with no ``.bai`` beside it.
    When:
        It is piled up in several contig batches.
    Then:
        It should warn what the scans cost and write the same table the
        indexed run writes.

    Without an index a contig predicate is still an ordinary filter, so
    the rows are identical; it is only no longer free. The warning must
    report that and change nothing else.
    """

    # Arrange
    bare = str(tmp_path / "bare.bam")
    shutil.copy(corpus.bams[0], bare)

    # Act
    indexed = pileup_pkg.pileup_alignments(
        [corpus.bams[0]],
        corpus.fasta,
        str(tmp_path / "i"),
        unstranded=True,
        contig_groups=2,
    )
    with pytest.warns(UserWarning, match="samtools index"):
        split = pileup_pkg.pileup_alignments(
            [bare],
            corpus.fasta,
            str(tmp_path / "b"),
            unstranded=True,
            contig_groups=2,
        )

    # Assert
    assert digest_track(indexed[0]) == digest_track(split[0])


def test_pileup_alignments_should_not_warn_when_the_alignment_is_indexed(
    corpus, tmp_path
):
    """Test that the common case stays quiet.

    Given:
        An alignment with its index beside it.
    When:
        It is piled up in several contig batches.
    Then:
        It should warn about nothing, the batches being nearly free.
    """

    # Act & assert
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pileup_pkg.pileup_alignments(
            [corpus.bams[0]],
            corpus.fasta,
            str(tmp_path / "ok"),
            unstranded=True,
            contig_groups=2,
        )


def test_pileup_alignments_should_not_warn_when_an_unindexed_run_is_not_split(
    corpus, tmp_path
):
    """Test that a single batch has nothing to report.

    Given:
        An alignment with no index.
    When:
        It is piled up as a single batch.
    Then:
        It should warn about nothing.

    One batch is one scan whether or not an index could have restricted
    it, so there is nothing to report and no reason to make the common
    case noisy.
    """

    # Arrange
    bare = str(tmp_path / "bare.bam")
    shutil.copy(corpus.bams[0], bare)

    # Act & assert
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pileup_pkg.pileup_alignments(
            [bare],
            corpus.fasta,
            str(tmp_path / "one"),
            unstranded=True,
            contig_groups=1,
        )


@pytest.mark.parametrize("spelling", ["x.bam.bai", "x.bai"])
def test_pileup_alignments_should_not_warn_for_either_htslib_index_spelling(
    corpus, tmp_path, spelling
):
    """Test that both index names htslib accepts count as an index.

    Given:
        An alignment whose index is spelled one of the two ways htslib
        accepts.
    When:
        It is piled up in several contig batches.
    Then:
        It should warn about nothing, both spellings letting the contig
        predicate reach the index.
    """

    # Arrange
    alignment = str(tmp_path / "x.bam")
    shutil.copy(corpus.bams[0], alignment)
    shutil.copy(corpus.bams[0] + ".bai", str(tmp_path / spelling))

    # Act & assert
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pileup_pkg.pileup_alignments(
            [alignment],
            corpus.fasta,
            str(tmp_path / "out"),
            unstranded=True,
            contig_groups=2,
        )


def test_pileup_alignments_should_warn_when_only_a_csi_index_is_present(
    corpus, tmp_path
):
    """Test that a CSI does not count as an index.

    Given:
        An alignment with a ``.csi`` beside it and no ``.bai``.
    When:
        It is piled up in several contig batches.
    Then:
        It should warn to run ``samtools index``, because polars-bio
        cannot restrict a scan with a CSI and treating one as an index
        would promise a pushdown that does not exist.
    """

    # Arrange
    alignment = str(tmp_path / "csi.bam")
    shutil.copy(corpus.bams[0], alignment)
    shutil.copy(corpus.bams[0] + ".bai", alignment + ".csi")

    # Act & assert
    with pytest.warns(UserWarning, match="samtools index"):
        pileup_pkg.pileup_alignments(
            [alignment],
            corpus.fasta,
            str(tmp_path / "out"),
            unstranded=True,
            contig_groups=2,
        )
