"""Tests for the shared fixture layer and the suite's own environment.

`corpus_factory` is the one builder every data-path suite draws its
inputs from, which makes its keyword surface load-bearing: a keyword
that silently stopped reaching the builders would leave every later
suite asserting against data it did not ask for. So each one is
exercised here rather than trusted.

The environment block is tested in a subprocess, because the only claim
worth making about it concerns what is true at *import* time, and by the
time a test body runs the import has long since happened.
"""

import gzip
import os
import sys
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import (
    FILE_BACKED,
    PROJECT_ROOT,
    TWO_CONTIGS,
    builders,
    record_placements,
    run_child,
)
from tests.conftest import Corpus

# Contig names from the suite's own vocabulary rather than arbitrary
# text: the builders beneath this factory already property-test
# arbitrary names, and what this factory adds is forwarding a geometry
# it did not choose.
corpus_sizes = st.dictionaries(
    st.sampled_from(["chr1", "chr2", "chr3", "chrX"]),
    st.integers(min_value=1, max_value=200),
    min_size=1,
    max_size=4,
)


def import_conftest(environment):
    """Import the suite's conftest in a fresh interpreter.

    Returns a mapping of what the child could observe and the parent
    cannot: ``cache_dir``, whether it ``existed`` at import time, whether
    the package was ``imported`` by the time the import finished, and the
    ``tqdm_disable`` the environment was left with. Existence is reported
    from inside the child because the child removes the directory when it
    exits.
    """

    program = (
        "import os, sys\n"
        "import tests.conftest\n"
        "cache_dir = os.environ['QUADRAT_CACHE_DIR']\n"
        "print(cache_dir)\n"
        "print(os.path.isdir(cache_dir))\n"
        "print('quadrat' in sys.modules)\n"
        "print(os.environ['TQDM_DISABLE'])\n"
    )
    env = dict(os.environ)
    for name, value in environment.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value

    result = run_child(
        [sys.executable, "-c", program],
        "importing the conftest failed",
        cwd=PROJECT_ROOT,
        environment=env,
    )

    cache_dir, existed, imported, tqdm_disable = result.stdout.splitlines()
    return {
        "cache_dir": cache_dir,
        "existed": existed == "True",
        "imported": imported == "True",
        "tqdm_disable": tqdm_disable,
    }


class TestCorpus:
    """The attribute bag the factory hands back."""

    def test___init___should_leave_every_optional_slot_unset(self, tmp_path):
        """Test that a fresh corpus claims nothing it was not given.

        Given:
            A root directory and a chromosome mapping.
        When:
            A corpus is instantiated.
        Then:
            It should expose that root and mapping with no alignments
            and no locus files, so a test cannot read a file nothing
            built.
        """

        # Act
        corpus = Corpus(tmp_path, TWO_CONTIGS)

        # Assert
        assert corpus.root == tmp_path
        assert corpus.chrom_sizes == TWO_CONTIGS
        assert corpus.fasta is None
        assert corpus.bams == []
        assert corpus.peaks is None
        assert corpus.negatives is None
        assert corpus.exclusion is None

    def test___init___should_copy_the_chromosome_sizes_it_is_given(self, tmp_path):
        """Test that a corpus does not alias the caller's mapping.

        Given:
            A caller's chromosome mapping.
        When:
            A corpus is built from it.
        Then:
            It should hold an equal but separate mapping, so a test
            that mutates its own geometry cannot corrupt a constant
            every later corpus in the session is built from.
        """

        # Arrange
        sizes = dict(TWO_CONTIGS)

        # Act
        corpus = Corpus(tmp_path, sizes)

        # Assert
        assert corpus.chrom_sizes == sizes
        assert corpus.chrom_sizes is not sizes

    def test_path_should_resolve_a_name_inside_the_corpus_root(self, tmp_path):
        """Test that a corpus resolves paths against its own directory.

        Given:
            A corpus rooted at a known directory.
        When:
            A name is resolved through it.
        Then:
            It should return that name inside the corpus directory as a
            string, which is what the builders take.
        """

        # Arrange
        corpus = Corpus(tmp_path, TWO_CONTIGS)

        # Act
        resolved = corpus.path("genome.fa")

        # Assert
        assert resolved == str(tmp_path / "genome.fa")
        assert isinstance(resolved, str)

    def test___setattr___should_raise_when_the_attribute_is_undeclared(self, tmp_path):
        """Test that a misspelt field fails where it is written.

        Given:
            A corpus.
        When:
            An attribute outside its declared slots is assigned.
        Then:
            It should raise AttributeError, which is the whole reason
            the class is an attribute bag rather than a dict.
        """

        # Arrange
        corpus = Corpus(tmp_path, TWO_CONTIGS)

        # Act & assert
        with pytest.raises(AttributeError):
            corpus.peeks = "typo"


def test_corpus_factory_should_build_only_what_it_was_asked_for(corpus_factory):
    """Test that unrequested pieces of a corpus stay absent.

    Given:
        A corpus requested with peaks and nothing else.
    When:
        The factory builds it.
    Then:
        It should populate the peaks and leave the rest unset, beside
        the FASTA it writes unconditionally, so a test cannot read a
        file it never asked to exist.
    """

    # Act
    corpus = corpus_factory("minimal", TWO_CONTIGS, peaks=[("chr1", 100, 140)])

    # Assert
    assert os.path.exists(corpus.fasta)
    assert os.path.exists(corpus.peaks)
    assert corpus.bams == []
    assert corpus.negatives is None
    assert corpus.exclusion is None


def test_corpus_factory_should_build_every_locus_file_it_is_asked_for(corpus_factory):
    """Test that the requested locus files are all written.

    Given:
        A corpus requested with peaks, negatives and an exclusion
        list.
    When:
        The factory builds it.
    Then:
        It should write all three, with only the exclusion list
        gzipped, as the suffix the factory gives each one dictates.
    """

    # Act
    corpus = corpus_factory(
        "populated",
        TWO_CONTIGS,
        peaks=[("chr1", 100, 140)],
        negatives=[("chr1", 500, 540)],
        exclusion=[("chr2", 10, 60)],
    )

    # Assert
    with open(corpus.peaks) as handle:
        assert handle.read() == "chr1\t100\t140\n"
    with open(corpus.negatives) as handle:
        assert handle.read() == "chr1\t500\t540\n"
    assert corpus.exclusion.endswith(".bed.gz")
    with gzip.open(corpus.exclusion, "rt") as handle:
        assert handle.read() == "chr2\t10\t60\n"


def test_corpus_factory_should_leave_an_empty_locus_request_unset(corpus_factory):
    """Test that asking for no loci is not the same as asking for some.

    Given:
        A corpus requested with an empty peaks iterable.
    When:
        The factory builds it.
    Then:
        It should leave the peaks attribute unset, so asking for no
        loci is today indistinguishable from not asking -- a
        limitation worth pinning, because the underlying writer does
        accept an empty row list.
    """

    # Act
    corpus = corpus_factory("empty-peaks", TWO_CONTIGS, peaks=[])

    # Assert
    assert corpus.peaks is None


def test_corpus_factory_should_give_each_corpus_its_own_directory(corpus_factory):
    """Test that two corpora of one geometry do not share files.

    Given:
        Two corpora requested under one name and identical chromosome
        sizes.
    When:
        Both are built.
    Then:
        It should give each its own directory, so neither can observe
        the other's writes even though the factory itself lives for the
        whole session.
    """

    # Act
    first = corpus_factory("twin", TWO_CONTIGS)
    second = corpus_factory("twin", TWO_CONTIGS)

    # Assert
    assert first.root != second.root


def test_corpus_factory_should_copy_the_chromosome_sizes_it_is_given(corpus_factory):
    """Test that a built corpus does not alias the caller's mapping.

    Given:
        A corpus built from a caller's chromosome mapping.
    When:
        That mapping is compared to the corpus's own.
    Then:
        It should hold an equal but separate mapping, so a test cannot
        corrupt a shared constant for the rest of the session.
    """

    # Arrange
    sizes = dict(TWO_CONTIGS)

    # Act
    corpus = corpus_factory("copied", sizes)

    # Assert
    assert corpus.chrom_sizes == sizes
    assert corpus.chrom_sizes is not sizes


def test_corpus_factory_should_pass_the_fasta_geometry_through(
    corpus_factory, tmp_path, pysam
):
    """Test that the sequence keywords reach the FASTA builder.

    Given:
        A corpus requested with a seed, a soft mask, an N run and
        FASTA indexing.
    When:
        The factory builds it.
    Then:
        It should produce a FASTA matching a direct render of the same
        request, with an index beside it.
    """

    # Arrange
    geometry = {
        "seed": 11,
        "soft_mask": [("chr1", 100, 120)],
        "n_runs": [("chr1", 200, 210)],
    }
    expected = builders.write_fasta(
        str(tmp_path / "expected.fa"), TWO_CONTIGS, **geometry
    )

    # Act
    corpus = corpus_factory("geometry", TWO_CONTIGS, index_fasta=True, **geometry)

    # Assert
    with open(corpus.fasta) as built, open(expected) as reference:
        assert built.read() == reference.read()
    assert os.path.exists(corpus.fasta + ".fai")


def test_corpus_factory_should_write_one_indexed_alignment_per_sample(
    corpus_factory, pysam
):
    """Test that every sample gets an alignment file of its own.

    Given:
        A corpus requested with two alignments.
    When:
        The factory builds it.
    Then:
        It should write two separately named BAMs with an index beside
        each, so a later suite can open either by name and seek in it.
    """

    # Act
    corpus = corpus_factory("samples", TWO_CONTIGS, n_bams=2, n_reads=6)

    # Assert
    assert [os.path.basename(path) for path in corpus.bams] == ["s0.bam", "s1.bam"]
    for path in corpus.bams:
        assert os.path.exists(path + ".bai")


def test_corpus_factory_should_forward_the_read_count_to_every_alignment(
    corpus_factory, pysam
):
    """Test that the requested read count reaches each alignment.

    Given:
        A corpus requested with two alignments of a given read count.
    When:
        The factory builds it.
    Then:
        It should write that many records to each of them, so the
        keyword reaches the builder rather than stopping here.
    """

    # Act
    corpus = corpus_factory("counted", TWO_CONTIGS, n_bams=2, n_reads=6)

    # Assert
    for path in corpus.bams:
        assert len(record_placements(path, pysam)) == 6


def test_corpus_factory_should_place_each_sample_differently(corpus_factory, pysam):
    """Test that two samples are not one alignment written twice.

    Given:
        A corpus requested with two alignments.
    When:
        Their placements are compared.
    Then:
        It should have placed them differently, so a suite reading two
        samples sees two.
    """

    # Arrange
    corpus = corpus_factory("distinct", TWO_CONTIGS, n_bams=2, n_reads=6)

    # Act
    placements = [record_placements(path, pysam) for path in corpus.bams]

    # Assert
    assert placements[0] != placements[1]


def test_corpus_factory_should_place_alignment_reads_identically_for_one_seed(
    corpus_factory, pysam
):
    """Test that one seed gives two corpora the same placements.

    Given:
        Two corpora of one geometry requested under the same seed.
    When:
        Their alignments are compared.
    Then:
        It should place every read identically, so a suite that rebuilds
        a corpus reasons about the same data twice.
    """

    # Arrange
    first = corpus_factory("one-seed-a", TWO_CONTIGS, n_bams=1, n_reads=6, seed=5)
    second = corpus_factory("one-seed-b", TWO_CONTIGS, n_bams=1, n_reads=6, seed=5)

    # Act
    placements = [record_placements(c.bams[0], pysam) for c in (first, second)]

    # Assert
    assert placements[0] == placements[1]


def test_corpus_factory_should_place_alignment_reads_differently_for_another_seed(
    corpus_factory, pysam
):
    """Test that the seed reaches the reads the factory places.

    Given:
        Two corpora of one geometry requested under different seeds.
    When:
        Their alignments are compared.
    Then:
        It should place the reads differently, so the seed is a real
        knob on the alignment path rather than one the factory accepts
        and drops.
    """

    # Arrange
    first = corpus_factory("two-seeds-a", TWO_CONTIGS, n_bams=1, n_reads=6, seed=5)
    second = corpus_factory("two-seeds-b", TWO_CONTIGS, n_bams=1, n_reads=6, seed=6)

    # Act
    placements = [record_placements(c.bams[0], pysam) for c in (first, second)]

    # Assert
    assert placements[0] != placements[1]


@pytest.mark.parametrize(
    "header_contigs",
    [{"chr1": 1000}, {"chr1": 1000, "chr2": 400}],
    ids=["omitted", "shorter"],
)
def test_corpus_factory_should_raise_when_header_contigs_do_not_cover_the_reference(
    corpus_factory, header_contigs
):
    """Test that a header narrower than the reference is refused.

    Given:
        Header contigs that leave out a contig the chromosome sizes
        declare, or declare one of them shorter than it is.
    When:
        The factory is asked to build an alignment over them.
    Then:
        It should raise ValueError naming the uncovered contig, because
        reads are placed against the reference and would otherwise be
        written half-placed or past the end the header declares.
    """

    # Act & assert
    with pytest.raises(ValueError, match="do not cover the reference.*chr2"):
        corpus_factory("uncovered", TWO_CONTIGS, n_bams=1, header_contigs=header_contigs)


def test_corpus_factory_should_declare_header_contigs_the_reference_lacks(
    corpus_factory, pysam
):
    """Test that a BAM header may name a contig outside the reference.

    Given:
        A corpus whose header contigs include one absent from the
        chromosome sizes the FASTA was built from.
    When:
        The factory builds an alignment over it.
    Then:
        It should write the BAM with that contig in its header, so a
        header-versus-reference mismatch is something a suite can
        construct rather than only hope to encounter.
    """

    # Arrange
    header_contigs = dict(TWO_CONTIGS, chrZ=800)

    # Act
    corpus = corpus_factory(
        "asymmetric", TWO_CONTIGS, n_bams=1, header_contigs=header_contigs
    )

    # Assert
    with pysam.AlignmentFile(corpus.bams[0], "rb") as bam:
        assert "chrZ" in set(bam.references)
    assert "chrZ" not in corpus.chrom_sizes


def test_corpus_factory_should_default_header_contigs_to_the_reference(
    corpus_factory, pysam
):
    """Test that the header follows the reference unless overridden.

    Given:
        A corpus requested with one alignment and no header contigs.
    When:
        The factory builds it.
    Then:
        It should give the BAM a header whose contigs and lengths equal
        the chromosome sizes, so the symmetric case needs no argument.
    """

    # Act
    corpus = corpus_factory("symmetric", TWO_CONTIGS, n_bams=1)

    # Assert
    with pysam.AlignmentFile(corpus.bams[0], "rb") as bam:
        assert list(bam.references) == list(TWO_CONTIGS)
        assert list(bam.lengths) == list(TWO_CONTIGS.values())


@settings(FILE_BACKED, max_examples=15)
@given(sizes=corpus_sizes)
def test_corpus_factory_should_render_a_corpus_to_any_geometry(
    corpus_factory, tmp_path, sizes
):
    """Test that any geometry reaches the artifacts built from it.

    Given:
        An arbitrary contig mapping and a request for peaks and an
        exclusion list over it.
    When:
        The factory builds a corpus.
    Then:
        It should report that geometry and write a sequence identical
        to a direct render of it, with both locus files present, so the
        one parameter the factory exists to take is not assumed at a
        single value.
    """

    # Arrange
    rows = [(chrom, 0, 10) for chrom in sizes]
    expected = builders.write_fasta(str(tmp_path / "expected.fa"), sizes, seed=3)

    # Act
    corpus = corpus_factory("any-geometry", sizes, seed=3, peaks=rows, exclusion=rows)

    # Assert
    assert corpus.chrom_sizes == dict(sizes)
    with open(corpus.fasta) as built, open(expected) as reference:
        assert built.read() == reference.read()
    assert os.path.exists(corpus.peaks)
    assert os.path.exists(corpus.exclusion)


@pytest.mark.packaging
def test_quadrat_cache_dir_should_be_set_before_the_package_is_imported():
    """Test that nothing can read the variable before it is in place.

    Given:
        A fresh interpreter with no cache variable set.
    When:
        The suite's conftest is imported and nothing else.
    Then:
        It should set the variable while leaving the package
        unimported, so any later import of it sees the variable already
        there, for the reason the conftest module docstring gives.
    """

    # Act
    observed = import_conftest({"QUADRAT_CACHE_DIR": None})

    # Assert
    assert observed["existed"]
    assert not observed["imported"]


@pytest.mark.packaging
def test_quadrat_cache_dir_should_replace_a_value_the_caller_exported():
    """Test that an exported cache directory does not survive.

    Given:
        A fresh interpreter with the cache variable already exported to
        a path of the caller's own.
    When:
        The suite's conftest is imported.
    Then:
        It should replace that value with a fresh temporary directory,
        for the reason the conftest module docstring gives.
    """

    # Arrange
    exported = os.path.join(tempfile.gettempdir(), "quadrat-caller-cache")

    # Act
    observed = import_conftest({"QUADRAT_CACHE_DIR": exported})

    # Assert
    assert observed["cache_dir"] != exported
    assert observed["existed"]


@pytest.mark.packaging
def test_tqdm_disable_should_defer_to_a_value_the_caller_exported():
    """Test that a developer may turn progress bars back on.

    Given:
        A fresh interpreter with progress bars explicitly enabled.
    When:
        The suite's conftest is imported.
    Then:
        It should leave that value in place, because a progress bar
        changes nothing a test can observe and watching one is how a
        hang gets diagnosed.
    """

    # Act
    observed = import_conftest({"TQDM_DISABLE": "0"})

    # Assert
    assert observed["tqdm_disable"] == "0"
