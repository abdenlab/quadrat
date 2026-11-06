"""Tests for reading a reference: contig lengths by coordinate.

The cache test is relocated from the ported ``tests/test_pileup_io.py``
with its assertions unchanged. The rest is new: nothing asserted what
`chrom_lengths` actually returns, only that it returned more than one
entry and that a deleted key came back, so a function answering
``{"chr1": 1, "chr2": 2}`` would have passed the whole suite.

`clear_cache` is tested here because it now lives here: the cache and
its reset moved out of `_engine`, which held the dict and read none of
its keys. Every assertion below still goes through `chrom_lengths` and
`clear_cache` is named only as the trigger.

That it has to be named at all is an API finding rather than a test
convention, and the move narrows it rather than settling it.
`clear_cache` carries no underscore and `reference`'s own module
docstring points a reader at it as the reclamation path for what this
caches, but it is exported from no ``__all__`` and carries no
``# public`` marker -- so whether cache reclamation is a capability
this package offers is still undecided. What is no longer true is that
the function was unreachable because its module was private.
"""

import collections
import os
import pathlib

import pytest

from quadrat import _engine, reference
from tests import THREE_CONTIGS, TWO_CONTIGS, builders


@pytest.fixture
def reference_fasta(corpus_factory):
    """An indexed reference over three contigs."""

    pytest.importorskip("pyfaidx")
    return corpus_factory("reference", THREE_CONTIGS, index_fasta=True).fasta


@pytest.fixture(autouse=True)
def _forget_cached_lengths():
    """Leave no cached mapping behind, the cache being process-global."""

    reference.clear_cache()
    yield
    reference.clear_cache()


def test_chrom_lengths_should_report_each_contig_at_its_declared_length(
    reference_fasta,
):
    """Test the mapping a reference's lengths come back as.

    Given:
        An indexed reference written to a known geometry.
    When:
        Its contig lengths are read.
    Then:
        It should report every contig at the length it was written at,
        in the reference's own order.

    The order is load-bearing rather than cosmetic: `pileup_alignments`
    builds its contig batches by walking this mapping, so two readers
    disagreeing about the order would disagree about which contigs land
    in which batch.
    """

    # Act
    lengths = reference.chrom_lengths(reference_fasta)

    # Assert
    assert list(lengths.items()) == list(THREE_CONTIGS.items())


def test_chrom_lengths_should_not_hand_back_its_own_cache(reference_fasta):
    """Test that the caller is given a copy.

    Given:
        A caller who edits the mapping `chrom_lengths` returned.
    When:
        The lengths are read again.
    Then:
        It should be unchanged by the edit.

    The mapping is cached for the process, and it used to be returned by
    reference. A caller who dropped an entry from what they were handed
    changed which contigs every later pileup in that process counted,
    and the short table that resulted still passed `require_complete` --
    a manifest listing fewer files is a manifest whose files all exist.
    """

    # Arrange
    first = reference.chrom_lengths(reference_fasta)
    assert len(first) > 1

    # Act
    victim = list(first)[-1]
    del first[victim]

    # Assert
    assert victim in reference.chrom_lengths(reference_fasta)
    assert reference.chrom_lengths(reference_fasta) != first


def test_chrom_lengths_should_accept_a_path_object(reference_fasta):
    """Test that a path-like reference is read rather than refused.

    Given:
        An indexed reference named by a `pathlib.Path`.
    When:
        Its contig lengths are read.
    Then:
        It should return what the equivalent string returns.

    `pathlib.Path` is the path type a caller reaches for first, and an
    ``isinstance(str)`` probe sent it to the refusal for in-memory
    sequence dictionaries -- a diagnosis that named neither the type it
    was given nor anything the caller had done.
    """

    # Act & assert
    assert reference.chrom_lengths(pathlib.Path(reference_fasta)) == (
        reference.chrom_lengths(reference_fasta)
    )


def test_chrom_lengths_should_accept_a_bytes_path(reference_fasta):
    """Test that a bytes reference is read rather than refused.

    Given:
        An indexed reference named by a ``bytes`` path.
    When:
        Its contig lengths are read.
    Then:
        It should return what the equivalent string returns.

    A ``bytes`` path is a path everywhere else in Python, and both
    pileup entry points admitted one, so this surface refused a shape
    its siblings accepted -- with a message about in-memory sequence
    dictionaries, which is not what a ``bytes`` path is.
    """

    # Act & assert
    assert reference.chrom_lengths(os.fsencode(reference_fasta)) == (
        reference.chrom_lengths(reference_fasta)
    )


def test_chrom_lengths_should_raise_type_error_for_an_in_memory_mapping():
    """Test the refusal of something that is not a reference at all.

    Given:
        A mapping of contig lengths rather than a path to a FASTA.
    When:
        Its contig lengths are read.
    Then:
        It should raise `TypeError` naming the type it got, the engine
        reading sequences from a file rather than from memory.

    `NotImplementedError` was the class, and it derives from
    `RuntimeError` rather than from `TypeError`: a caller writing
    ``except TypeError`` around argument validation -- which is the
    obvious guard, and the one `quadrat.require_complete` answers to
    for the identical mistake -- did not catch this. That the shape is
    a planned capability is prose, not an exception class.
    """

    # Act & assert
    with pytest.raises(TypeError, match="dict"):
        reference.chrom_lengths(dict(TWO_CONTIGS))


def test_chrom_lengths_should_reread_when_the_fasta_path_changes(corpus_factory):
    """Test that the cache is keyed on the reference it was read from.

    Given:
        Two indexed references of different geometries, in one process.
    When:
        The second one's lengths are read after the first one's.
    Then:
        It should report the second reference's geometry, not the
        first's.
    """

    # Arrange
    pytest.importorskip("pyfaidx")
    first = corpus_factory("ref-two", TWO_CONTIGS, index_fasta=True).fasta
    second = corpus_factory("ref-three", THREE_CONTIGS, index_fasta=True).fasta

    # Act
    before = reference.chrom_lengths(first)
    after = reference.chrom_lengths(second)

    # Assert
    assert list(before.items()) == list(TWO_CONTIGS.items())
    assert list(after.items()) == list(THREE_CONTIGS.items())


def test_chrom_lengths_should_reread_a_reference_replaced_at_the_same_path(
    tmp_path, pysam
):
    """Test that a regenerated reference is not served from the cache.

    Given:
        A reference whose lengths have been read, then rewritten at the
        same path to a different geometry.
    When:
        The lengths are read again, without the cache being cleared.
    Then:
        It should report the new geometry.

    The cache used to be keyed on the path alone, so this read answered
    with the previous genome -- and the lengths are what bound a
    position to its contig and what a contig batch is built from, so
    the stale answer did not merely go stale, it silently changed what
    a table contained. The key carries the modification time and the
    size, either of which a rewrite moves.
    """

    # Arrange
    pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    original = reference.chrom_lengths(path)

    # Act
    builders.write_fasta(path, {"chr1": 777}, index=True)
    replaced = reference.chrom_lengths(path)

    # Assert
    assert list(original.items()) == list(TWO_CONTIGS.items())
    assert list(replaced.items()) == [("chr1", 777)]


def test_chrom_lengths_should_cache_the_path_and_the_lengths_as_one_value(
    tmp_path, pysam
):
    """Test that the cache cannot hold a path and lengths that disagree.

    Given:
        Two references at two paths, read in turn.
    When:
        The cache slot is inspected.
    Then:
        It should hold the second reference's key and its lengths as a
        single value, so no reader can observe one without the other.

    The pair used to live in two slots written by two statements, with
    a file read between the check and the writes and no lock anywhere
    -- `chrom_lengths` is published, is in ``quadrat.__all__`` and
    documents no single-thread precondition. Two threads on two
    references could leave one genome's lengths filed under the other's
    key, permanently, passing the validity check for every later caller
    of that path. A single assignment is what makes that unreachable
    rather than unlikely, and the shape is the whole of the fix, so the
    shape is what this pins.
    """

    # Arrange
    pytest.importorskip("pyfaidx")
    first = str(tmp_path / "first.fa")
    second = str(tmp_path / "second.fa")
    builders.write_fasta(first, TWO_CONTIGS, index=True)
    builders.write_fasta(second, {"chr9": 404}, index=True)

    # Act
    reference.chrom_lengths(first)
    lengths = reference.chrom_lengths(second)

    # Assert
    assert reference.cache["chrom_lengths"] == (
        reference._fasta_key(second),
        lengths,
    )


def test_chrom_lengths_should_return_what_it_computed_not_the_cache(
    tmp_path, mocker, pysam
):
    """Test that a concurrent write cannot change this caller's answer.

    Given:
        A reference whose lengths are read, and a cache that is
        overwritten the instant this call publishes to it.
    When:
        The lengths are read.
    Then:
        It should return the geometry it computed rather than whatever
        the cache holds afterwards.

    The return read the module-global back after writing it, which made
    the torn state observable inside a single call: a thread could be
    handed another genome's mapping on the path where it did the work
    itself. A real interleaving cannot be forced at that granularity
    from a test, so the concurrent writer is stood in for by a cache
    that replaces whatever is published to it -- which is what any
    second caller publishing between the write and the read-back
    amounts to.

    Those lengths build the chrom-sizes table, are what the header
    check is run against and are what the contig batches split, so the
    failure is a short table that still passes `require_complete`.
    """

    # Arrange
    pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    decoy = collections.OrderedDict([("chr9", 404)])

    class Clobbered(dict):
        def __setitem__(self, key, value):
            if key == "chrom_lengths" and value is not None:
                value = (value[0], decoy)
            super().__setitem__(key, value)

    mocker.patch.object(reference, "cache", Clobbered(reference.cache))

    # Act
    lengths = reference.chrom_lengths(path)

    # Assert
    assert list(lengths.items()) == list(TWO_CONTIGS.items())
    assert reference.cache["chrom_lengths"][1] is decoy


def test_chrom_lengths_should_not_leave_the_fasta_bound_in_the_session(
    tmp_path, mocker, pysam
):
    """Test that the no-index fallback releases what it registered.

    Given:
        A reference with no readable index, whose lengths are therefore
        read by scanning every sequence.
    When:
        The lengths come back.
    Then:
        The FASTA table should be unqueryable, so the provider holding
        that file is released rather than merely forgotten.

    The registration was recorded as a key in a dict and nothing ever
    dropped the table: a value has no teardown hook, so the provider
    stayed bound in the process-global session holding its file handle
    for the life of the process. Three claims rested on it being
    released -- `clear_cache`'s summary, `pileup_alignments`' "everything
    this call puts into the shared session is undone on the way out",
    and the fork prohibition `pileup_many` publishes on every runner a
    deployment may use. The shape of this assertion is the one
    ``tests/test_alignments.py`` already uses for the BAM tables.
    """

    # Arrange
    pb = pytest.importorskip("polars_bio")
    pyfaidx = pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    # Indexing is refused rather than the index merely being absent,
    # because `tmp_path` is writable and pyfaidx would otherwise build
    # one in place and never reach the fallback. Same seam as the
    # warning test below.
    mocker.patch.object(
        pyfaidx, "Fasta", side_effect=pyfaidx.FastaIndexingError("cannot index")
    )

    # Act
    with pytest.warns(UserWarning, match="no usable FASTA index"):
        lengths = reference.chrom_lengths(path)

    # Assert
    assert list(lengths.items()) == list(TWO_CONTIGS.items())
    with pytest.raises(Exception):
        pb.sql("SELECT * FROM {} LIMIT 1".format(_engine.FASTA_TABLE)).collect()


def test_clear_cache_should_leave_the_lengths_readable(tmp_path, pysam):
    """Test that clearing the cache costs a read rather than an answer.

    Given:
        A reference whose lengths have been read.
    When:
        The cache is cleared and the lengths are read again.
    Then:
        It should report the same geometry.

    `clear_cache` is documented as the reclamation path for what this
    module caches, and this is now the whole of its observable effect
    on lengths: the cache invalidates itself when the file changes, so
    there is no staleness left for clearing to cure. What it still
    buys is the memory, which Test Guide §8 puts out of reach of a test
    that may not name private state.
    """

    # Arrange
    pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    original = reference.chrom_lengths(path)

    # Act
    reference.clear_cache()
    again = reference.chrom_lengths(path)

    # Assert
    assert list(original.items()) == list(TWO_CONTIGS.items())
    assert list(again.items()) == list(original.items())


def test_chrom_lengths_should_warn_when_it_falls_back_to_scanning(
    tmp_path, mocker, pysam
):
    """Test that the expensive path announces itself.

    Given:
        A reference pyfaidx cannot index, so the lengths come from
        scanning every sequence.
    When:
        The lengths are read.
    Then:
        It should warn, naming the cost and the remedy, and still
        answer correctly.

    The scan is three orders of magnitude more expensive than the index
    -- upstream measured 4.8 s and a 10.2 GB peak on hg38 against
    0.00 s and 0.03 GB -- and it is in no term of the memory model the
    fan-out plans against, so under the process backend every worker
    pays it and none of it was reserved. The package already warns
    about the cheaper surprise of an unindexed alignment, where only
    the wall clock changes.
    """

    # Arrange
    pytest.importorskip("polars_bio")
    pyfaidx = pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    mocker.patch.object(
        pyfaidx, "Fasta", side_effect=pyfaidx.FastaIndexingError("cannot index")
    )

    # Act
    with pytest.warns(UserWarning, match="no usable FASTA index") as caught:
        lengths = reference.chrom_lengths(path)

    # Assert
    assert list(lengths.items()) == list(TWO_CONTIGS.items())
    assert "samtools faidx" in str(caught[0].message)


def test_chrom_lengths_should_not_warn_when_the_index_is_usable(
    reference_fasta, recwarn
):
    """Test that the cheap path says nothing.

    Given:
        A reference with a usable index beside it.
    When:
        The lengths are read.
    Then:
        It should emit no scan warning.
    """

    # Act
    lengths = reference.chrom_lengths(reference_fasta)

    # Assert
    assert list(lengths.items()) == list(THREE_CONTIGS.items())
    assert not [
        caught
        for caught in recwarn.list
        if "no usable FASTA index" in str(caught.message)
    ]


def test_chrom_lengths_should_propagate_a_pyfaidx_failure_it_cannot_name(
    tmp_path, mocker, pysam
):
    """Test that an unrecognised pyfaidx failure ends the read.

    Given:
        A reference whose index read raises something outside the set
        the scan is the documented fallback for.
    When:
        The lengths are read.
    Then:
        It should propagate rather than buying a full-genome scan.

    The catch used to be `except Exception` over the import, the open,
    the reads and the close alike, so a corrupt index and a pyfaidx
    version break both bought the scan silently. Narrowing it turns
    those from a slow run into a dead one, which is the right way round
    for a fallback this expensive.
    """

    # Arrange
    pyfaidx = pytest.importorskip("pyfaidx")
    path = str(tmp_path / "genome.fa")
    builders.write_fasta(path, TWO_CONTIGS, index=True)
    mocker.patch.object(
        pyfaidx, "Fasta", side_effect=RuntimeError("pyfaidx changed under us")
    )

    # Act & assert
    with pytest.raises(RuntimeError, match="changed under us"):
        reference.chrom_lengths(path)
