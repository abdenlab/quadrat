"""Tests for tracks the training loader piles up itself.

`Alignments` is pinned on what it accepts and refuses, subclassing
included. `alignments_key` is pinned on each input that moves it,
`managed_tracks` on piling an entry up once, reusing it, and answering
what a direct `quadrat.pileup_alignments` call writes, and
`resolve_alignments` and `managed_track_dir_factory` on the
specifications and the directory they answer. The loader's own use of
them is pinned in ``tests/test_loader.py``.

Ordered foundational to derived, so ``pytest -x`` stops at the most
fundamental failure.
"""

import dataclasses
import fcntl
import multiprocessing
import os
import pathlib
import shutil
import time

import polars
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from quadrat import Aggregate, managed, pileup_alignments, tables
from tests import TWO_CONTIGS, aggregates

#: Path names a property draws: short, and safe in every spelling.
_path_names = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=8
).map(lambda name: name + ".bam")

#: A path name in one of the three spellings `Alignments` accepts.
_spelled_paths = st.one_of(
    _path_names,
    _path_names.map(os.fsencode),
    _path_names.map(pathlib.Path),
)

#: Group layouts as `quadrat.groups.normalize_signal_groups` answers
#: them: a few groups of a few channels each.
_group_layouts = st.lists(st.integers(min_value=1, max_value=4), max_size=5)

#: How long a child is given to reach the lock before the test releases
#: it, in seconds. Generous, because a spawned child imports the whole
#: package before it gets there; a child that is late still finds the
#: entry the first one committed, so a slow machine weakens the test
#: rather than failing it.
_LOCK_HOLD_SECONDS = 8

#: How long a child may take to pile up and return, in seconds.
_CHILD_TIMEOUT = 120


@pytest.fixture(scope="module")
def corpus(corpus_factory):
    """Two small indexed alignments over two contigs, with an indexed reference."""

    pytest.importorskip("polars_bio")
    return corpus_factory("managed", TWO_CONTIGS, n_bams=2, index_fasta=True)


@pytest.fixture
def copied(corpus, tmp_path):
    """One alignment and its index, copied where a test may rewrite them."""

    path = str(tmp_path / "copy.bam")
    shutil.copy(corpus.bams[0], path)
    shutil.copy(corpus.bams[0] + ".bai", path + ".bai")
    return path


def entry_of(paths):
    """Return the entry directory a list of managed track paths sits in."""

    return os.path.dirname(paths[0])


def pile_up_in_child(bam, fasta, track_dir, queue):
    """Pile one `Alignments` up in a child process and report the entry's inode.

    Module-level so a spawned child can import it.
    """

    paths = managed.managed_tracks(managed.Alignments(bam), fasta, track_dir)
    queue.put((paths, os.stat(entry_of(paths)).st_ino))


class TestAlignments:
    """Tests for `Alignments`."""

    def test___init___should_hold_one_path_as_a_tuple_of_one(self):
        """Test that a bare path is one alignment rather than its characters.

        Given:
            One path, as a `str` and as a `pathlib.Path`.
        When:
            An `Alignments` is built from each.
        Then:
            Its ``paths`` should be a `tuple` holding that path as a
            `str`.
        """

        # Act & assert
        for path in ("a.bam", pathlib.Path("a.bam")):
            assert managed.Alignments(path).paths == ("a.bam",)

    def test___init___should_hold_a_list_of_paths_in_order(self):
        """Test that several alignments keep the order they were given in.

        Given:
            A `list` of a `str`, a `bytes` and a `pathlib.Path`.
        When:
            An `Alignments` is built from it.
        Then:
            Its ``paths`` should be those three paths as `str`, in order.
        """

        # Act
        built = managed.Alignments(["b.bam", b"a.bam", pathlib.Path("c.bam")])

        # Assert
        assert built.paths == ("b.bam", "a.bam", "c.bam")

    @pytest.mark.parametrize("paths", [7, ["a.bam", None], {"a.bam"}])
    def test___init___should_raise_type_error_when_paths_is_not_paths(self, paths):
        """Test the refusal of alignments that are not paths.

        Given:
            A number, a list holding something that is not a path, and
            a `set`, which states no order.
        When:
            An `Alignments` is built from each.
        Then:
            It should raise `TypeError`.
        """

        # Act & assert
        with pytest.raises(TypeError, match="one path or a list or tuple"):
            managed.Alignments(paths)

    def test___init___should_raise_value_error_when_paths_is_empty(self):
        """Test the refusal of a channel with no alignment to pile up.

        Given:
            An empty `list` and an empty `tuple`.
        When:
            An `Alignments` is built from each.
        Then:
            It should raise `ValueError`.
        """

        # Act & assert
        for paths in ([], ()):
            with pytest.raises(ValueError, match="holds no path"):
                managed.Alignments(paths)

    def test___init___should_raise_type_error_when_aggregate_is_not_one(self):
        """Test the refusal of an aggregate that is not an `Aggregate`.

        Given:
            A `dict` of aggregate fields.
        When:
            An `Alignments` is built with it.
        Then:
            It should raise `TypeError`.
        """

        # Act & assert
        with pytest.raises(TypeError, match="as an Aggregate"):
            managed.Alignments("a.bam", {"position": "span"})

    @settings(max_examples=100)
    @given(
        paths=st.lists(_spelled_paths, min_size=1, max_size=4), as_tuple=st.booleans()
    )
    def test___init___should_hold_any_spelling_of_paths_as_str_in_order(
        self, paths, as_tuple
    ):
        """Test path normalisation over every accepted spelling.

        Given:
            Any non-empty list of paths, each a `str`, `bytes` or
            `pathlib.Path`, given as a `list` or as a `tuple`.
        When:
            An `Alignments` is built from it.
        Then:
            Its ``paths`` should be the same paths as `str`, in order.
        """

        # Act
        built = managed.Alignments(tuple(paths) if as_tuple else paths)

        # Assert
        assert built.paths == tuple(os.fsdecode(os.fspath(path)) for path in paths)

    @settings(max_examples=100)
    @given(names=st.lists(_path_names, min_size=1, max_size=4))
    def test___eq___should_compare_by_fields(self, names):
        """Test that two spellings of one channel are equal, and another is not.

        Given:
            Any paths, once as `str` and once as `pathlib.Path`, and the
            same paths under another aggregate.
        When:
            The three are compared.
        Then:
            The first two should be equal, and the third should differ.
        """

        # Arrange
        first = managed.Alignments(names)
        second = managed.Alignments([pathlib.Path(name) for name in names])
        third = managed.Alignments(names, Aggregate(unstranded=True))

        # Act & assert
        assert first == second
        assert third != first

    @settings(max_examples=100)
    @given(names=st.lists(_path_names, min_size=1, max_size=4))
    def test___hash___should_agree_with_equality(self, names):
        """Test that two spellings of one channel hash alike.

        Given:
            Any paths, once as `str` and once as `bytes`.
        When:
            Both are hashed.
        Then:
            The hashes should be equal, so either spelling finds the
            other in a `set`.
        """

        # Arrange
        first = managed.Alignments(names)
        second = managed.Alignments([os.fsencode(name) for name in names])

        # Act & assert
        assert hash(first) == hash(second)
        assert second in {first}

    def test___init_subclass___should_refuse_a_subclass(self):
        """Test that a subclass cannot bypass the validation readers trust.

        Given:
            A class deriving from `Alignments`.
        When:
            It is defined.
        Then:
            It should raise `TypeError` naming the subclass.
        """

        # Act & assert
        with pytest.raises(TypeError, match="cannot be subclassed, and Mine"):

            class Mine(managed.Alignments):
                pass


def test_alignments_key_should_be_stable_for_unchanged_inputs(corpus):
    """Test that a key is a function of its inputs.

    Given:
        One `Alignments` and its reference.
    When:
        The key is taken twice, the second time from an equal
        `Alignments` built anew.
    Then:
        The two keys should be equal.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])

    # Act
    first = managed.alignments_key(spec, corpus.fasta)
    second = managed.alignments_key(managed.Alignments(corpus.bams[0]), corpus.fasta)

    # Assert
    assert first == second


def test_alignments_key_should_change_with_the_aggregate_and_the_alignments(corpus):
    """Test that what the tracks hold moves the key.

    Given:
        One alignment under the default aggregate; the same under an
        unstranded one; the other alignment; and both, in each order.
    When:
        Each key is taken.
    Then:
        Every key should differ from every other.
    """

    # Arrange
    one, two = corpus.bams
    specs = [
        managed.Alignments(one),
        managed.Alignments(one, Aggregate(unstranded=True)),
        managed.Alignments(two),
        managed.Alignments([one, two]),
        managed.Alignments([two, one]),
    ]

    # Act
    keys = [managed.alignments_key(spec, corpus.fasta) for spec in specs]

    # Assert
    assert len(set(keys)) == len(keys)


def test_alignments_key_should_change_when_an_alignment_is_rewritten(copied, corpus):
    """Test that a rewritten alignment moves the key.

    Given:
        A copied alignment, keyed, then rewritten in place with one
        more byte.
    When:
        It is keyed again.
    Then:
        The key should differ.
    """

    # Arrange
    spec = managed.Alignments(copied)
    before = managed.alignments_key(spec, corpus.fasta)
    with open(copied, "ab") as handle:
        handle.write(b"\0")

    # Act
    after = managed.alignments_key(spec, corpus.fasta)

    # Assert
    assert after != before


def test_alignments_key_should_change_when_the_index_is_removed(copied, corpus):
    """Test that an alignment's index is part of its identity.

    Given:
        A copied alignment with its index, keyed, then with the index
        removed.
    When:
        It is keyed again.
    Then:
        The key should differ.
    """

    # Arrange
    spec = managed.Alignments(copied)
    before = managed.alignments_key(spec, corpus.fasta)
    os.remove(copied + ".bai")

    # Act
    after = managed.alignments_key(spec, corpus.fasta)

    # Assert
    assert after != before


def test_alignments_key_should_change_when_the_index_is_rewritten(copied, corpus):
    """Test that an index's stamp, and not only its presence, is in the key.

    Given:
        A copied alignment with its index, keyed, then the index
        rewritten in place with one more byte.
    When:
        It is keyed again.
    Then:
        The key should differ.
    """

    # Arrange
    spec = managed.Alignments(copied)
    before = managed.alignments_key(spec, corpus.fasta)
    with open(copied + ".bai", "ab") as handle:
        handle.write(b"\0")

    # Act
    after = managed.alignments_key(spec, corpus.fasta)

    # Assert
    assert after != before


def test_alignments_key_should_change_with_the_alignments_path(copied, tmp_path, corpus):
    """Test that an alignment's path is in the key, beside its stamp.

    Given:
        A copied alignment and index, and a second copy of both at
        another path, given the same sizes and modification times.
    When:
        Each is keyed.
    Then:
        The two keys should differ.
    """

    # Arrange
    moved = str(tmp_path / "moved.bam")
    for source, target in ((copied, moved), (copied + ".bai", moved + ".bai")):
        shutil.copy(source, target)
        info = os.stat(source)
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))

    # Act
    keys = [
        managed.alignments_key(managed.Alignments(path), corpus.fasta)
        for path in (copied, moved)
    ]

    # Assert
    assert keys[0] != keys[1]


def test_alignments_key_should_change_with_the_reference(corpus, tmp_path):
    """Test that the reference's index is in the key, not only its path.

    Given:
        One alignment, and one reference path holding a contig of 1000
        bases, then rewritten with one of 1001 and its stale index
        removed.
    When:
        The alignment is keyed over each.
    Then:
        The two keys should differ, so another genome behind one path
        never answers the first genome's entry.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    reference = tmp_path / "ref.fa"
    keys = []

    # Act
    for size in (1000, 1001):
        reference.write_text(">chr1\n{}\n".format("A" * size))
        pathlib.Path(str(reference) + ".fai").unlink(missing_ok=True)
        keys.append(managed.alignments_key(spec, str(reference)))

    # Assert
    assert keys[0] != keys[1]


def test_alignments_key_should_answer_one_key_for_a_relative_and_an_absolute_path(
    copied, corpus, monkeypatch
):
    """Test that an alignment is keyed by the file it names, however it is spelled.

    Given:
        A copied alignment named by its absolute path, and by a path
        relative to its directory as the working directory.
    When:
        Each is keyed.
    Then:
        The two keys should be equal.
    """

    # Arrange
    monkeypatch.chdir(os.path.dirname(copied))

    # Act
    keys = [
        managed.alignments_key(managed.Alignments(path), corpus.fasta)
        for path in (copied, os.path.basename(copied))
    ]

    # Assert
    assert keys[0] == keys[1]


def test_alignments_key_should_change_with_the_package_version(corpus, mocker):
    """Test that the installed version of the package is in the key.

    Given:
        One alignment keyed, then keyed again with the package's
        installed version read as another, the metadata being a system
        boundary.
    When:
        Each is keyed.
    Then:
        The two keys should differ.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    before = managed.alignments_key(spec, corpus.fasta)
    mocker.patch.object(managed, "version", return_value="0.0.0+other")

    # Act
    after = managed.alignments_key(spec, corpus.fasta)

    # Assert
    assert after != before


@pytest.mark.parametrize(
    "field", [field.name for field in dataclasses.fields(Aggregate)]
)
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@given(first=aggregates(), second=aggregates())
def test_alignments_key_should_change_with_any_field_of_the_aggregate(
    corpus, field, first, second
):
    """Test that each field of the aggregate is in the key on its own.

    Given:
        Any aggregate `Aggregate` accepts, and a second differing from
        it in the one named field only, over one alignment and one
        reference.
    When:
        Each is keyed.
    Then:
        The two keys should differ.
    """

    # Arrange
    try:
        variant = dataclasses.replace(first, **{field: getattr(second, field)})
    except (TypeError, ValueError):
        variant = first
    assume(variant != first)

    # Act
    keys = [
        managed.alignments_key(
            managed.Alignments(corpus.bams[0], aggregate), corpus.fasta
        )
        for aggregate in (first, variant)
    ]

    # Assert
    assert keys[0] != keys[1]


def test_alignments_key_should_raise_file_not_found_error_for_a_missing_alignment(
    corpus, tmp_path
):
    """Test the refusal of an alignment that does not exist.

    Given:
        An `Alignments` naming a path with nothing at it.
    When:
        It is keyed.
    Then:
        It should raise `FileNotFoundError`.
    """

    # Arrange
    spec = managed.Alignments(str(tmp_path / "absent.bam"))

    # Act & assert
    with pytest.raises(FileNotFoundError):
        managed.alignments_key(spec, corpus.fasta)


def test_alignments_key_should_raise_type_error_when_the_reference_is_not_a_path(
    corpus,
):
    """Test the refusal of a reference that is not a path.

    Given:
        A reference given as a number.
    When:
        An `Alignments` is keyed over it.
    Then:
        It should raise `TypeError`.
    """

    # Act & assert
    with pytest.raises(TypeError, match="sequences must be"):
        managed.alignments_key(managed.Alignments(corpus.bams[0]), 7)


@pytest.mark.parametrize("unstranded", [False, True], ids=["stranded", "unstranded"])
def test_managed_tracks_should_answer_what_a_direct_pileup_writes(
    corpus, tmp_path, unstranded
):
    """Test that a managed entry holds the tracks a pileup of its alignments writes.

    Given:
        Both alignments, stranded and unstranded.
    When:
        They are piled up through `managed_tracks` and, separately,
        through `quadrat.pileup_alignments`.
    Then:
        The two should answer as many tracks, each complete and holding
        the same rows.
    """

    # Arrange
    aggregate = Aggregate(unstranded=unstranded)
    direct = pileup_alignments(
        corpus.bams, corpus.fasta, str(tmp_path / "direct"), aggregate=aggregate
    )

    # Act
    paths = managed.managed_tracks(
        managed.Alignments(corpus.bams, aggregate),
        corpus.fasta,
        str(tmp_path / "tracks"),
    )

    # Assert
    assert len(paths) == len(direct) == (1 if unstranded else 2)
    for ours, theirs in zip(paths, direct):
        tables.require_complete(ours)
        assert polars.read_parquet(ours).equals(polars.read_parquet(theirs))


def test_managed_tracks_should_reuse_a_complete_entry(corpus, tmp_path):
    """Test that a second call finds the entry and piles nothing up.

    Given:
        An `Alignments` piled up once into a managed directory.
    When:
        It is asked for again.
    Then:
        The same paths should be answered, and the entry should be the
        same directory, which a second pileup would have replaced.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    track_dir = str(tmp_path / "tracks")
    first = managed.managed_tracks(spec, corpus.fasta, track_dir)
    inode = os.stat(entry_of(first)).st_ino

    # Act
    second = managed.managed_tracks(spec, corpus.fasta, track_dir)

    # Assert
    assert second == first
    assert os.stat(entry_of(second)).st_ino == inode


def test_managed_tracks_should_answer_a_complete_entry_it_cannot_write(corpus, tmp_path):
    """Test that reading a complete entry needs no write access.

    Given:
        An entry piled up once, its lock file removed and its managed
        directory made read-only, as one another user populated.
    When:
        It is asked for again.
    Then:
        Its paths should be answered, with no lock file made.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    track_dir = tmp_path / "tracks"
    first = managed.managed_tracks(spec, corpus.fasta, str(track_dir))
    for lock in track_dir.glob("*.lock"):
        lock.unlink()
    track_dir.chmod(0o500)

    # Act
    try:
        again = managed.managed_tracks(spec, corpus.fasta, str(track_dir))
    finally:
        track_dir.chmod(0o700)

    # Assert
    assert again == first
    assert list(track_dir.glob("*.lock")) == []


def test_managed_tracks_should_pile_up_again_over_an_incomplete_entry(corpus, tmp_path):
    """Test that an entry whose record is gone is a miss, not an answer.

    Given:
        An entry piled up once, then stripped of its completeness
        record, as a crash between the swap and the record leaves it.
    When:
        It is asked for again.
    Then:
        It should be piled up again, so its record is back and every
        track is complete.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    track_dir = str(tmp_path / "tracks")
    paths = managed.managed_tracks(spec, corpus.fasta, track_dir)
    os.remove(tables.manifest_path(entry_of(paths)))

    # Act
    again = managed.managed_tracks(spec, corpus.fasta, track_dir)

    # Assert
    assert tables.read_manifest(entry_of(again)) is not None
    for path in again:
        tables.require_complete(path)


def test_managed_tracks_should_pile_up_again_when_a_track_is_incomplete(
    corpus, tmp_path
):
    """Test that an entry whose record stands but whose track does not is a miss.

    Given:
        An entry piled up once, then one of its tracks stripped of its
        own completeness record, the entry's record left in place.
    When:
        It is asked for again.
    Then:
        It should be committed afresh, a new directory, with every
        track complete.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    track_dir = str(tmp_path / "tracks")
    paths = managed.managed_tracks(spec, corpus.fasta, track_dir)
    inode = os.stat(entry_of(paths)).st_ino
    os.remove(tables.manifest_path(paths[0]))

    # Act
    again = managed.managed_tracks(spec, corpus.fasta, track_dir)

    # Assert
    assert os.stat(entry_of(again)).st_ino != inode
    for path in again:
        tables.require_complete(path)


def test_managed_tracks_should_commit_nothing_when_the_pileup_fails(corpus, tmp_path):
    """Test that a failed pileup leaves no entry and no staging directory.

    Given:
        An `Alignments` whose alignment names contigs the reference
        does not have, which `quadrat.pileup_alignments` refuses.
    When:
        It is piled up into an empty managed directory.
    Then:
        The refusal should be raised, and the directory should hold
        nothing but the lock.
    """

    # Arrange
    track_dir = tmp_path / "tracks"
    reference = str(tmp_path / "other.fa")
    with open(reference, "w") as handle:
        handle.write(">chrZ\nACGTACGTACGT\n")

    # Act & assert
    with pytest.raises(ValueError, match="none of the 2 contigs"):
        managed.managed_tracks(
            managed.Alignments(corpus.bams[0]), reference, str(track_dir)
        )
    assert [name for name in os.listdir(track_dir) if not name.endswith(".lock")] == []


def test_managed_tracks_should_pile_an_entry_up_once_under_concurrent_callers(
    corpus, tmp_path
):
    """Test that two processes asking for one entry share one pileup.

    Given:
        The entry's lock held by the test, and two spawned processes
        each asking for the same `Alignments` while it is held.
    When:
        The lock is released and both finish.
    Then:
        Nothing should be staged or committed while the lock is held,
        and both should answer the same paths and the same entry
        directory, which a second pileup would have replaced.
    """

    # Arrange
    spec = managed.Alignments(corpus.bams[0])
    track_dir = str(tmp_path / "tracks")
    os.makedirs(track_dir)
    key = managed.alignments_key(spec, corpus.fasta)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    children = [
        context.Process(
            target=pile_up_in_child,
            args=(corpus.bams[0], corpus.fasta, track_dir, queue),
        )
        for _ in range(2)
    ]

    # Act
    try:
        with open(os.path.join(track_dir, key + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for child in children:
                child.start()
            time.sleep(_LOCK_HOLD_SECONDS)
            during = sorted(os.listdir(track_dir))
        for child in children:
            child.join(timeout=_CHILD_TIMEOUT)
        exit_codes = [child.exitcode for child in children]
        answers = [queue.get_nowait() for child in children if child.exitcode == 0]
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()

    # Assert
    assert during == [key + ".lock"]
    assert exit_codes == [0, 0]
    (first_paths, first_inode), (second_paths, second_inode) = answers
    assert first_paths == second_paths
    assert first_inode == second_inode


def test_resolve_alignments_should_answer_specifications_without_one_unchanged():
    """Test that specifications with no `Alignments` are not looked at.

    Given:
        Tracks and group sizes holding no `Alignments`, and ``None``
        with no group sizes, as two specifications.
    When:
        They are resolved with a directory callable that must not be
        called.
    Then:
        Each should be answered as given, in order.
    """

    # Arrange
    def track_dir_factory():
        raise AssertionError("no directory is needed")

    # Act
    resolved = managed.resolve_alignments(
        [(["a", "b", "c"], [1, 2]), (None, None)], "ref.fa", track_dir_factory
    )

    # Assert
    assert resolved == [(["a", "b", "c"], [1, 2]), (None, None)]


@settings(max_examples=100)
@given(
    specifications=st.lists(
        st.one_of(
            st.none(),
            _group_layouts.flatmap(
                lambda sizes: st.tuples(
                    st.lists(
                        st.text(min_size=1, max_size=4),
                        min_size=sum(sizes),
                        max_size=sum(sizes),
                    ),
                    st.just(sizes),
                )
            ),
        ),
        max_size=3,
    )
)
def test_resolve_alignments_should_answer_any_specification_without_one_unchanged(
    specifications,
):
    """Test the identity on every specification that holds no `Alignments`.

    Given:
        Up to three specifications, each ``None`` or any group layout
        filled with plain track names, and a directory callable that
        must not be called.
    When:
        They are resolved.
    Then:
        Each should be answered as given, in order.
    """

    # Arrange
    def track_dir_factory():
        raise AssertionError("no directory is needed")

    given = [
        (None, None) if spec is None else (list(spec[0]), list(spec[1]))
        for spec in specifications
    ]

    # Act
    resolved = managed.resolve_alignments(given, "ref.fa", track_dir_factory)

    # Assert
    assert resolved == given


def test_resolve_alignments_should_expand_each_alignments_into_its_group(
    corpus, tmp_path
):
    """Test where the tracks of an `Alignments` land and how its group grows.

    Given:
        A track; a stranded `Alignments` as a group of its own; and a
        group of a track and an unstranded `Alignments`; and a second
        specification of the unstranded one alone.
    When:
        The two are resolved.
    Then:
        The stranded one should become a group of its two tracks, and
        the unstranded one its single track inside the group that held
        it, every other track in place, and the second specification
        its one track.
    """

    # Arrange
    track_dir = str(tmp_path / "tracks")
    stranded = managed.Alignments(corpus.bams[0])
    unstranded = managed.Alignments(corpus.bams[1], Aggregate(unstranded=True))

    # Act
    first, second = managed.resolve_alignments(
        [(["t0", stranded, "t1", unstranded], [1, 1, 2]), ([unstranded], [1])],
        corpus.fasta,
        lambda: track_dir,
    )

    # Assert
    plus, minus = managed.managed_tracks(stranded, corpus.fasta, track_dir)
    (pooled,) = managed.managed_tracks(unstranded, corpus.fasta, track_dir)
    assert first == (["t0", plus, minus, "t1", pooled], [1, 2, 2])
    assert second == ([pooled], [1])


@pytest.mark.parametrize(
    "group",
    [
        ["t0", managed.Alignments("a.bam")],
        [managed.Alignments("a.bam"), managed.Alignments("b.bam")],
    ],
    ids=["with-a-track", "with-another"],
)
def test_resolve_alignments_should_raise_for_a_stranded_alignments_in_a_group(group):
    """Test the refusal of a stranded pair a mirror would not swap with itself.

    Given:
        A first specification holding an `Alignments` that would be
        piled up, and a second whose group holds a stranded
        `Alignments` beside a track, or beside another stranded
        `Alignments`, with a directory callable that must not be
        called.
    When:
        The two are resolved.
    Then:
        It should raise `ValueError` saying a stranded `Alignments` is a
        group of its own, before anything is piled up for either.
    """

    # Arrange
    def track_dir_factory():
        raise AssertionError("nothing is piled up for a refused specification")

    # Act & assert
    with pytest.raises(ValueError, match="a signal group of its own"):
        managed.resolve_alignments(
            [([managed.Alignments("s.bam")], [1]), (group, [len(group)])],
            "ref.fa",
            track_dir_factory,
        )


def test_managed_track_dir_factory_should_answer_a_named_directory_and_keep_it(tmp_path):
    """Test that a directory the caller names is used and outlives the block.

    Given:
        A managed track directory named as a `pathlib.Path`.
    When:
        The block answers it and exits.
    Then:
        It should answer the path as a `str`, and the block should
        remove nothing.
    """

    # Arrange
    named = tmp_path / "tracks"
    named.mkdir()

    # Act
    with managed.managed_track_dir_factory(named) as track_dir_factory:
        answered = track_dir_factory()

    # Assert
    assert answered == str(named)
    assert named.is_dir()


def test_managed_track_dir_factory_should_remove_a_scratch_directory_on_exit(
    tmp_path, monkeypatch
):
    """Test that a scratch directory is made under the scratch root and removed.

    Given:
        No managed track directory, with a scratch root named in the
        environment.
    When:
        The block answers its directory twice, a file is written into
        it, and the block exits.
    Then:
        Both answers should be one directory under the scratch root,
        and the root should be empty afterwards.
    """

    # Arrange
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("QUADRAT_SCRATCH_DIR", str(root))

    # Act
    with managed.managed_track_dir_factory(None) as track_dir_factory:
        first, second = track_dir_factory(), track_dir_factory()
        pathlib.Path(first, "entry").write_text("x")

    # Assert
    assert first == second and os.path.dirname(first) == str(root)
    assert list(root.iterdir()) == []


def test_managed_track_dir_factory_should_remove_a_scratch_directory_when_the_block_raises(
    tmp_path, monkeypatch
):
    """Test that a scratch directory is removed when the block exits by an exception.

    Given:
        No managed track directory, with a scratch root named in the
        environment.
    When:
        The factory is called, a file is written into its directory,
        and the block raises.
    Then:
        The exception should propagate, and the scratch root should be
        empty afterwards.
    """

    # Arrange
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("QUADRAT_SCRATCH_DIR", str(root))

    # Act
    with pytest.raises(ValueError, match="a failed pileup"):
        with managed.managed_track_dir_factory(None) as track_dir_factory:
            pathlib.Path(track_dir_factory(), "partial").write_text("x")
            raise ValueError("a failed pileup")

    # Assert
    assert list(root.iterdir()) == []


def test_managed_track_dir_factory_should_make_nothing_when_never_asked(
    tmp_path, monkeypatch
):
    """Test that a block that piles nothing up makes no scratch directory.

    Given:
        No managed track directory, under a scratch root named in the
        environment.
    When:
        The block is entered and exits without being asked.
    Then:
        The scratch root should hold nothing at any point.
    """

    # Arrange
    root = tmp_path / "scratch"
    root.mkdir()
    monkeypatch.setenv("QUADRAT_SCRATCH_DIR", str(root))

    # Act & assert
    with managed.managed_track_dir_factory(None):
        assert list(root.iterdir()) == []
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("track_dir", [False, True, 7, ["tracks"]])
def test_managed_track_dir_factory_should_raise_type_error_for_a_directory_that_is_not_one(
    track_dir,
):
    """Test the refusal of a managed track directory that is not a path.

    Given:
        ``False``, ``True``, a number and a `list`.
    When:
        A block is entered with each.
    Then:
        It should raise `TypeError` on entry.
    """

    # Act & assert
    with pytest.raises(TypeError, match="track_dir is a directory or None"):
        with managed.managed_track_dir_factory(track_dir):
            pass
