"""Tests for the track format's readers and the paths it is built from.

The readers are tested for what they answer over each shape a track
takes and for the guards that refuse a track that is incomplete or no
track at all.

Every directory built by hand here is built through `manifest_path` and
`track_chrom_path` rather than by spelling the layout out, so the
format is stated once, in the tests of those two functions.
"""

import importlib
import json
import math
import os
import pathlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import quadrat
from quadrat import tables
from tests import FILE_BACKED


def manifest_for(table, rows, files):
    """Write a manifest beside ``table`` and return the record written."""

    record = {"version": tables.MANIFEST_VERSION, "rows": rows, "files": files}
    pathlib.Path(tables.manifest_path(str(table))).write_text(json.dumps(record))
    return record


def chrom_file(table, chrom):
    """Create an empty per-contig file and return its manifest key."""

    path = tables.track_chrom_path(str(table), chrom)
    pathlib.Path(path).write_bytes(b"")
    return os.path.basename(path)


def truncated_table(root, suffix=None):
    """A track directory whose manifest lists a file that is not there."""

    table = root / ("x" + (tables.TRACK_SUFFIX if suffix is None else suffix))
    table.mkdir()
    present = chrom_file(table, "chr1")
    absent = os.path.basename(tables.track_chrom_path(str(table), "chr2"))
    manifest_for(table, 2, {present: 1, absent: 1})
    return table


def declared_table(root, **fields):
    """A complete one-contig track directory whose record carries ``fields``.

    The fields replace or join the ones `manifest_for` writes, so a test
    can give a record any ``version`` and any ``signal``, or none.
    """

    table = root / ("declared" + tables.TRACK_SUFFIX)
    table.mkdir()
    record = manifest_for(table, 1, {chrom_file(table, "chr1"): 1})
    record.update(fields)
    pathlib.Path(tables.manifest_path(str(table))).write_text(json.dumps(record))
    return str(table)


def listed_table(root, chroms, rows=None, suffix=None):
    """A track directory whose manifest lists every file it holds."""

    table = root / ("listed" + (tables.TRACK_SUFFIX if suffix is None else suffix))
    table.mkdir()
    files = {chrom_file(table, chrom): 1 for chrom in chroms}
    manifest_for(table, len(files) if rows is None else rows, files)
    return table


def test_manifest_path_should_sit_beside_the_track_directory(tmp_path):
    """Test where the completeness record is kept.

    Given:
        A path to a track directory.
    When:
        Its manifest path is derived.
    Then:
        It should sit beside the directory rather than inside it.

    Beside is load-bearing: a track directory must hold Parquet and
    nothing else, because polars refuses to scan a directory of mixed
    extensions, and so does anything else pointed at the dataset.
    """

    # Arrange
    table = str(tmp_path / ("x" + tables.TRACK_SUFFIX))

    # Act
    beside = tables.manifest_path(table)

    # Assert
    assert os.path.dirname(beside) == os.path.dirname(table)
    assert not beside.startswith(table + os.sep)
    assert beside.endswith(tables.MANIFEST_SUFFIX)


def test_manifest_path_should_accept_a_path_object(tmp_path):
    """Test that a path-like track is named the same way.

    Given:
        A track directory named by a `pathlib.Path`.
    When:
        Its manifest path is derived.
    Then:
        It should be the string the equivalent string form gives.
    """

    # Arrange
    table = tmp_path / ("x" + tables.TRACK_SUFFIX)

    # Act & assert
    assert tables.manifest_path(table) == tables.manifest_path(str(table))


def test_track_chrom_path_should_name_one_contigs_file_inside_the_directory(tmp_path):
    """Test where one contig's rows are kept.

    Given:
        A track directory and an ordinary contig name.
    When:
        That contig's file path is derived.
    Then:
        It should be a Parquet file inside the directory, named for the
        contig.
    """

    # Arrange
    table = str(tmp_path / ("x" + tables.TRACK_SUFFIX))

    # Act
    path = tables.track_chrom_path(table, "chr1")

    # Assert
    assert path == os.path.join(table, "chr1.parquet")


def test_track_chrom_path_should_escape_a_contig_name_that_is_not_safe(tmp_path):
    """Test that an awkward contig name still yields one filename.

    Given:
        A contig name holding a slash, a colon and a space, as HLA and
        patch contigs do.
    When:
        That contig's file path is derived.
    Then:
        It should escape them into a single valid filename inside the
        directory, rather than a nested path or a name the filesystem
        refuses.
    """

    # Arrange
    table = str(tmp_path / ("x" + tables.TRACK_SUFFIX))
    pathlib.Path(table).mkdir()

    # Act
    path = tables.track_chrom_path(table, "HLA-A*01:01/weird name")

    # Assert
    assert os.path.dirname(path) == table
    assert "/" not in os.path.basename(path)[: -len(".parquet")]
    pathlib.Path(path).write_bytes(b"")
    assert os.path.isfile(path)


def test_read_manifest_should_return_the_record_beside_a_track_directory(tmp_path):
    """Test the record a readable manifest comes back as.

    Given:
        A track directory with a manifest beside it.
    When:
        The manifest is read.
    Then:
        It should return the parsed record.
    """

    # Arrange
    table = truncated_table(tmp_path)

    # Act & assert
    assert tables.read_manifest(str(table))["rows"] == 2


def test_read_manifest_should_find_a_manifest_beside_a_path_object(tmp_path):
    """Test that a path-like track finds its own record.

    Given:
        A track directory named by a `pathlib.Path`.
    When:
        Its manifest is read.
    Then:
        It should return the record rather than report it absent.

    `read_manifest` carried the same ``isinstance(str)`` probe the
    completeness gate did, so it answered ``None`` -- "no manifest" --
    for a manifest that exists and is perfectly readable.
    """

    # Arrange
    table = truncated_table(tmp_path)

    # Act & assert
    assert tables.read_manifest(str(table)) == tables.read_manifest(pathlib.Path(table))
    assert tables.read_manifest(pathlib.Path(table))["rows"] == 2


def test_read_manifest_should_find_the_record_of_a_track_under_the_former_suffix(
    tmp_path,
):
    """Test that a track written by cherimoya still finds its record.

    Given:
        A track directory carrying cherimoya's ``.cuts.parquet`` suffix.
    When:
        Its manifest is read.
    Then:
        It should return the record, `manifest_path` deriving the name
        from the path it was given rather than from the current suffix.
    """

    # Arrange
    legacy = truncated_table(tmp_path, suffix=".cuts.parquet")

    # Act & assert
    assert not str(legacy).endswith(tables.TRACK_SUFFIX)
    assert tables.read_manifest(str(legacy))["rows"] == 2


def test_require_complete_should_raise_when_the_path_does_not_exist(tmp_path):
    """Test the refusal of a track that is not there.

    Given:
        A path naming no file or directory.
    When:
        Completeness is required.
    Then:
        It should raise rather than pass the path back.

    A mistyped or since-deleted table is the input this gate exists to
    refuse, and it used to get the widest pass of any: the guard asked
    whether the path was a directory and returned it unchanged when it
    was not.
    """

    # Act & assert
    with pytest.raises(ValueError, match="no such file or directory"):
        tables.require_complete(str(tmp_path / ("absent" + tables.TRACK_SUFFIX)))


@pytest.mark.parametrize("source", [None, 42, object()])
def test_require_complete_should_raise_when_given_something_that_is_not_a_track(
    source,
):
    """Test the refusal of an object that is no kind of track.

    Given:
        A value that is neither a path nor an in-memory frame.
    When:
        Completeness is required.
    Then:
        It should raise `TypeError` naming the type it got.

    The docstring says there is deliberately no override and names two
    pass-throughs. Returning anything else unchecked is a third.
    """

    # Act & assert
    with pytest.raises(TypeError, match="require_complete verifies"):
        tables.require_complete(source)


def test_require_complete_should_raise_when_the_manifest_lists_an_absent_file(
    tmp_path,
):
    """Test the refusal of a track that stopped partway.

    Given:
        A track directory whose manifest lists a file that is not
        there.
    When:
        Completeness is required.
    Then:
        It should report the table as truncated.

    A reader returns nothing for a contig whose file is absent, and that
    is correct for a contig with no rows, so a writer killed partway
    leaves a directory that reads as valid. The manifest, written last,
    is the only thing that tells the two apart.
    """

    # Arrange
    table = truncated_table(tmp_path)

    # Act & assert
    with pytest.raises(ValueError, match="truncated"):
        tables.require_complete(str(table))


def test_require_complete_should_count_the_absent_files_it_does_not_list(tmp_path):
    """Test how the refusal names more absent files than it spells out.

    Given:
        A track directory whose manifest lists seven files, none of
        which is there.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` spelling out the first five in
        order and counting the two it leaves out.
    """

    # Arrange
    table = tmp_path / ("x" + tables.TRACK_SUFFIX)
    table.mkdir()
    absent = [
        os.path.basename(tables.track_chrom_path(str(table), "chr{}".format(index)))
        for index in range(1, 8)
    ]
    manifest_for(table, 7, {name: 1 for name in absent})

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"\(`chr1\.parquet`, `chr2\.parquet`, `chr3\.parquet`, "
        r"`chr4\.parquet`, `chr5\.parquet` and 2 more\)",
    ):
        tables.require_complete(str(table))


def test_require_complete_should_raise_when_a_truncated_track_is_a_path_object(
    tmp_path,
):
    """Test that a path-like track is verified rather than waved through.

    Given:
        A truncated track named by a `pathlib.Path`.
    When:
        Completeness is required.
    Then:
        It should raise, as it does for the same track named by a
        string.

    `pathlib.Path` is the path type a caller reaches for first, and the
    ``isinstance(str)`` probe treated it as a non-path -- so the gate
    answered "nothing to check" for exactly the shape most likely to
    arrive.
    """

    # Arrange
    table = truncated_table(tmp_path)

    # Act & assert
    with pytest.raises(ValueError, match="truncated"):
        tables.require_complete(pathlib.Path(table))


def test_require_complete_should_verify_a_track_under_the_former_suffix(tmp_path):
    """Test that a track written by cherimoya is still verifiable.

    Given:
        A complete track directory carrying cherimoya's
        ``.cuts.parquet`` suffix.
    When:
        Completeness is required.
    Then:
        It should verify it, the suffix being a write-side convention
        that nothing reads back.

    A reader is handed a path and globs inside it, so renaming what this
    package *writes* cannot strand a table anyone already has. That
    asymmetry is what the rename rests on.
    """

    # Arrange
    legacy = tmp_path / ("x" + ".cuts.parquet")
    legacy.mkdir()
    manifest_for(legacy, 1, {chrom_file(legacy, "chr1"): 1})

    # Act & assert
    assert not str(legacy).endswith(tables.TRACK_SUFFIX)
    assert tables.require_complete(str(legacy)) == str(legacy)


def test_require_complete_should_raise_when_the_row_total_disagrees(tmp_path):
    """Test the one internal consistency the record has.

    Given:
        A track whose manifest records a row total that disagrees with
        its own per-file counts.
    When:
        Completeness is required.
    Then:
        It should report that the record does not add up.

    The writer increments the total and records each file's height from
    the same frame slice, so the two agree by construction. A
    disagreement means the record did not come from one uninterrupted
    write -- two manifests merged, a hand edit, or a writer that is not
    this one.
    """

    # Arrange
    table = listed_table(tmp_path, ["chr1", "chr2"], rows=99)

    # Act & assert
    with pytest.raises(ValueError, match="does not add up"):
        tables.require_complete(str(table))


def test_require_complete_should_raise_when_a_float_row_total_disagrees(tmp_path):
    """Test that the consistency check is not skipped by a float total.

    Given:
        A track whose manifest records that same disagreeing total as a
        float rather than an integer.
    When:
        Completeness is required.
    Then:
        It should raise exactly as it does for the integer, a record
        that does not add up describing no finished write either way.

    JSON has one numeric type, so a producer that writes ``2.0`` is
    writing a row count. The guard tested for ``int``, so a float total
    skipped the check entirely and a truncated table verified clean.
    """

    # Arrange
    table = listed_table(tmp_path, ["chr1", "chr2"], rows=99.0)

    # Act & assert
    with pytest.raises(ValueError, match="does not add up"):
        tables.require_complete(str(table))


def test_require_complete_should_return_the_path_when_no_row_counts_are_recorded(
    tmp_path,
):
    """Test the pass the ``int`` guard was actually written for.

    Given:
        A manifest listing its files with no row counts, as a window
        cache sharing this schema does.
    When:
        Completeness is required.
    Then:
        It should return the path, there being nothing to add up.
    """

    # Arrange
    table = tmp_path / ("cache" + tables.TRACK_SUFFIX)
    table.mkdir()
    manifest_for(table, None, {chrom_file(table, "chr1"): None})

    # Act & assert
    assert tables.require_complete(str(table)) == str(table)


def test_require_complete_should_return_an_in_memory_frame_unchanged():
    """Test the documented pass-through for a frame.

    Given:
        An in-memory polars frame, eager and lazy.
    When:
        Completeness is required.
    Then:
        It should return the frame unchanged, a frame having no partial
        state to verify.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame({"chrom": ["chr1"], "pos": [1], "value": [1.0]})

    # Act & assert
    assert tables.require_complete(frame) is frame
    assert tables.require_complete(frame.lazy()) is not None


def test_require_complete_should_return_the_path_of_a_single_file_table(tmp_path):
    """Test the documented pass-through for a plain file.

    Given:
        A path naming a single Parquet file rather than a directory.
    When:
        Completeness is required.
    Then:
        It should return the path, a truncated Parquet file failing on
        its own footer rather than needing a manifest.
    """

    # Arrange
    table = tmp_path / "one.parquet"
    table.write_bytes(b"")

    # Act & assert
    assert tables.require_complete(str(table)) == str(table)


def test_track_contigs_should_name_the_contigs_its_manifest_lists(tmp_path):
    """Test which contigs a track reports.

    Given:
        A track directory whose manifest lists the files it holds.
    When:
        Its contigs are listed.
    Then:
        It should name those contigs and no others, answering from the
        filenames rather than reading a single row.
    """

    # Arrange
    table = listed_table(tmp_path, ["chr1", "chr2"])

    # Act & assert
    assert tables.track_contigs(str(table)) == {"chr1", "chr2"}


@pytest.mark.parametrize("listed", [True, False])
def test_track_contigs_should_list_the_contigs_of_a_track_under_the_former_suffix(
    tmp_path, listed
):
    """Test the third reader criterion #5 names as former-name-safe.

    Given:
        A track directory written under cherimoya's ``.cuts.parquet``
        name, once with a manifest listing its files and once with
        none.
    When:
        Its contigs are listed.
    Then:
        It should name them from either arm.

    The criterion makes the suffix rename safe by verification rather
    than assertion and names three readers; two were pinned and this
    was not. It is also the least obviously safe of the three: it is
    the one that parses the filename rather than deriving a sibling
    path, and the one whose answer branches on whether the manifest
    lists files -- so both arms are covered rather than the one a
    directory written by this package would take.
    """

    # Arrange
    former = ".cuts.parquet"
    if listed:
        table = listed_table(tmp_path, ["chr1", "chr2"], suffix=former)
    else:
        table = tmp_path / ("bare" + former)
        table.mkdir()
        for chrom in ("chr1", "chr2"):
            chrom_file(table, chrom)

    # Act & assert
    assert tables.track_contigs(str(table)) == {"chr1", "chr2"}


def test_track_contigs_should_list_a_directory_named_by_a_path_object(tmp_path):
    """Test that a path-like track is listed rather than refused.

    Given:
        A track directory named by a `pathlib.Path`.
    When:
        Its contigs are listed.
    Then:
        It should answer as it does for the same directory named by a
        string.

    This carried the same ``isinstance(str)`` probe the completeness
    gate did, and the failure was worse than a wrong answer: the path
    fell through to the frame reader, which complained about in-memory
    frames from a function the caller never called.
    """

    # Arrange
    table = listed_table(tmp_path, ["chr1", "chr2"])

    # Act & assert
    assert tables.track_contigs(pathlib.Path(table)) == tables.track_contigs(str(table))


def test_require_complete_should_distinguish_a_corrupt_manifest_from_an_absent_one(
    tmp_path,
):
    """Test that the refusal names the state the caller is actually in.

    Given:
        One track whose manifest is absent and one whose manifest is
        present but not valid JSON.
    When:
        Completeness is required of each.
    Then:
        It should refuse both, and only the first should be told the
        record is not there.

    `read_manifest` collapses absent, unreadable and non-dict to
    ``None``, which this function is entitled to treat alike. The
    message is not: the two states have different fixes, and telling
    the second caller their file is missing sends them looking for the
    wrong thing.
    """

    # Arrange
    absent = tmp_path / ("absent" + tables.TRACK_SUFFIX)
    absent.mkdir()
    chrom_file(absent, "chr1")
    corrupt = tmp_path / ("corrupt" + tables.TRACK_SUFFIX)
    corrupt.mkdir()
    chrom_file(corrupt, "chr1")
    pathlib.Path(tables.manifest_path(str(corrupt))).write_text("{not json")

    # Act
    with pytest.raises(ValueError) as absent_refusal:
        tables.require_complete(str(absent))
    with pytest.raises(ValueError) as corrupt_refusal:
        tables.require_complete(str(corrupt))

    # Assert
    assert "has no" in str(absent_refusal.value)
    assert "cannot be read as a manifest record" in str(corrupt_refusal.value)
    assert "has no" not in str(corrupt_refusal.value)


def test_require_complete_should_distinguish_a_manifest_that_is_not_an_object(tmp_path):
    """Test the refusal for a record whose top-level value is a list.

    Given:
        A track whose manifest file holds a JSON array.
    When:
        Completeness is required.
    Then:
        It should refuse as a malformed record rather than as a missing
        one.
    """

    # Arrange
    table = tmp_path / ("array" + tables.TRACK_SUFFIX)
    table.mkdir()
    chrom_file(table, "chr1")
    pathlib.Path(tables.manifest_path(str(table))).write_text('["chr1.parquet"]')

    # Act & assert
    with pytest.raises(ValueError, match="cannot be read as a manifest record"):
        tables.require_complete(str(table))


def test_require_complete_should_raise_value_error_when_files_is_not_a_mapping(
    tmp_path,
):
    """Test that a malformed ``files`` is refused rather than crashing.

    Given:
        A manifest whose ``files`` is a list of filenames.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError`, which is what this function
        documents, rather than the `AttributeError` that reaching
        ``.values()`` on a list produces.

    Criterion #5 makes this record an interoperability contract with
    two named third-party writers, and a list is the shape to get wrong
    first -- the keys alone look like the whole of the information. A
    validator that crashes with a type error on a malformed record is
    telling the writer nothing they can act on.
    """

    # Arrange
    table = tmp_path / ("listy" + tables.TRACK_SUFFIX)
    table.mkdir()
    present = chrom_file(table, "chr1")
    manifest_for(table, 1, [present])

    # Act & assert
    with pytest.raises(ValueError, match="`files` is list"):
        tables.require_complete(str(table))


def test_require_complete_should_raise_when_a_row_total_has_no_counts_to_support_it(
    tmp_path,
):
    """Test that one absent count cannot excuse the whole row sum.

    Given:
        A manifest recording a row total while one of its two files
        declares no count.
    When:
        Completeness is required.
    Then:
        It should refuse, naming the file that declares nothing.

    A single unusable value used to disable the check for the whole
    record, so a manifest claiming any total at all passed. That is
    backwards: a record stating a total it has no per-file heights to
    support is exactly a record that did not come from one
    uninterrupted write, which is the only internal consistency it has.
    """

    # Arrange
    table = tmp_path / ("partial" + tables.TRACK_SUFFIX)
    table.mkdir()
    first = chrom_file(table, "chr1")
    second = chrom_file(table, "chr2")
    manifest_for(table, 12345, {first: 1, second: None})

    # Act & assert
    with pytest.raises(ValueError, match="declare no count"):
        tables.require_complete(str(table))


def test_require_complete_should_count_the_uncounted_files_it_does_not_list(tmp_path):
    """Test how the refusal names more uncounted files than it spells out.

    Given:
        A manifest recording a row total while none of its seven files
        declares a count.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` spelling out the first five in
        order and counting the two it leaves out.
    """

    # Arrange
    table = tmp_path / ("partial" + tables.TRACK_SUFFIX)
    table.mkdir()
    names = [chrom_file(table, "chr{}".format(index)) for index in range(1, 8)]
    manifest_for(table, 7, {name: None for name in names})

    # Act & assert
    with pytest.raises(
        ValueError,
        match=r"declare no count \(`chr1\.parquet`, `chr2\.parquet`, "
        r"`chr3\.parquet`, `chr4\.parquet`, `chr5\.parquet` and 2 more\)",
    ):
        tables.require_complete(str(table))


def test_require_complete_should_accept_a_table_built_from_the_published_rules(
    tmp_path,
):
    """Test that the documented contract is sufficient to conform to.

    Given:
        A track directory and manifest built using only the published
        surface -- `MANIFEST_VERSION`, `manifest_path`,
        `track_chrom_path` and the field list `read_manifest`
        documents -- and nothing private.
    When:
        Completeness is required.
    Then:
        It should return the path.

    Criterion #5 names two third-party writers for this format, so the
    question this answers is the one the contract exists for: can a
    writer that reads only the documentation produce a table this
    accepts? Every name below is in ``quadrat.__all__``; the test fails
    to import rather than to assert if one stops being published.
    """

    # Arrange
    table = tmp_path / ("conforming" + quadrat.TRACK_SUFFIX)
    table.mkdir()
    heights = {"chr1": 3, "chr2": 2}
    files = {}
    for chrom, rows in heights.items():
        path = quadrat.track_chrom_path(str(table), chrom)
        pathlib.Path(path).write_bytes(b"")
        files[os.path.basename(path)] = rows
    record = {
        "version": quadrat.MANIFEST_VERSION,
        "rows": sum(files.values()),
        "files": files,
        "parameters": {"written_by": "a third party"},
    }
    pathlib.Path(quadrat.manifest_path(str(table))).write_text(json.dumps(record))

    # Act & assert
    assert quadrat.require_complete(str(table)) == str(table)
    assert quadrat.track_contigs(str(table)) == set(heights)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"version": 1, "signal": "counts"},
        {"version": 2, "signal": "coverage_deltas", "scale": 0.1},
    ],
    ids=["absent", "counts", "coverage_deltas"],
)
def test_require_complete_should_accept_each_signal_at_its_own_version(tmp_path, fields):
    """Test the pairings of signal and version a record may declare.

    Given:
        A complete track whose record declares no signal at version 1,
        counts at version 1, or coverage deltas at version 2 with a
        scale.
    When:
        Completeness is required.
    Then:
        It should return the path, an absent signal reading as counts.
    """

    # Arrange
    table = declared_table(tmp_path, **fields)

    # Act & assert
    assert tables.require_complete(table) == table


@pytest.mark.parametrize(
    "fields",
    [
        {"version": 1, "signal": "coverage_deltas"},
        {"version": 2, "signal": "counts"},
        {"version": 2},
    ],
    ids=["deltas_at_1", "counts_at_2", "absent_at_2"],
)
def test_require_complete_should_raise_when_the_version_is_not_the_signals_own(
    tmp_path, fields
):
    """Test that a signal and a version must agree.

    Given:
        A complete track declaring coverage deltas at version 1, which
        a reader that knows only version 1 would read as counts, or
        counts at version 2, declared or by absence.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` naming the version that signal is
        written at.
    """

    # Arrange
    table = declared_table(tmp_path, **fields)

    # Act & assert
    with pytest.raises(ValueError, match=r"track is written at version"):
        tables.require_complete(table)


@pytest.mark.parametrize(
    "version", [True, 1.0, 2.0], ids=["true", "one_float", "two_float"]
)
def test_require_complete_should_raise_when_the_version_is_not_an_integer(
    tmp_path, version
):
    """Test that a version equal to a known one but not an integer is refused.

    Given:
        A complete track of counts whose record declares its version as
        ``true`` or ``1.0``, both equal to 1 in Python, or as ``2.0``.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` naming the version, rather than
        read the record as one the format fixes.
    """

    # Arrange
    table = declared_table(tmp_path, version=version)

    # Act & assert
    with pytest.raises(ValueError, match=r"declares manifest schema version"):
        tables.require_complete(table)


@pytest.mark.parametrize("signal", ["methylation", 5, ["counts"], None])
def test_require_complete_should_raise_when_the_signal_is_unknown(tmp_path, signal):
    """Test that a signal this does not know is refused, not defaulted.

    Given:
        A complete track whose record names a signal that is not one of
        the documented strings, including a ``null``.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` rather than read it as counts.
    """

    # Arrange
    table = declared_table(tmp_path, signal=signal)

    # Act & assert
    with pytest.raises(ValueError, match=r"whose signal is"):
        tables.require_complete(table)


@pytest.mark.parametrize(
    "scale",
    [None, "1.0", True, float("inf"), [1.0], 10**400],
    ids=["null", "string", "bool", "infinite", "list", "beyond-a-float"],
)
def test_require_complete_should_raise_when_a_coverage_record_has_no_usable_scale(
    tmp_path, scale
):
    """Test that a coverage-deltas record must state a finite scale.

    Given:
        A complete track declaring coverage deltas whose ``scale`` is
        ``null``, a string, a boolean, infinite, a list, or an integer
        too large to hold as a float.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` naming the scale, because the
        deltas are unscaled and the record is what gives them a
        magnitude.
    """

    # Arrange
    table = declared_table(tmp_path, version=2, signal="coverage_deltas", scale=scale)

    # Act & assert
    with pytest.raises(ValueError, match=r"declares coverage deltas with scale"):
        tables.require_complete(table)


def test_require_complete_should_raise_when_a_coverage_record_omits_its_scale(
    tmp_path,
):
    """Test that a coverage-deltas record without a scale is refused.

    Given:
        A complete track declaring coverage deltas at version 2 and no
        ``scale`` at all.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError`, the field being required on that
        record rather than defaulted.
    """

    # Arrange
    table = declared_table(tmp_path, version=2, signal="coverage_deltas")

    # Act & assert
    with pytest.raises(ValueError, match=r"declares coverage deltas with scale None"):
        tables.require_complete(table)


@pytest.mark.parametrize(
    "fields",
    [{"scale": 1.0}, {"signal": "counts", "scale": 2.0}],
    ids=["absent", "counts"],
)
def test_require_complete_should_raise_when_a_counts_record_carries_a_scale(
    tmp_path, fields
):
    """Test that a scale is refused on a record of counts.

    Given:
        A complete track of counts, by absence or declared, whose record
        carries a ``scale``.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError`, because counts are written scaled
        and the record cannot say whether to scale them again.
    """

    # Arrange
    table = declared_table(tmp_path, **fields)

    # Act & assert
    with pytest.raises(ValueError, match=r"declares a scale of .* on a 'counts' track"):
        tables.require_complete(table)


@pytest.mark.parametrize("version", [0, 3, "1", None])
def test_require_complete_should_raise_when_the_version_is_unknown(tmp_path, version):
    """Test the schema versions this reads.

    Given:
        A complete track whose record declares a version that is
        neither 1 nor 2.
    When:
        Completeness is required.
    Then:
        It should raise `ValueError` naming the versions it reads.
    """

    # Arrange
    table = declared_table(tmp_path, version=version)

    # Act & assert
    with pytest.raises(ValueError, match=r"reads versions `1`, `2`"):
        tables.require_complete(table)


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, ("counts", None)),
        (
            {"version": 2, "signal": "coverage_deltas", "scale": 0.25},
            ("coverage_deltas", 0.25),
        ),
    ],
    ids=["absent", "coverage_deltas"],
)
def test_track_signal_should_answer_the_signal_and_scale_a_record_declares(
    tmp_path, fields, expected
):
    """Test what a reader is told a track holds.

    Given:
        A track whose record declares no signal, or coverage deltas
        with a scale.
    When:
        Its signal is asked for.
    Then:
        It should answer counts with no scale for the absent one, and
        the declared signal and scale otherwise.
    """

    # Arrange
    table = declared_table(tmp_path, **fields)

    # Act & assert
    assert tables.track_signal(table) == expected


def test_track_signal_should_answer_counts_for_a_track_with_no_record(tmp_path):
    """Test the shapes that cannot declare a signal.

    Given:
        A single Parquet file and an in-memory frame.
    When:
        Each one's signal is asked for.
    Then:
        It should answer counts with no scale for each.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    single = tmp_path / "single.parquet"
    frame = polars.DataFrame({"chrom": ["chr1"], "pos": [1], "value": [1.0]})
    frame.write_parquet(single)

    # Act
    answers = [tables.track_signal(source) for source in (str(single), frame)]

    # Assert
    assert answers == [("counts", None)] * 2


def test_track_signal_should_raise_when_the_track_does_not_verify(tmp_path):
    """Test that the signal is not answered for an unverifiable track.

    Given:
        A track directory with no record beside it.
    When:
        Its signal is asked for.
    Then:
        It should raise `ValueError` as `require_complete` does, rather
        than answer counts for a directory that cannot say.
    """

    # Arrange
    bare = tmp_path / ("bare" + tables.TRACK_SUFFIX)
    bare.mkdir()

    # Act & assert
    with pytest.raises(ValueError, match=r"cannot be verified complete"):
        tables.track_signal(str(bare))


def test_track_signal_should_raise_when_the_signal_is_unknown(tmp_path):
    """Test that a reader is not told an unknown signal is counts.

    Given:
        A track whose record names a signal that is not documented.
    When:
        Its signal is asked for.
    Then:
        It should raise `ValueError` rather than answer counts.
    """

    # Arrange
    table = declared_table(tmp_path, signal="methylation")

    # Act & assert
    with pytest.raises(ValueError, match=r"whose signal is 'methylation'"):
        tables.track_signal(table)


class TestTrackWriter:
    """The writer's record of what its values mean."""

    @pytest.mark.parametrize("signal", ["coverage", None, ["counts"]])
    def test___init___should_refuse_a_signal_it_cannot_record(self, tmp_path, signal):
        """Test that a writer does not record a meaning no reader knows.

        Given:
            A signal that is not one of the documented strings,
            including an unhashable one, which the manifest's reader
            refuses with `ValueError` as well.
        When:
            A writer is constructed with it.
        Then:
            It should raise `ValueError` and stage nothing.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))

        # Act & assert
        with pytest.raises(ValueError, match="signal must be one of"):
            tables.TrackWriter(path, signal=signal)
        assert os.listdir(str(tmp_path)) == []

    @pytest.mark.parametrize(
        "scale",
        [None, float("nan"), True, "1", 10**400],
        ids=["none", "nan", "bool", "string", "beyond-a-float"],
    )
    def test___init___should_refuse_coverage_deltas_without_a_finite_scale(
        self, tmp_path, scale
    ):
        """Test that a writer of coverage deltas must record a scale.

        Given:
            Coverage deltas, with no scale, a scale that is not finite,
            one that is not a number, or an integer too large to hold
            as a float.
        When:
            A writer is constructed with them.
        Then:
            It should raise `ValueError` and stage nothing, because the
            deltas are written unscaled and the scale is the only record
            of their magnitude.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))

        # Act & assert
        with pytest.raises(ValueError, match="scale must be a finite number"):
            tables.TrackWriter(path, signal="coverage_deltas", scale=scale)
        assert os.listdir(str(tmp_path)) == []

    @settings(FILE_BACKED, max_examples=100)
    @given(
        scale=st.one_of(
            st.floats(), st.integers(min_value=-(10**400), max_value=10**400)
        )
    )
    def test___init___should_take_a_scale_exactly_when_a_float_holds_it_finitely(
        self, tmp_path, scale
    ):
        """Test the domain a writer of coverage deltas holds its scale to.

        Given:
            Any float, finite or not, and any integer up to 400 digits,
            including those too large to hold as a float.
        When:
            A writer of coverage deltas is constructed with it.
        Then:
            It should hold the equal `float` when that float is finite,
            and raise `ValueError` otherwise, never another exception.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))
        try:
            held = float(scale)
        except OverflowError:
            held = math.inf

        # Act & assert
        if math.isfinite(held):
            writer = tables.TrackWriter(path, signal="coverage_deltas", scale=scale)
            writer.abort()
            assert writer.scale == held
        else:
            with pytest.raises(ValueError, match="scale must be a finite number"):
                tables.TrackWriter(path, signal="coverage_deltas", scale=scale)

    def test___init___should_refuse_a_scale_for_counts(self, tmp_path):
        """Test that a writer of counts takes no scale.

        Given:
            Counts, and a scale.
        When:
            A writer is constructed with them.
        Then:
            It should raise `ValueError` and stage nothing, because
            counts are written already scaled.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))

        # Act & assert
        with pytest.raises(ValueError, match="scale must be None"):
            tables.TrackWriter(path, signal="counts", scale=2.0)
        assert os.listdir(str(tmp_path)) == []

    @pytest.mark.parametrize(
        "signal, scale, version", [("counts", None, 1), ("coverage_deltas", 0.1, 2)]
    )
    def test_commit_should_record_the_signal_at_its_version(
        self, tmp_path, signal, scale, version
    ):
        """Test that every record carries a signal, and the version it implies.

        Given:
            A writer told each signal, and a scale for coverage deltas.
        When:
            One frame is written and committed.
        Then:
            It should record that signal, at version 1 for counts and 2
            for coverage deltas, the scale beside coverage deltas alone,
            and the track should verify.
        """

        # Arrange
        polars = pytest.importorskip("polars")
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))
        frame = polars.DataFrame(
            {"chrom": ["chr1"], "pos": [3], "value": [1.0]},
            schema={"chrom": polars.Utf8, "pos": polars.Int64, "value": polars.Float64},
        )

        # Act
        with tables.TrackWriter(path, signal=signal, scale=scale) as writer:
            writer.write(frame)

        # Assert
        record = tables.read_manifest(path)
        assert (record["signal"], record["version"]) == (signal, version)
        assert record.get("scale") == scale and ("scale" in record) == (
            scale is not None
        )
        assert tables.require_complete(path) == path

    @settings(FILE_BACKED, max_examples=100)
    @given(scale=st.floats(allow_nan=False, allow_infinity=False))
    def test_commit_should_record_any_finite_scale_as_the_reader_answers_it(
        self, tmp_path, scale
    ):
        """Test the scale's round-trip through a track's record.

        Given:
            Any finite float as the scale of a writer of coverage
            deltas.
        When:
            It commits and the track's signal is read back.
        Then:
            It should answer coverage deltas at exactly that scale.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))

        # Act
        with tables.TrackWriter(path, signal="coverage_deltas", scale=scale):
            pass

        # Assert
        assert tables.track_signal(path) == ("coverage_deltas", scale)

    def test_commit_should_record_counts_when_no_signal_is_given(self, tmp_path):
        """Test that the signal is written unconditionally.

        Given:
            A writer given no signal.
        When:
            It commits.
        Then:
            Its record should still declare counts, at version 1.
        """

        # Arrange
        path = str(tmp_path / ("w" + tables.TRACK_SUFFIX))

        # Act
        with tables.TrackWriter(path):
            pass

        # Assert
        record = tables.read_manifest(path)
        assert (record["signal"], record["version"]) == ("counts", 1)


def test_track_directory_should_hold_parquet_and_nothing_else(tmp_path):
    """Test the invariant the manifest sits outside the directory for.

    Given:
        A track directory holding several contigs' files, with its
        manifest written.
    When:
        The directory is listed.
    Then:
        It should hold at least one entry, every entry a Parquet file,
        and the manifest should exist beside it.

    polars refuses to scan a directory of mixed extensions, and so does
    anything else pointed at the dataset, which is why the completeness
    record is a sibling of the directory rather than a member of it.
    """

    # Arrange
    table = tmp_path / ("held" + tables.TRACK_SUFFIX)
    table.mkdir()
    for chrom in ("chr1", "chr2", "chrUn_KI270302v1"):
        chrom_file(table, chrom)
    manifest_for(table, 0, {})

    # Act
    held = sorted(os.listdir(str(table)))

    # Assert
    assert held
    assert all(name.endswith(".parquet") for name in held), held
    assert os.path.exists(tables.manifest_path(str(table)))
    assert not os.path.dirname(tables.manifest_path(str(table))).endswith(
        tables.TRACK_SUFFIX
    )


# The acceptance sweep over the path shapes and the wrong types. One
# test per property rather than one per function, because the property
# *is* that the functions agree: a per-function test can pass while two
# of them answer differently, which is the state these were written
# against.
PATH_READERS = (
    "manifest_path",
    "read_manifest",
    "require_complete",
    "track_chrom_path",
    "track_paths",
    "track_contigs",
)


def call_reader(name, source):
    """Call one published table reader with ``source`` as its path."""

    function = getattr(quadrat, name, None) or getattr(tables, name)
    if name == "track_chrom_path":
        return function(source, "chr1")
    if name == "track_paths":
        return function(source, True)
    return function(source)


@pytest.mark.parametrize("name", PATH_READERS)
def test_the_table_readers_should_answer_alike_for_every_path_shape(name, tmp_path):
    """Test that a published reader does not care how a path is spelled.

    Given:
        One verified track directory, named as a ``str``, a
        `pathlib.Path` and a ``bytes``.
    When:
        Each published reader is called with each spelling.
    Then:
        It should answer identically for all three.

    `pileup_alignments` claimed this ("Normalized at the boundary, as
    every sibling that takes a path does ... each with a test pinning
    it") and no test exercised ``bytes`` anywhere: a grep over the
    suite for ``b"/``, ``b'/`` and ``os.fsencode`` returned nothing.
    What it covered for was real -- `manifest_path` built
    ``"b'/data/x.track.parquet'.manifest.json"`` out of a repr, and
    `bam_contigs` read a ``bytes`` path as an unreadable header -- and
    both are a published answer rather than a refusal, which is the
    failure a caller cannot see.
    """

    # Arrange
    table = listed_table(tmp_path, ("chr1", "chr2"))
    spellings = (str(table), pathlib.Path(str(table)), os.fsencode(str(table)))

    # Act
    answers = [call_reader(name, source) for source in spellings]

    # Assert
    expected = call_reader(name, str(table))
    for spelling, answer in zip(spellings, answers):
        if name == "require_complete":
            # It returns its argument, so compare the normalized form.
            assert os.fsdecode(answer) == str(table), spelling
        else:
            assert answer == expected, spelling


@pytest.mark.parametrize("name", PATH_READERS)
def test_the_table_readers_should_refuse_a_wrong_type_alike(name):
    """Test that one wrong type gets one answer across the surface.

    Given:
        An integer where a path belongs.
    When:
        Each published reader is called with it.
    Then:
        It should raise `TypeError` naming the parameter, so a caller
        meets the same verdict wherever they are.

    Four answers were in play. `manifest_path` returned
    ``"42.manifest.json"`` -- a path, for an integer, which no caller
    could tell from a real one. `read_manifest` returned ``None``, a
    fifth meaning its four documented causes do not carry, and one a
    reader acts on by treating an unverifiable table as one predating
    manifests. `track_chrom_path` and `track_paths` raised from inside
    a path builder, naming neither the function nor the argument. Only
    `require_complete` named both.
    """

    # Act & assert
    with pytest.raises(TypeError, match=name):
        call_reader(name, 42)


def test_track_contigs_should_refuse_a_non_track_in_its_own_words():
    """Test that the refusal describes the call the caller made.

    Given:
        An integer where a track belongs.
    When:
        `track_contigs` is asked for its contigs.
    Then:
        It should raise `TypeError` whose message says
        `track_contigs` reads a track and says nothing about signals.

    A caller who asked which contigs a track holds is told about that
    call, not about reading signals, an operation they did not invoke.
    """

    # Act
    with pytest.raises(TypeError) as caught:
        quadrat.track_contigs(42)

    # Assert
    message = str(caught.value)
    assert "track_contigs reads a track" in message
    assert "signals" not in message


def test_track_contigs_should_name_an_incomplete_install_when_polars_is_absent(mocker):
    """Test what a reader says when a base dependency will not import.

    Given:
        A reader whose work needs polars, and no importable polars.
    When:
        It is called.
    Then:
        It should raise `ImportError` saying the environment is
        incomplete rather than naming an extra to install.

    polars is a base dependency, because reading a track is what the
    base install is for. So its absence is a broken environment rather
    than a minimal one, and pointing the caller at the ``pileup`` extra
    would send them to install something that does not contain it.

    Every deferred import goes through one seam, so this is also the
    test that the seam refuses rather than letting a bare
    ``ModuleNotFoundError`` out of the middle of a read.
    """

    # Arrange
    mocker.patch.object(importlib, "import_module", side_effect=ImportError("no polars"))

    # Act & assert
    with pytest.raises(ImportError, match=r"polars is declared.*incomplete"):
        tables.track_contigs(42)


def test_track_contigs_should_read_a_single_file_table(tmp_path):
    """Test that the legacy single-file shape still answers.

    Given:
        A track written as one Parquet file rather than a directory.
    When:
        Its contigs are read.
    Then:
        It should report what the ``chrom`` column holds.

    It is the shape the reference page records as readable but
    unverifiable.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    table = tmp_path / ("single" + tables.TRACK_SUFFIX)
    polars.DataFrame(
        {"chrom": ["chr1", "chr2"], "pos": [1, 2], "value": [1.0, 2.0]}
    ).write_parquet(table)

    # Act
    contigs = quadrat.track_contigs(str(table))

    # Assert
    assert contigs == {"chr1", "chr2"}


def test_track_contigs_should_read_both_frame_shapes():
    """Test that an in-memory frame answers whether lazy or eager.

    Given:
        The same three columns as a ``DataFrame`` and as a
        ``LazyFrame``.
    When:
        Their contigs are read.
    Then:
        Both should report the same contigs, which is what lets a
        caller run BAM to tensors without touching disk.
    """

    # Arrange
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame(
        {"chrom": ["chr1", "chr2"], "pos": [1, 2], "value": [1.0, 2.0]}
    )

    # Act
    eager = quadrat.track_contigs(frame)
    lazy = quadrat.track_contigs(frame.lazy())

    # Assert
    assert eager == lazy == {"chr1", "chr2"}


def staged_track(path, rows=(("chr1", 1, 2.0),)):
    """Commit a one-contig track at ``path`` through the shared protocol."""

    staging = tables.stage_directory(path)
    with open(os.path.join(staging, "chr1.parquet"), "w") as handle:
        handle.write("x")
    tables.commit_directory(
        path,
        staging,
        {"version": tables.MANIFEST_VERSION, "files": {"chr1.parquet": None}},
    )
    return path


def siblings(directory, name):
    """Return the entries in ``directory`` other than ``name`` and its manifest."""

    return sorted(
        entry
        for entry in os.listdir(str(directory))
        if entry != name and entry != name + tables.MANIFEST_SUFFIX
    )


def test_stage_directory_should_sweep_a_staging_sibling_a_killed_writer_left(tmp_path):
    """Test what a new write does about an interrupted one's staging directory.

    Given:
        A staging sibling of the size of a whole channel, as a writer
        killed before its rename leaves.
    When:
        A new write is staged on the same path.
    Then:
        The sibling should be gone, because a crash loop would
        otherwise fill the disk a channel at a time.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    stale = path + ".staging.deadbeef"
    os.makedirs(stale)

    # Act
    tables.stage_directory(path)

    # Assert
    assert not os.path.exists(stale)


def test_stage_directory_should_sweep_a_discard_sibling_a_killed_commit_left(tmp_path):
    """Test what a new write does about an interrupted commit's discard directory.

    Given:
        A discard sibling, as a commit killed between its two renames
        leaves.
    When:
        A new write is staged on the same path.
    Then:
        The sibling should be gone, for the same reason as a staging
        one: nothing ever reads it and it is the size of a channel.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    stale = path + ".trash.deadbeef"
    os.makedirs(stale)

    # Act
    tables.stage_directory(path)

    # Assert
    assert not os.path.exists(stale)


def test_stage_directory_should_sweep_a_manifest_temporary_a_killed_commit_left(
    tmp_path,
):
    """Test what a new write does about an interrupted commit's record file.

    Given:
        The manifest's temporary file, as a commit killed between
        writing it and renaming it onto the record leaves.
    When:
        A new write is staged on the same path.
    Then:
        It should be gone. It is small, and for a track it is one stray
        file -- but a window cache commits through here under a cache
        directory, one per failed key, and the write that leaves it
        swallows its own failure, so nothing else would ever report or
        reap them.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    stale = path + tables.MANIFEST_SUFFIX + ".tmp"
    with open(stale, "w") as handle:
        handle.write("{")

    # Act
    tables.stage_directory(path)

    # Assert
    assert not os.path.exists(stale)


def test_stage_directory_should_not_sweep_the_directory_it_just_made(tmp_path):
    """Test the order of the sweep against the staging directory's creation.

    Given:
        A path with no siblings.
    When:
        A write is staged.
    Then:
        The staging directory should exist afterwards, because the
        sweep runs before it is made and would otherwise reap it.
    """

    # Act
    staging = tables.stage_directory(str(tmp_path / "t.track.parquet"))

    # Assert
    assert os.path.isdir(staging)


def test_stage_directory_should_sweep_only_its_own_siblings(tmp_path):
    """Test a path whose name carries glob metacharacters.

    Given:
        Two tracks whose names differ only in characters a glob reads
        as wildcards, each with a staging sibling.
    When:
        A write is staged on the first.
    Then:
        Only the first's sibling should be swept, because a track name
        is a caller's path and a cache entry's name is a hex digest
        under a directory the caller chose -- an unescaped glob would
        reap another write's staging directory or miss its own.
    """

    # Arrange
    mine = str(tmp_path / "t[1].track.parquet")
    theirs = str(tmp_path / "t1.track.parquet")
    os.makedirs(mine + ".staging.aaaa")
    os.makedirs(theirs + ".staging.bbbb")

    # Act
    tables.stage_directory(mine)

    # Assert
    assert not os.path.exists(mine + ".staging.aaaa")
    assert os.path.isdir(theirs + ".staging.bbbb")


def test_commit_directory_should_leave_a_verifiable_track_and_no_siblings(tmp_path):
    """Test what a successful commit leaves behind.

    Given:
        A staged directory and its record.
    When:
        It is committed.
    Then:
        The track should verify and nothing should sit beside it,
        because the discard copy is reaped only after the record lands.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")

    # Act
    staged_track(path)

    # Assert
    assert quadrat.require_complete(path) == path
    assert siblings(tmp_path, "t.track.parquet") == []


def test_commit_directory_should_replace_a_track_already_there(tmp_path):
    """Test a commit onto a path that already holds a track.

    Given:
        A committed track, and a second write staged on the same path.
    When:
        The second is committed.
    Then:
        The path should hold the second write and verify, with no
        discard or staging sibling left, so a rebuild in place is one
        step for every reader of it.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    staged_track(path)

    # Act
    staging = tables.stage_directory(path)
    with open(os.path.join(staging, "chr2.parquet"), "w") as handle:
        handle.write("y")
    tables.commit_directory(
        path,
        staging,
        {"version": tables.MANIFEST_VERSION, "files": {"chr2.parquet": None}},
    )

    # Assert
    assert quadrat.require_complete(path) == path
    assert os.listdir(path) == ["chr2.parquet"]
    assert siblings(tmp_path, "t.track.parquet") == []


def test_commit_directory_should_restore_the_previous_track_when_the_swap_fails(
    tmp_path, mocker
):
    """Test a rebuild that cannot complete over a track that already exists.

    Given:
        A committed track, and a second write whose swap into place
        fails after the previous directory has been moved aside.
    When:
        The commit runs.
    Then:
        It should raise, and the previous track's files should be back
        where they were, with no staging or discard sibling left --
        because a rebuild that fails must leave what was there rather
        than destroy it, and a track costs minutes to rebuild. The
        restored track should not verify, because its record was
        removed before the swap and the commit cannot reconstruct one it
        did not write.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    staged_track(path)
    staging = tables.stage_directory(path)
    with open(os.path.join(staging, "chr2.parquet"), "w") as handle:
        handle.write("y")
    real = os.replace

    def failing(src, dst):
        if str(src) == staging:
            raise OSError("no swap")
        return real(src, dst)

    mocker.patch.object(tables.os, "replace", side_effect=failing)

    # Act
    with pytest.raises(OSError, match="no swap"):
        tables.commit_directory(
            path,
            staging,
            {"version": tables.MANIFEST_VERSION, "files": {"chr2.parquet": None}},
        )

    # Assert
    assert os.listdir(path) == ["chr1.parquet"]
    assert siblings(tmp_path, "t.track.parquet") == []
    with pytest.raises(ValueError, match="cannot be verified complete"):
        quadrat.require_complete(path)


def test_commit_directory_should_leave_an_unverifiable_track_when_the_record_fails(
    tmp_path,
):
    """Test a commit whose record cannot be written after the swap.

    Given:
        A staged directory and a record holding a value JSON cannot
        serialise.
    When:
        It is committed.
    Then:
        It should raise with the new directory already in place and no
        record, so the track fails verification rather than reading as
        complete -- which is the one direction a half-finished commit
        is allowed to fail in.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    staging = tables.stage_directory(path)
    with open(os.path.join(staging, "chr1.parquet"), "w") as handle:
        handle.write("x")

    # Act
    with pytest.raises(TypeError, match="not JSON serializable"):
        tables.commit_directory(
            path,
            staging,
            {"version": tables.MANIFEST_VERSION, "files": {"chr1.parquet": object()}},
        )

    # Assert
    assert os.listdir(path) == ["chr1.parquet"]
    assert not os.path.exists(tables.manifest_path(path))
    with pytest.raises(ValueError, match="cannot be verified complete"):
        quadrat.require_complete(path)


def test_commit_directory_should_replace_a_plain_file_sitting_where_it_goes(tmp_path):
    """Test a commit onto a path holding a file rather than a directory.

    Given:
        A plain file where the track directory goes, as a single-file
        table of the shape the readers also accept.
    When:
        A directory is committed there.
    Then:
        The file should be gone and the track should verify, because
        the path names a track and not a shape.
    """

    # Arrange
    path = str(tmp_path / "t.track.parquet")
    with open(path, "w") as handle:
        handle.write("single file")

    # Act
    staged_track(path)

    # Assert
    assert os.path.isdir(path)
    assert quadrat.require_complete(path) == path
