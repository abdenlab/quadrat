"""The track format, and the atomic writer that produces it.

A track is a directory of per-contig Parquet files holding
sorted ``(chrom, pos, value)`` rows, with a completeness manifest
beside it rather than inside it. Beside, because a track directory
must hold Parquet and nothing else: polars refuses to scan a directory
of mixed extensions, and so does anything else pointed at the dataset.

The format and ``MANIFEST_VERSION`` are deliberately unchanged from
cherimoya's. Tables either package writes are readable by the other,
and the alternative is re-piling every corpus to move between them.
"""

import glob
import json
import os
import shutil
import urllib.parse
import uuid

from quadrat._engine import fspath_or_none, import_on_first_use

# Suffix of a track directory. Write-side only: nothing reads it back.
# A reader is handed a path and globs `*.parquet` inside it, and
# `manifest_path` derives the record's name from the path it was given,
# so a directory written under cherimoya's older `.cuts.parquet` name
# reads through `read_manifest`, `require_complete` and
# `track_contigs` unchanged. Only new writes carry this name.
# public
TRACK_SUFFIX = ".track.parquet"


# Suffix of the completeness record that sits *beside* a track
# directory, and the schema version it declares. See
# `require_complete` for what it is for.
#
# Beside, not inside: a track directory must hold Parquet and nothing
# else, because polars refuses to scan a directory of mixed extensions
# and so does anything else pointed at the dataset. That invariant is
# pinned by `test_track_directory_holds_parquet_and_nothing_else`,
# which an earlier memory-mapped read index already broke once.
MANIFEST_SUFFIX = ".manifest.json"


# public
MANIFEST_VERSION = 1


# Parquet row-group size for the track files. See
# `TrackWriter` for why this number and not a rounder one.
_ROW_GROUP_SIZE = 10_000


# Prefixes for the staging and discard directories the atomic write uses.
# Both are siblings of the final path so the renames stay within one
# filesystem; `_sweep_stale` reaps any left behind by a killed writer.
_STAGING_INFIX = ".staging."


_TRASH_INFIX = ".trash."


#: The manifest is written beside its own name and renamed onto it, so
#: a commit interrupted between the two leaves this behind. A fixed
#: name rather than a hex-suffixed family, because only one commit at a
#: time can be renaming onto a given marker.
_MARKER_TMP_SUFFIX = ".tmp"


class TrackWriter(object):
    """One track, assembled from one or more frames.

    The pileup runs the genome in contig groups so that it holds only
    part of the aggregate at a time, and so hands a table over in
    several disjoint pieces rather than one. The staging directory, the
    file list and the commit belong to a *writer*, not to a path: two
    separate writers on the same path still replace one another
    wholesale, so a table assembled from several frames is
    byte-identical to the same rows written in one.

    No frame is retained. Every part written is a zero-copy ``slice`` of
    the caller's frame, so holding one would pin the whole group and
    give back the memory the grouping exists to save; only a filename
    and a row count are kept.

    Use it as a context manager. A writer abandoned without one leaks
    its staging directory until the next `_sweep_stale`, which is the
    same exposure a killed process has always had.

    .. rubric:: Implementation notes

    This class decides the track format, so the measurements behind
    it are recorded here rather than beside a caller.

    ``path`` keeps its ``.track.parquet`` name but is a *directory*, so
    every caller that builds these names from `TRACK_SUFFIX`
    is unaffected by the layout.

    Sorted per chromosome, with 10,000-row groups.

    The sorting was measured on ENCSR483RKN (26.2M rows): against a
    single file with 1,000,000-row groups a windowed read went from
    6.7 ms to 0.78 ms. That comparison moved two things at once,
    though -- the per-contig split and the group size together -- and
    the group size it landed on was 50,000. The split cuts rows per
    file about twelvefold, which moves where the optimum sits, so a
    group size chosen against the single-file layout does not carry
    over to this one.

    Re-derived here, over 300 random 1 kb window reads per point,
    page-cache warm, against sorted positions drawn uniformly over the
    contig:

    ==========  ==========  ============  =========
    rows/file   occupancy   best group    at 50,000
    ==========  ==========  ============  =========
    50,000      0.20%       5,000         2.86x
    2,200,000   0.88%       10,000        1.79x
    20,000,000  8.03%       20,000        1.00x
    ==========  ==========  ============  =========

    10,000 rather than the per-row optimum, because one constant has
    to serve every contig in one track and a track's contigs differ
    hundredfold in length. It is the optimum at ATAC cut-site density
    and within 6% of it on a small contig, where 50,000 is 2.86x
    worse because a 50,000-row contig is a single group and a windowed
    read scans the whole file.

    The old 10,000-regresses-to-2.2 ms figure is not reproduced and is
    not a disagreement: it was measured on the single-file layout,
    where 26.2M rows in one file put it in the dense column of the
    table above rather than the sparse one.

    The last row is the standing caveat. Occupancy, not depth, is what
    moves this -- past about 8% the optimum walks back up and 10,000
    costs 1.46x against 50,000 -- so a per-base or span-coverage
    aggregate wants a larger group than a cut-site one, and one
    constant cannot serve both. That is a reason to size the group
    from the frame's height when such a mode lands, not before it:
    every track this package writes today is a point-event track at
    the top two rows.

    The dtypes are deliberately left as Int64/Float64. Narrowing
    ``pos`` to Int32 and ``value`` to UInt16 *doubles* the file on
    disk, because sorted Int64 positions delta-encode far better than
    Int32 ones. A reader that wants them narrow can narrow them in
    memory, where it is free.

    Each contig is taken as a ``slice`` of the frame rather than a
    ``partition_by`` entry. ``partition_by(as_dict=True)`` builds every
    contig at once while the parent is still alive, so it doubles the
    footprint of a table that is already the largest thing in the
    process; a slice is a view over the same buffers and costs nothing.
    That requires the frame to be grouped by contig, which the
    pileup's ``ORDER BY`` guarantees and which `write` checks
    rather than assumes -- an ungrouped frame is sorted once and then
    sliced the same way.

    The ``chrom`` column is kept inside each file although the filename
    already implies it. It costs nothing once dictionary-encoded, and
    it is what keeps a globbed scan of the whole directory filterable
    on ``chrom``.

    The atomic write itself belongs to `stage_directory` and
    `commit_directory`; where the directory is built, what makes the
    move into place atomic, and what each crash point leaves behind are
    stated there.

    ``parameters`` is recorded in the manifest so a table carries the
    pileup arguments that produced it; see `require_complete` for
    what the manifest is for.
    """

    def __init__(self, path, parameters=None):
        self.path = path
        self.rows = 0
        self.parameters = dict(parameters or {})
        self._files = {}
        self._staging = stage_directory(path)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if self._staging is None:
            return False
        if kind is None:
            self.commit()
        else:
            self.abort()
        return False

    def write(self, frame):
        """Stage one frame's contigs.

        The frame must cover contigs no earlier frame covered. That is
        checked rather than assumed, because writing a contig twice
        fails silently: the second file replaces the first, ``rows``
        counts both, and the table still verifies, so training would
        read a contig that is missing half its rows and nothing
        would say so.
        """

        if self._staging is None:
            raise RuntimeError("{} is already closed".format(self.path))

        try:
            runs = _runs(frame, "chrom")
            if len({chrom for chrom, _, _ in runs}) != len(runs):
                # A contig appears in more than one run, so the slices
                # below would write the same filename twice. The pileup
                # never lands here -- its ORDER BY guarantees one run
                # per contig -- but the eager helpers and the tests
                # build frames in whatever order suits them, and pay
                # one sort for it. It sorts a group, not the genome.
                frame = frame.sort("chrom", "pos")
                runs = _runs(frame, "chrom")

            for chrom, offset, length in runs:
                name = _chrom_filename(chrom)
                if name in self._files:
                    raise ValueError(
                        "{} was already written to {}; the frames given "
                        "to one track writer must cover disjoint "
                        "contigs".format(chrom, self.path)
                    )
                part = frame.slice(offset, length)
                if not part.get_column("pos").is_sorted():
                    part = part.sort("pos")
                part.write_parquet(
                    os.path.join(self._staging, name),
                    row_group_size=_ROW_GROUP_SIZE,
                    statistics=True,
                )
                self._files[name] = part.height
                self.rows += part.height
        except BaseException:
            self.abort()
            raise

    def commit(self):
        """Swap the staged directory in and mark it complete."""

        if self._staging is None:
            raise RuntimeError("{} is already closed".format(self.path))
        staging, self._staging = self._staging, None
        commit_directory(
            self.path,
            staging,
            {
                "version": MANIFEST_VERSION,
                "rows": int(self.rows),
                "files": self._files,
                "parameters": self.parameters,
            },
        )

    def abort(self):
        """Discard whatever was staged. A no-op once closed."""

        staging, self._staging = self._staging, None
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


# public
def track_chrom_path(source, chrom):
    """Path to one chromosome's Parquet file inside a track directory.

    Published because it is half of the naming rule a conforming writer
    has to satisfy, and the half easiest to get wrong: the filename is
    ``urllib.parse.quote(chrom, safe="") + ".parquet"``, so a contig
    whose name carries a character the filesystem reserves round-trips.
    Deriving it through this function rather than restating the rule is
    what keeps a third-party table readable here.

    :param source: A track directory, as a `str`, `bytes` or
        `os.PathLike`.
    :param chrom: The contig name, which is quoted rather than assumed
        safe.
    :raises TypeError: if ``source`` is not a path. It reached
        ``os.path.join`` and raised from there -- "Can't mix strings
        and bytes in path components" for a `bytes` path this now
        accepts, and a message naming neither this function nor the
        argument for anything else.
    """

    path = fspath_or_none(source)
    if path is None:
        raise TypeError(
            "track_chrom_path builds a path inside a track directory: "
            "source must be a str, bytes or os.PathLike, got {}.".format(
                type(source).__name__
            )
        )
    return os.path.join(path, _chrom_filename(chrom))


# public
def manifest_path(source):
    """Path to the completeness record that belongs beside ``source``.

    The other half of the naming rule, and published for the same
    reason. Beside the directory rather than inside it, because a track
    directory must hold Parquet and nothing else.

    :param source: A track directory, as a `str`, `bytes` or
        `os.PathLike`.
    :returns: The record's path, as a `str`.
    :raises TypeError: if ``source`` is not a path.

    .. rubric:: Implementation notes

    ``os.fsdecode`` through `quadrat._engine.fspath_or_none` rather
    than `str`. `str` answers for anything, so this returned a path
    built from a repr and no caller could tell: a `bytes` path gave
    ``"b'/data/x.track.parquet'.manifest.json"`` and an integer gave
    ``"42.manifest.json"``. A wrong answer from the function that
    publishes half the naming rule a conforming writer must satisfy is
    worse than a refusal, which is why the refusal is here.
    """

    path = fspath_or_none(source)
    if path is None:
        raise TypeError(
            "manifest_path names the record beside a track: source must "
            "be a str, bytes or os.PathLike, got {}.".format(type(source).__name__)
        )
    return path + MANIFEST_SUFFIX


# public
def read_manifest(source):
    """Completeness record for a track directory, or ``None``.

    ``None`` means the table predates manifests, is a plain file, or the
    record is absent or unreadable -- all of which `require_complete`
    treats the same way, because none of them can be distinguished from a
    write that did not finish.

    The record is a JSON object beside the track directory, at
    `manifest_path`, and this is the published schema: an
    interoperability contract rather than an implementation detail,
    because a writer other than this package must produce it for its
    tables to be readable here.

    :param source: A track directory, as a `str`, `bytes` or
        `os.PathLike`.
    :raises TypeError: if ``source`` is not a path. ``None`` is kept
        for the four causes above, which are all "there is no record
        to read"; a wrong type is a caller error, and answering
        ``None`` to it gave that caller a fifth meaning this list does
        not carry -- one a reader acts on by treating an unverifiable
        table as one that predates manifests.
    :returns: The record as a `dict`, or ``None``. Its fields are the
        published schema:

        ``version``
            Schema version, which must equal `MANIFEST_VERSION`.
            `require_complete` refuses any other value rather than
            guessing, so a writer that cannot reach this constant
            cannot conform.

        ``files``
            Object mapping each per-contig filename to the number of
            rows in it. The filenames are what `track_chrom_path`
            derives, relative to the directory. One entry per contig
            the table holds a row for and none for a contig it does
            not, so an absent contig and an empty one are the same
            thing here as they are on disk. Must be an object: a list
            of filenames is refused.

        ``rows``
            Rows in the whole table. ``sum(files.values())`` must equal
            it -- the one internal consistency the record has, and the
            one check that distinguishes a record written by a single
            uninterrupted write from two merged or hand-edited. A
            record omitting ``rows`` is accepted and not summed, and
            may then list files with ``null`` counts; a record stating
            ``rows`` while any entry declares no count is refused.

        ``parameters``
            Object recording the pileup arguments that produced the
            table, so a table carries its own provenance. Not
            interpreted here and not verified.

    .. rubric:: Implementation notes

    The fields are a definition list under the return value rather than
    a ``:param:`` field list. They are the keys of a record and not the
    arguments of a function, and autodoc renders a ``:param:`` as a
    parameter: this page told a caller that ``read_manifest`` accepts
    ``version``, ``files``, ``rows`` and ``parameters``, which it does
    not, while documenting its one real argument nowhere.
    """

    original, source = source, fspath_or_none(source)
    if source is None:
        raise TypeError(
            "read_manifest reads the record beside a track: source must "
            "be a str, bytes or os.PathLike, got {}.".format(type(original).__name__)
        )
    if not os.path.isdir(source):
        return None
    try:
        with open(manifest_path(source)) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest


# public
def require_complete(source):
    """Raise unless ``source`` is a track that finished writing.

    The check exists because the reader cannot otherwise tell a truncated
    table from a complete one. A reader returns ``None``
    for a chromosome whose file is absent, and that is *correct* for a
    contig with no rows -- routine for small unplaced ones -- so a
    writer killed after chr1-chr8 leaves a directory that reads as valid
    and silently trains on a genome missing most of its signal. The
    manifest, written last, is the only thing that distinguishes the two.

    There is deliberately no override. A table costs minutes to rebuild
    (234 s for twenty BAMs on a 128-core box) and a silently partial one
    costs a training run plus however long it takes to doubt the result,
    so "read it anyway" is never the right trade. Directories written
    before manifests existed fail here and must be rebuilt; they are
    genuinely unverifiable, not merely old.

    In-memory frames and single-file tables pass through untouched -- a
    frame has no partial state, and a truncated Parquet file fails on its
    own footer.
    """

    path = fspath_or_none(source)
    if path is None:
        # Not a path at all. An in-memory frame has no partial state to
        # verify, which is the one documented pass-through of that
        # shape; anything else is a caller error rather than a table,
        # and returning it unchecked would let `require_complete(42)`
        # or `require_complete(None)` read as verified.
        polars = import_on_first_use("polars")
        if isinstance(source, (polars.DataFrame, polars.LazyFrame)):
            return source
        raise TypeError(
            "require_complete verifies a track: a path to one, "
            "or an in-memory polars frame. Got {}.".format(type(source).__name__)
        )

    if os.path.isfile(path):
        # A single-file table has no partial state this can see: a
        # truncated Parquet file fails on its own footer.
        return source

    if not os.path.isdir(path):
        raise ValueError(
            "{} is not a track; there is no such file or "
            "directory. A path that does not exist is exactly the input "
            "this refuses rather than passes through, because a "
            "mistyped or since-deleted table is indistinguishable from "
            "one that never finished.".format(path)
        )

    manifest = read_manifest(path)
    if manifest is None:
        # `read_manifest` collapses absent, unreadable and non-dict to
        # `None` -- defended in its own docstring, because this
        # function treats all three the same way. The *message* must
        # not collapse them: a record that is missing and a record
        # that is present but malformed have different fixes, and
        # telling the second caller their file is not there sends them
        # looking for the wrong thing. Stat'd here rather than
        # returned from `read_manifest`, so that function keeps the
        # single documented return shape.
        marker = manifest_path(path)
        if os.path.exists(marker):
            raise ValueError(
                "{} has a {} beside it, but it cannot be read as a "
                "manifest record -- it is unreadable, not valid JSON, or "
                "its top-level value is not an object. The table cannot "
                "be verified complete without it. Rebuild it by piling "
                "up the alignments again.".format(path, os.path.basename(marker))
            )
        raise ValueError(
            "{} has no {} beside it and cannot be verified complete. A "
            "track directory whose write was interrupted is "
            "indistinguishable from one whose contigs are genuinely empty, "
            "so it is not safe to train on. Rebuild it by piling up the "
            "alignments again.".format(path, os.path.basename(marker))
        )

    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            "{} declares manifest schema version {!r}, but this quadrat "
            "writes and reads version {}. Rebuild the table.".format(
                path, version, MANIFEST_VERSION
            )
        )

    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        # Refused here because otherwise it reaches `.values()` below
        # and raises `AttributeError` from a function documented to
        # raise `ValueError` -- a malformed record from a third-party
        # writer, which the published track format invites, crashing
        # with a type error rather than being told what is wrong with
        # it. A list is the shape to get wrong first, since the keys
        # alone look like the whole of the information.
        raise ValueError(
            "{} has a manifest whose `files` is {}, not an object "
            "mapping each Parquet filename to its row count. The record "
            "does not describe a track this can verify; rebuild the "
            "table.".format(path, type(files).__name__)
        )
    missing = [name for name in files if not os.path.exists(os.path.join(path, name))]
    if missing:
        raise ValueError(
            "{} is missing {} of the {} Parquet files its manifest lists "
            "({}). The table has been truncated since it was written; "
            "rebuild it.".format(
                path, len(missing), len(files), ", ".join(sorted(missing)[:5])
            )
        )

    # Every file is present, which says nothing about whether the manifest
    # describes them correctly. `TrackWriter` increments `rows` and
    # records the per-file height from the same frame slice, so the two
    # agree by construction and a disagreement means the record did not
    # come from one uninterrupted write -- two manifests merged, a hand
    # edit, or a writer that is not this one. Cheap to check and the only
    # internal consistency the record has.
    #
    # A record with no `rows` at all has nothing to add up, and that is
    # the one case skipped: window-cache entries share
    # `MANIFEST_VERSION`, carry no `rows` key and list their files with
    # `None`, so the absent total is what excuses them rather than the
    # absent counts.
    #
    # Excusing it on the counts as well is backwards, and would be the
    # easy reading of the same guard: one unusable value would disable
    # the check for the *whole* manifest, so `{"a": 1, "b": None}`
    # against `"rows": 12345` would pass. A record stating a total it
    # cannot support is precisely a record that did not come from one
    # uninterrupted write, which is the only internal consistency it
    # has.
    #
    # Numberhood rather than `int`, because JSON has one numeric type:
    # a producer that writes `2.0` is writing a row count, and an
    # `isinstance(int)` probe let a float total skip the check.
    rows = manifest.get("rows")
    numbers = (int, float)
    if isinstance(rows, numbers):
        unusable = sorted(
            name for name, count in files.items() if not isinstance(count, numbers)
        )
        if unusable:
            raise ValueError(
                "{} has a manifest recording {} rows in total while {} of "
                "its {} files declare no count ({}). A record cannot state "
                "a total it has no per-file heights to support; rebuild "
                "the table.".format(
                    path, rows, len(unusable), len(files), ", ".join(unusable[:5])
                )
            )
        total = sum(files.values())
        if total != rows:
            raise ValueError(
                "{} has a manifest that does not add up: its {} files "
                "declare {} rows between them but the manifest records "
                "{} in total. The record does not describe a single "
                "finished write; rebuild the table.".format(
                    path, len(files), total, rows
                )
            )

    return source


# public
def track_contigs(source):
    """Every contig a track holds at least one row for.

    The directory layout answers from its filenames: `TrackWriter`
    writes one file per contig run and none at all for a contig with no
    rows, so for the tables training reads this costs a listing and no
    bytes. A single-file table or an in-memory frame has to be asked,
    which reads the ``chrom`` column and nothing else.

    The manifest is preferred over the listing when there is one, on
    the assumption that the caller has already verified the table --
    nothing on this path calls `require_complete`, and a truncated
    table therefore reports whatever its manifest lists rather than
    what is on the disk. That is the right trade for the training path,
    which verifies a table once and then addresses it many times; a
    caller who has not verified it should.

    The counterpart of `bam_contigs` on the other side of the
    pileup. #5's signal join compares its output against this, to
    refuse a window set reaching a contig the track holds no rows for.

    :param source: A ``.track.parquet`` directory, a single Parquet
        file, or an in-memory polars frame -- each as `require_complete`
        accepts them, a path being a `str`, `bytes` or `os.PathLike`.
    :returns: The contig names, as a `set`.
    :raises TypeError: if ``source`` is none of those.

    .. rubric:: Implementation notes

    The three shapes are dispatched here rather than through a shared
    normalizer. There was one -- a module-private helper written for
    #5's signal reader and called from this one place -- and it carried
    the same path-or-frame dispatch `require_complete` performs over
    the same four input shapes, with its own separately-worded
    refusal. Its directory branch was unreachable from here, because
    the branch above has already taken that case, so the half #5 needs
    was dead and the half it does not was live. Its message was the
    other cost: asked which contigs a track holds, a caller was told
    that "the GIQL engine reads signals from .track.parquet tables",
    about an operation they had not invoked.
    """

    path = fspath_or_none(source)
    if path is not None and os.path.isdir(path):
        manifest = read_manifest(path)
        names = list((manifest.get("files") or {}) if manifest else [])
        if not names:
            names = [name for name in os.listdir(path) if name.endswith(".parquet")]
        return {urllib.parse.unquote(name[: -len(".parquet")]) for name in names}

    polars = import_on_first_use("polars")
    if path is not None:
        # A plain file is still read, so tables written before the
        # per-contig layout keep loading.
        frame = polars.scan_parquet(path)
    elif isinstance(source, polars.LazyFrame):
        frame = source
    elif isinstance(source, polars.DataFrame):
        frame = source.lazy()
    else:
        raise TypeError(
            "track_contigs reads a track: a {} directory, a single "
            "Parquet file, or an in-memory polars frame with chrom, pos "
            "and value columns. Got {}.".format(TRACK_SUFFIX, type(source).__name__)
        )
    return set(frame.select("chrom").unique().collect()["chrom"].to_list())


def strand_slices(frame, unstranded):
    """The rows of each output channel, as views over ``frame``.

    A ``filter`` per strand is the obvious way to do this and costs
    three copies: each call allocates half the table, the sort that
    follows allocates half again, and all of it while the parent frame
    is still referenced. Since the SQL orders by strand first, each
    channel is one contiguous run, so ``slice`` returns it without
    copying any of it. A strand with no reads yields an empty frame,
    which is what a filter would have given too.

    :raises ValueError: if a strand appears in more than one run, which
        means the frame is not grouped by strand and a slice is not the
        channel.
    """

    columns = ["chrom", "pos", "value"]
    if unstranded:
        return [frame.select(columns)]

    runs = _runs(frame, "strand")
    spans = {strand: (offset, length) for strand, offset, length in runs}
    if len(spans) != len(runs):
        # The precondition is the same one `TrackWriter.write` checks,
        # and the two resolve a violation in opposite directions: that
        # one sorts the frame and slices it again, this one refuses.
        # The asymmetry is deliberate and criterion #2 is why -- the
        # output must be byte-identical whatever produced it, and
        # sorting by contig reorders nothing a reader sees while
        # sorting by strand reorders the channels themselves. Slicing
        # rather than repairing is also what keeps this free: a
        # `filter` per strand costs three copies of half the table, as
        # the summary above says.
        #
        # It is load-bearing for correctness here rather than only for
        # memory: a dict comprehension over
        # the runs keeps the last run per strand, so every earlier run
        # is dropped from the written table -- with `rows` and the
        # manifest both computed from the truncated slices, so they
        # agree and `require_complete` passes. Checked rather than
        # assumed, at one comparison per batch on a path that nothing
        # in this package reaches: `pileup_sql` emits
        # `ORDER BY g.strand, ...` and the cast preserves row order.
        raise ValueError(
            "a stranded channel was asked for from a frame that is not "
            "grouped by strand: {} runs over {} distinct strand(s). Each "
            "channel is taken as one contiguous slice, so an ungrouped "
            "frame would be silently truncated to its last run per "
            "strand. Sort by strand before splitting.".format(len(runs), len(spans))
        )
    return [
        frame.slice(*spans.get(strand, (0, 0))).select(columns) for strand in ("+", "-")
    ]


def track_paths(name, unstranded):
    """The track directories one pileup of ``name`` writes, in channel order.

    The other half of the naming rule `track_chrom_path` publishes, and
    the half a caller derives rather than reads: an unstranded pileup
    writes ``{name}.track.parquet``, and a stranded one writes
    ``{name}.+.track.parquet`` and ``{name}.-.track.parquet``, the
    channel infix sitting between the name and `TRACK_SUFFIX` so that
    both compose with a name that is itself a path.

    :param name: The output prefix, as a `str`, `bytes` or
        `os.PathLike`.
    :param unstranded: Whether the pileup pooled both strands.
    :raises TypeError: if ``name`` is not a path. It used to reach the
        concatenation below and raise from there, naming neither this
        function nor the argument -- after the alignment tables had
        been registered into the process-global session, which is the
        failure the boundary normalization exists to pre-empt.
    """

    prefix = fspath_or_none(name)
    if prefix is None:
        raise TypeError(
            "track_paths derives the track directories a pileup writes: "
            "name must be a str, bytes or os.PathLike, got {}.".format(
                type(name).__name__
            )
        )
    if unstranded:
        return [prefix + TRACK_SUFFIX]
    return [prefix + ".+" + TRACK_SUFFIX, prefix + ".-" + TRACK_SUFFIX]


def stage_directory(path):
    """Create a fresh staging sibling of ``path`` for one atomic write to fill.

    The first half of the protocol `commit_directory` completes. The
    staging directory is a sibling of ``path``.

    Sweeps the siblings an interrupted write left behind, before the new
    one exists.

    :param path: The path the staged directory will become, as a `str`.
    :returns: The staging directory, created and empty.

    .. rubric:: Implementation notes

    The two halves are separate calls rather than one because a writer
    does arbitrary work in between, and what fills the directory
    differs from one writer to the next. What they share is where the
    directory is built, and that is the half worth having once.

    A sibling of ``path`` rather than a directory under ``tempfile``,
    because ``os.replace`` is atomic only within one filesystem and a
    sibling is the single placement that guarantees that by
    construction. It is also what keeps the scheme sound on NFS, where
    directory rename is atomic in practice but not by contract.

    The sweep runs before the new directory exists, and so exactly
    once, because it reaps every ``{path}.staging.*``, which would
    include this one if it ran after.
    """

    _sweep_stale(path)
    staging = "{}{}{}".format(path, _STAGING_INFIX, uuid.uuid4().hex)
    os.makedirs(staging)
    return staging


def commit_directory(path, staging, manifest):
    """Swap ``staging`` in as ``path`` and record it as complete.

    The second half of `stage_directory`'s protocol, and the half that
    decides what a crash leaves behind.

    The order is: remove the record, replace the directory, write the
    record. So the sequence a reader can observe is unverifiable,
    replaced, verifiable, and every crash point leaves something that
    fails `require_complete` rather than a new directory wearing the
    previous write's record. Unverifiable is the one direction this is
    allowed to fail in, because the caller can always build again and
    cannot detect a directory that lies. The cost is that a successful
    write is briefly complete-but-unverifiable, and a record is a few
    hundred bytes, so that window is one small write long.

    A swap that fails puts the previous directory back, because a failed
    rebuild should not also destroy what was already there. It comes
    back unverifiable -- its record is already gone, and this cannot
    reconstruct one it did not write -- which is the same answer every
    other crash point gives.

    :param path: The final path, as a `str`. A directory already there
        is moved aside and removed once the swap succeeds; a plain file
        is removed outright.
    :param staging: The directory to swap in, from `stage_directory`.
        It is removed if the swap fails.
    :param manifest: The completeness record, as a `dict`. Written to
        `manifest_path` of ``path`` after the swap, through a ``.tmp``
        sibling and one more ``os.replace``, because a reader must
        never be able to parse a half-flushed record.
    :raises OSError: if the swap fails, with the previous directory back
        in place; or if writing the record fails after the swap, with
        the new directory in place and no record beside it, so that it
        fails `require_complete`; any previous directory, and the
        record's temporary file if one was created, are then left as
        siblings for the next `stage_directory` of ``path`` to sweep.
    """

    marker = manifest_path(path)
    try:
        os.remove(marker)
    except FileNotFoundError:
        pass

    trash = None
    if os.path.isdir(path):
        trash = "{}{}{}".format(path, _TRASH_INFIX, uuid.uuid4().hex)
        os.replace(path, trash)
    elif os.path.exists(path):
        os.remove(path)

    try:
        os.replace(staging, path)
    except BaseException:
        if trash is not None and not os.path.exists(path):
            os.replace(trash, path)
            trash = None
        shutil.rmtree(staging, ignore_errors=True)
        raise

    with open(marker + _MARKER_TMP_SUFFIX, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    os.replace(marker + _MARKER_TMP_SUFFIX, marker)

    if trash is not None:
        shutil.rmtree(trash, ignore_errors=True)


def _chrom_filename(chrom):
    """Filesystem-safe filename for one contig.

    ``urllib.parse.quote`` leaves alphanumerics and ``_.-~`` alone, so the
    ordinary names stay readable (``chr1.parquet``,
    ``chr14_GL000009v2_random.parquet``) while an assembly that ships a
    contig with a slash or a space in it still lands on a valid path.
    """

    return urllib.parse.quote(str(chrom), safe="") + ".parquet"


def _sweep_stale(path):
    """Remove staging/discard siblings left by an interrupted write.

    A writer killed between ``makedirs`` and the final rename leaves a
    ``{path}.staging.{hex}`` directory behind; one killed between the two
    renames leaves a ``{path}.trash.{hex}``. Neither is ever read, but
    they are the size of a whole channel, so sweeping them at the start of
    the next write keeps a crash loop from filling the disk. Failures are
    swallowed: a leftover directory someone else is actively writing (two
    pileups racing on the same name) is their problem to report, not a
    reason to fail this one before it starts.

    The manifest's temporary file goes with them. It is small, and for a
    track it is one stray file beside a directory of channels -- but a
    writer that swallows its own failures leaves one behind with nothing
    to report it, so unless this sweep takes it nothing ever reaps it.
    """

    # `glob.escape` because a track name is a caller's path and may
    # carry `[`, `*` or `?`; without it a name containing one would
    # match the wrong siblings or none.
    quoted = glob.escape(os.path.abspath(path))
    for infix in (_STAGING_INFIX, _TRASH_INFIX):
        for entry in glob.iglob(quoted + glob.escape(infix) + "*"):
            try:
                shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass

    # By name rather than by glob, the marker's temporary file being one
    # fixed name. Swallowed for the same reason as the directories.
    try:
        os.remove(os.path.abspath(path) + MANIFEST_SUFFIX + _MARKER_TMP_SUFFIX)
    except OSError:
        pass


def _runs(frame, column):
    """Contiguous ``(value, offset, length)`` runs of one column.

    A frame grouped by that column -- which is what the pileup's
    ``ORDER BY`` delivers -- has exactly one run per distinct value, and a
    run is exactly the rows one output file needs. Taking it with
    ``slice`` is a view over the same Arrow buffers; ``filter`` and
    ``partition_by`` both allocate a second copy of everything they are
    handed, which is the difference between holding the result once and
    holding it three times.
    """

    runs = []
    offset = 0
    for entry in frame.get_column(column).rle():
        length = int(entry["len"])
        runs.append((entry["value"], offset, length))
        offset += length
    return runs
