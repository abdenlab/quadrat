"""The cut-site table format, and the atomic writer that produces it.

A cut-site table is a directory of per-contig Parquet files holding
sorted ``(chrom, pos, value)`` rows, with a completeness manifest
beside it rather than inside it. Beside, because a cut-site directory
must hold Parquet and nothing else: polars refuses to scan a directory
of mixed extensions, and so does anything else pointed at the dataset.

The format and ``MANIFEST_VERSION`` are deliberately unchanged from
cherimoya's. Tables either package writes are readable by the other,
and the alternative is re-piling every corpus to move between them.
"""

import json
import os
import shutil
import urllib.parse
import uuid

# public
CUT_SITE_SUFFIX = ".cuts.parquet"


# Parquet row-group size for the cut-site tables. See `_write_cut_sites`
# for why this number and not a rounder one.
_ROW_GROUP_SIZE = 50_000


# Suffix of the completeness record that sits *beside* a cut-site
# directory, and the schema version it declares. See `_write_cut_sites`
# for what it is for.
#
# Beside, not inside: a cut-site directory must hold Parquet and nothing
# else, because polars refuses to scan a directory of mixed extensions
# and so does anything else pointed at the dataset. That invariant is
# pinned by `test_cut_site_directory_holds_parquet_and_nothing_else`,
# which an earlier memory-mapped read index already broke once.
MANIFEST_SUFFIX = ".manifest.json"


MANIFEST_VERSION = 1


# Prefixes for the staging and discard directories the atomic write uses.
# Both are siblings of the final path so the renames stay within one
# filesystem; `_sweep_stale` reaps any left behind by a killed writer.
_STAGING_INFIX = ".staging."


_TRASH_INFIX = ".trash."


def _chrom_filename(chrom):
    """Filesystem-safe filename for one contig.

    ``urllib.parse.quote`` leaves alphanumerics and ``_.-~`` alone, so the
    ordinary names stay readable (``chr1.parquet``,
    ``chr14_GL000009v2_random.parquet``) while an assembly that ships a
    contig with a slash or a space in it still lands on a valid path.
    """

    return urllib.parse.quote(str(chrom), safe="") + ".parquet"


def cut_site_chrom_path(source, chrom):
    """Path to one chromosome's Parquet file inside a cut-site directory."""

    return os.path.join(source, _chrom_filename(chrom))


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
    """

    parent = os.path.dirname(os.path.abspath(path)) or "."
    prefix = os.path.basename(os.path.abspath(path))
    try:
        entries = os.listdir(parent)
    except OSError:
        return
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        rest = entry[len(prefix) :]
        if not (rest.startswith(_STAGING_INFIX) or rest.startswith(_TRASH_INFIX)):
            continue
        try:
            shutil.rmtree(os.path.join(parent, entry), ignore_errors=True)
        except OSError:
            pass


def manifest_path(source):
    """Path to the completeness record that belongs beside ``source``."""

    return str(source) + MANIFEST_SUFFIX


# public
def read_manifest(source):
    """Completeness record for a cut-site directory, or ``None``.

    ``None`` means the table predates manifests, is a plain file, or the
    record is absent or unreadable -- all of which :func:`require_complete`
    treats the same way, because none of them can be distinguished from a
    write that did not finish.
    """

    if not isinstance(source, str) or not os.path.isdir(source):
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
    """Raise unless ``source`` is a cut-site table that finished writing.

    The check exists because the reader cannot otherwise tell a truncated
    table from a complete one. :func:`_chrom_cut_sites` returns ``None``
    for a chromosome whose file is absent, and that is *correct* for a
    contig with no cut sites -- routine for small unplaced ones -- so a
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

    if not isinstance(source, str) or not os.path.isdir(source):
        return source

    manifest = read_manifest(source)
    if manifest is None:
        raise ValueError(
            "{} has no {} beside it and cannot be verified complete. A "
            "cut-site directory whose write was interrupted is "
            "indistinguishable from one whose contigs are genuinely empty, "
            "so it is not safe to train on. Rebuild it by piling up the "
            "alignments again.".format(source, os.path.basename(manifest_path(source)))
        )

    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            "{} declares manifest schema version {!r}, but this quadrat "
            "writes and reads version {}. Rebuild the table.".format(
                source, version, MANIFEST_VERSION
            )
        )

    files = manifest.get("files") or {}
    missing = [name for name in files if not os.path.exists(os.path.join(source, name))]
    if missing:
        raise ValueError(
            "{} is missing {} of the {} Parquet files its manifest lists "
            "({}). The table has been truncated since it was written; "
            "rebuild it.".format(
                source, len(missing), len(files), ", ".join(sorted(missing)[:5])
            )
        )

    # Every file is present, which says nothing about whether the manifest
    # describes them correctly. `_CutSiteWriter` increments `rows` and
    # records the per-file height from the same frame slice, so the two
    # agree by construction and a disagreement means the record did not
    # come from one uninterrupted write -- two manifests merged, a hand
    # edit, or a writer that is not this one. Cheap to check and the only
    # internal consistency the record has.
    #
    # Guarded on the values being counts because window-cache entries
    # share `MANIFEST_VERSION` and list their files with `None`, having no
    # row count to record; for those there is nothing to add up.
    rows = manifest.get("rows")
    counts = list(files.values())
    if isinstance(rows, int) and all(isinstance(n, int) for n in counts):
        total = sum(counts)
        if total != rows:
            raise ValueError(
                "{} has a manifest that does not add up: its {} files "
                "declare {} rows between them but the manifest records "
                "{} in total. The record does not describe a single "
                "finished write; rebuild the table.".format(
                    source, len(files), total, rows
                )
            )

    return source


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


class _CutSiteWriter(object):
    """One cut-site table, assembled from one or more frames.

    The pileup runs the genome in contig groups so that it holds only
    part of the aggregate at a time, and so hands a table over in
    several disjoint pieces rather than one. The staging directory, the
    file list and the commit belong to a *writer*, not to a path: two
    separate writers on the same path still replace one another
    wholesale, which is what lets :func:`_write_cut_sites` keep meaning
    exactly what it meant before groups existed.

    No frame is retained. Every part written is a zero-copy ``slice`` of
    the caller's frame, so holding one would pin the whole group and
    give back the memory the grouping exists to save; only a filename
    and a row count are kept.

    Use it as a context manager. A writer abandoned without one leaks
    its staging directory until the next `_sweep_stale`, which is the
    same exposure a killed process has always had.
    """

    def __init__(self, path, parameters=None):
        # Sweep before the staging directory exists, and exactly once:
        # `_sweep_stale` reaps every `{path}.staging.*` sibling, which
        # would include ours if it ran afterwards.
        _sweep_stale(path)
        self.path = path
        self.rows = 0
        self.parameters = dict(parameters or {})
        self._marker = manifest_path(path)
        self._files = {}
        self._staging = "{}{}{}".format(path, _STAGING_INFIX, uuid.uuid4().hex)
        os.makedirs(self._staging)

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
        read a contig that is missing half its cut sites and nothing
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
                        "to one cut-site writer must cover disjoint "
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

        # Invalidate the old table before touching it, so a crash
        # anywhere below leaves something that refuses to be read
        # rather than a new directory wearing a previous write's
        # manifest.
        try:
            os.remove(self._marker)
        except FileNotFoundError:
            pass

        trash = None
        if os.path.isdir(self.path):
            trash = "{}{}{}".format(self.path, _TRASH_INFIX, uuid.uuid4().hex)
            os.replace(self.path, trash)
        elif os.path.exists(self.path):
            os.remove(self.path)

        try:
            os.replace(staging, self.path)
        except BaseException:
            # Put the previous table back; a failed rebuild should not
            # also destroy what was already there. It stays
            # unverifiable, because its manifest is already gone and we
            # cannot reconstruct one we did not write.
            if trash is not None and not os.path.exists(self.path):
                os.replace(trash, self.path)
                trash = None
            shutil.rmtree(staging, ignore_errors=True)
            raise

        manifest = {
            "version": MANIFEST_VERSION,
            "rows": int(self.rows),
            "files": self._files,
            "parameters": self.parameters,
        }
        # The manifest is the completeness marker, so it must land
        # whole: a reader must never be able to parse a half-flushed
        # JSON object.
        with open(self._marker + ".tmp", "w") as handle:
            json.dump(manifest, handle, indent=1, sort_keys=True)
        os.replace(self._marker + ".tmp", self._marker)

        if trash is not None:
            shutil.rmtree(trash, ignore_errors=True)

    def abort(self):
        """Discard whatever was staged. A no-op once closed."""

        staging, self._staging = self._staging, None
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _write_cut_sites(path, frame, parameters=None):
    """Write one channel as a directory of per-chromosome Parquet files.

    ``path`` keeps its ``.cuts.parquet`` name but is a *directory*, so every
    caller that builds these names from `CUT_SITE_SUFFIX` --
    `_cut_site_paths` and anything downstream that reads the emitted
    manifest -- is unaffected.

    Sorted per chromosome with 50,000-row groups, which is what lets the
    whole table be scanned and grouped into a
    :class:`WindowSignalCache` in well under a second. Both choices were
    measured on ENCSR483RKN (26.2M cut sites): against the previous single
    file with 1,000,000-row groups, a windowed read went from 6.7 ms to
    0.78 ms, and smaller groups are not better -- 10,000 regresses to
    2.2 ms as per-group metadata outweighs what it prunes.

    Training no longer reads a window out of here directly, so that figure
    now bounds the one-off grouping pass rather than every sample. It still
    matters for the eager extractor and the outlier aggregate, which do
    scan these files.

    The dtypes are deliberately left as Int64/Float64. Narrowing ``pos`` to
    Int32 and ``value`` to UInt16 *doubles* the file on disk, because
    sorted Int64 positions delta-encode far better than Int32 ones; the
    cache narrows them in memory instead, where it is free.

    Each contig is taken as a ``slice`` of ``frame`` rather than a
    ``partition_by`` entry. ``partition_by(as_dict=True)`` builds every
    contig at once while the parent is still alive, so it doubles the
    footprint of a table that is already the largest thing in the
    process; a slice is a view over the same buffers and costs nothing.
    This requires the frame to be grouped by contig, which the pileup's
    ``ORDER BY`` guarantees and which is checked rather than assumed --
    an ungrouped frame is sorted once and then sliced the same way.

    The ``chrom`` column is kept inside each file even though the filename
    already implies it. It costs nothing once dictionary-encoded and it keeps
    the eager `extract_signal` path, which filters on ``chrom``, working
    against a globbed scan of the directory.

    The directory is assembled under a ``.staging.{hex}`` sibling and moved
    into place with ``os.replace``, which is atomic within a filesystem --
    the staging sibling guarantees that by construction, and it is also
    what keeps the scheme sound on NFS, where directory rename is atomic
    in practice but not by contract.

    The manifest is written *after* that swap, and the stale one is
    removed *before* it, so the commit order is: table unverifiable,
    table replaced, table verifiable. Every crash point therefore leaves
    the table failing :func:`require_complete` rather than silently
    readable, which is the direction this is allowed to fail in. The cost
    is that a successful write is briefly complete-but-unverifiable; the
    manifest is a few hundred bytes, so that window is a single small
    write long.

    ``parameters`` is recorded in the manifest so a table carries the
    pileup arguments that produced it; see :func:`require_complete` for
    what the manifest is for.

    This is the whole-table form. The pileup runs the genome in contig
    groups and so drives :class:`_CutSiteWriter` directly, giving it one
    frame per group and committing once at the end; the two produce
    byte-identical tables.
    """

    with _CutSiteWriter(path, parameters=parameters) as writer:
        writer.write(frame)


def _strand_slices(frame, unstranded):
    """The rows of each output channel, as views over ``frame``.

    Stranded output used to come from two ``filter`` calls, each
    allocating half the table, and then a sort allocating half again --
    all while the parent frame was still referenced. Since the SQL orders
    by strand first, each channel is one contiguous run, and ``slice``
    returns it without copying. A strand with no reads yields an empty
    frame, as the filter did.
    """

    columns = ["chrom", "pos", "value"]
    if unstranded:
        return [frame.select(columns)]

    spans = {
        strand: (offset, length) for strand, offset, length in _runs(frame, "strand")
    }
    return [
        frame.slice(*spans.get(strand, (0, 0))).select(columns) for strand in ("+", "-")
    ]


def _cut_site_paths(name, unstranded):
    if unstranded:
        return [name + CUT_SITE_SUFFIX]
    return [name + ".+" + CUT_SITE_SUFFIX, name + ".-" + CUT_SITE_SUFFIX]


###
# Loci
###


def _cut_site_frame(source):
    """Normalize one cut-site source to a polars LazyFrame.

    A source is either a path to a ``.cuts.parquet`` table written by
    :func:`pileup_alignments`, or an in-memory polars ``DataFrame`` /
    ``LazyFrame`` with the same ``chrom, pos, value`` columns. The in-memory
    form is what lets a caller run BAM to tensors without touching disk.

    ``.cuts.parquet`` is a directory of per-chromosome files (see
    :func:`_write_cut_sites`); it is globbed back into one frame here because
    this eager path filters on ``chrom`` and wants every contig visible at
    once. A plain file is still accepted so tables written before the layout
    change keep loading.
    """

    import polars

    if isinstance(source, str):
        if os.path.isdir(source):
            require_complete(source)
            return polars.scan_parquet(os.path.join(source, "*.parquet"))
        return polars.scan_parquet(source)
    if isinstance(source, polars.LazyFrame):
        return source
    if isinstance(source, polars.DataFrame):
        return source.lazy()

    raise NotImplementedError(
        "The GIQL engine reads signals from {} tables "
        "or in-memory polars frames; got {!r}".format(
            CUT_SITE_SUFFIX, type(source).__name__
        )
    )


def cut_site_contigs(source):
    """Every contig a cut-site source holds at least one row for.

    The directory layout answers from its filenames: `_write_cut_sites`
    writes one file per contig run and none at all for a contig with no
    rows, so for the tables training reads this costs a listing and no
    bytes. The manifest is preferred over the listing when there is
    one, because :func:`require_complete` has already checked those
    names against the disk. A single-file table or an in-memory frame
    has to be asked, which reads the ``chrom`` column and nothing else.

    The counterpart of :func:`bam_contigs` on the other side of the
    pileup, and what :func:`_require_signal_contigs` compares against.
    """

    if isinstance(source, str) and os.path.isdir(source):
        manifest = read_manifest(source)
        names = list((manifest.get("files") or {}) if manifest else [])
        if not names:
            names = [name for name in os.listdir(source) if name.endswith(".parquet")]
        return {urllib.parse.unquote(name[: -len(".parquet")]) for name in names}

    frame = _cut_site_frame(source)
    return set(frame.select("chrom").unique().collect()["chrom"].to_list())


def _source_name(source):
    """How to name a cut-site source in a message."""

    return source if isinstance(source, str) else "an in-memory frame"
