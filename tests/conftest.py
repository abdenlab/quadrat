"""Suite-wide setup and the one fixture every data-path module wants.

The environment block runs at import time, before anything imports
`quadrat`, because the package resolves its cache directory from the
environment on every call rather than once at import, so a later
assignment would leave the first reader pointed at the real user cache.

The cache directory has two tiers and the import-time one is what this
block owns: a session directory that is in place before the first
import and is what the two subprocess tests observe, since neither
starts a pytest session. During a test body `_per_test_cache_dir`
supersedes it, and that fixture's docstring states what it guarantees.

.. rubric:: Implementation notes

Two reasons the session directory is not the user's own, and the
second is the one that bites: writing to the user's cache pollutes a
real install, and an entry surviving from an earlier run would satisfy
a test written to exercise a cold build, which then passes while
testing nothing.

``QUADRAT_CACHE_DIR`` is therefore assigned rather than defaulted,
since defaulting would honour an exported value and an exported value
is exactly the surviving entry a fresh directory exists to rule out.
``TQDM_DISABLE`` is the other way round: a developer who turns progress
bars back on to watch a hang changes nothing a test can observe, so
that one defers to the environment.
"""

import atexit
import os
import shutil
import tempfile

import pytest

from tests import builders

# Progress bars in a captured suite are noise that also slows it down.
os.environ.setdefault("TQDM_DISABLE", "1")

# Fresh per session and never the user's own; see the module docstring.
# The variable's name is the whole contract until `quadrat` grows the
# function that reads it, so a port that renames it breaks this guard
# silently and must keep the name or change both.
_CACHE_DIR = tempfile.mkdtemp(prefix="quadrat-test-cache-")
os.environ["QUADRAT_CACHE_DIR"] = _CACHE_DIR

# Removed at exit rather than on a fixture's teardown, because the
# subprocess tests re-run this module in a fresh interpreter and never
# start a session for a fixture to hang off.
atexit.register(shutil.rmtree, _CACHE_DIR, True)


class Corpus:
    """Paths to one synthetic genome and whatever was built over it.

    ``fasta``, ``peaks``, ``negatives`` and ``exclusion`` stay ``None``
    until something builds them, where ``None`` means nothing built
    that rather than built and empty. ``bams`` and ``tracks`` are empty
    lists for the same reason.

    ``in_window``, ``out_window`` and ``max_jitter`` are recorded
    rather than used: nothing here renders a window, but a test that
    reads one needs the geometry its tracks were built to agree with
    the geometry it asks for, and carrying both on the corpus is what
    keeps them from being spelled twice.

    :param root: The directory holding this corpus's files, as a
        `pathlib.Path`. `Corpus.path` returns members of it as strings,
        which is the form the builders take.
    :param chrom_sizes: The geometry the corpus was built to, copied
        rather than aliased, so a caller mutating its own mapping
        afterwards cannot alter what this corpus reports.
    :param in_window: The input window span, or ``None`` where the
        caller asked for no geometry.
    :param out_window: The output window span, or ``None``.
    :param max_jitter: The largest offset a draw may be shifted by.

    .. rubric:: Implementation notes

    A plain attribute bag rather than a dict, so a typo is an
    `AttributeError` at the point of use rather than a key that is
    silently missing.
    """

    __slots__ = (
        "root",
        "chrom_sizes",
        "fasta",
        "bams",
        "tracks",
        "peaks",
        "negatives",
        "exclusion",
        "in_window",
        "out_window",
        "max_jitter",
    )

    def __init__(
        self, root, chrom_sizes, *, in_window=None, out_window=None, max_jitter=0
    ):
        self.root = root
        self.chrom_sizes = dict(chrom_sizes)
        self.fasta = None
        self.bams = []
        self.tracks = []
        self.peaks = None
        self.negatives = None
        self.exclusion = None
        self.in_window = in_window
        self.out_window = out_window
        self.max_jitter = max_jitter

    def path(self, name):
        """Return a path inside this corpus's directory, as a string."""

        return str(self.root / name)


@pytest.fixture
def machine(mocker, tmp_path):
    """Pin the cores and the memory the planner reads, for one test.

    The calls this patches are the planner's only window onto the
    machine, and they are OS boundaries, which is the one thing §11 of
    the test guide permits mocking. The alternative the ported tests
    used was to patch the private `n_cpus` and `memory_budget`, which
    made the helper the subject rather than the behaviour.

    `plan.py` does `import os`, so `plan.os` is this module and
    patching it here is what the planner sees. ``affinity=False`` makes
    `os.sched_getaffinity` raise `AttributeError`, which is how a
    platform without it presents -- `create=True` is needed the rest of
    the time because Darwin has no such attribute to patch.

    The allocation sources are pointed into ``tmp_path`` and the SLURM
    variables cleared on every call, whether or not a test asks about
    them. Without that, a pinned machine would answer differently
    inside a container than on a developer's laptop -- which is the
    whole point of the planner reading them, and exactly what a test
    that pins a machine must not inherit.

    :param cores: Cores `sched_getaffinity` and `cpu_count` report.
    :param ram_bytes: Host physical memory, or ``None`` to make
        `sysconf` raise as an unreadable `SC_PHYS_PAGES` does.
    :param affinity: ``False`` to present a platform with no
        `sched_getaffinity`.
    :param cgroup: Contents of the cgroup v2 limit file, written only
        when given -- an integer of bytes, or the literal ``"max"``.
    :param cgroup_v1: Contents of the cgroup v1 limit file, same.
    :param slurm: Mapping of SLURM variable to value, set after the
        clearing.
    """

    def pin(
        cores,
        ram_bytes,
        affinity=True,
        cgroup=None,
        cgroup_v1=None,
        slurm=None,
    ):
        # Patched by dotted name rather than through an imported
        # module object, which is what keeps the planner out of this
        # module's import. The environment block above has to run
        # before anything imports `quadrat` -- which is what
        # `test_quadrat_cache_dir_should_be_set_before_the_package_is_imported`
        # pins -- and a top-level import of the planner would import
        # the package at conftest import time and break it. A
        # function-local import would defer it too, and the constraint
        # is an initialisation order rather than a circular dependency,
        # which is not what that exception is for; `mocker.patch`
        # resolves its target when the patch is applied, which is
        # inside the test, so the deferral costs no import statement at
        # all.
        for attribute, name, contents in (
            ("_CGROUP_V2_LIMIT", "memory.max", cgroup),
            ("_CGROUP_V1_LIMIT", "memory.limit_in_bytes", cgroup_v1),
        ):
            target = tmp_path / name
            if contents is not None:
                target.write_text("{}\n".format(contents))
            mocker.patch("quadrat.pileup.plan.{}".format(attribute), str(target))
        mocker.patch.dict(os.environ)
        for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
            os.environ.pop(name, None)
        for name, value in (slurm or {}).items():
            os.environ[name] = str(value)
        if ram_bytes is None:
            # What an unreadable `SC_PHYS_PAGES` looks like: the planner
            # catches OSError and falls back to a guess, which is the
            # one branch of it the suite never otherwise reaches.
            mocker.patch.object(os, "sysconf", side_effect=OSError)
        else:
            mocker.patch.object(
                os,
                "sysconf",
                lambda name: {
                    "SC_PAGE_SIZE": 4096,
                    "SC_PHYS_PAGES": ram_bytes // 4096,
                }[name],
            )
        if affinity:
            mocker.patch.object(
                os, "sched_getaffinity", lambda pid: set(range(cores)), create=True
            )
        else:
            mocker.patch.object(
                os, "sched_getaffinity", side_effect=AttributeError, create=True
            )
        mocker.patch.object(os, "cpu_count", lambda: cores)

    return pin


@pytest.fixture
def sized(mocker):
    """Pin what `os.path.getsize` reports, for one test.

    The other OS boundary the planner reads, patched the way `machine`
    patches the first one. Nine tests in ``tests/test_run.py`` had
    restated the same arrangement -- hold the real function, patch it
    with a lambda answering a stub for the paths of interest and
    delegating for the rest -- and two of the nine were byte-identical,
    so a change to the pattern had to be made in nine places.

    The real function is held before the patch and called for every
    path the argument does not cover, because a corpus's own FASTA and
    index are `stat`ed by the same code path as the alignments and a
    blanket stub would size those too.

    :param sizes: An integer, which every ``.bam`` path reports; or a
        mapping of path to size, which only those paths report. Keys
        are compared as strings, so a `pathlib.Path` and an equal
        ``str`` are the same key.
    """

    real = os.path.getsize

    def pin(sizes):
        if isinstance(sizes, int):

            def getsize(path):
                return sizes if str(path).endswith(".bam") else real(path)

        else:
            stubbed = {str(path): size for path, size in sizes.items()}

            def getsize(path):
                # `in` rather than a falsy-default `get`, so a stub of
                # zero bytes reports zero rather than falling through.
                key = str(path)
                return stubbed[key] if key in stubbed else real(key)

        mocker.patch.object(os.path, "getsize", getsize)

    return pin


@pytest.fixture(scope="session")
def pysam():
    """Return the `pysam` module, skipping the requesting test without it.

    Requested per test rather than imported at a test module's top, so
    a test module that needs no alignment still runs without `pysam`.
    """

    return pytest.importorskip("pysam")


@pytest.fixture(scope="session")
def corpus_factory(tmp_path_factory):
    """Build a synthetic corpus to a given geometry.

    Each call gets its own directory, so two corpora never share a file
    even when their chromosome sizes agree, which is what makes one
    session-scoped factory safe for the whole suite to share.

    Window geometry is recorded, not rendered. Nothing here builds a
    window, but a corpus that carries tracks has to agree with the
    geometry the test reading them asks for, so the factory takes the
    geometry, validates that it can be placed, and hands it back on the
    corpus rather than leaving each module to spell it twice.

    :param name: Prefix for the corpus's own temporary directory.
    :param chrom_sizes: Contig name to length, rendered into every
        artifact the call asks for.
    :param n_bams: How many alignment files to write; ``0`` writes none
        and leaves ``bams`` empty.
    :param n_reads: Reads per alignment file.
    :param seed: Base seed. The sequence is rendered with it and sample
        ``n`` is written with ``seed + n``, so two samples of one corpus
        differ.
    :param index_fasta: Reaches `builders.write_fasta` as ``index``.
    :param soft_mask: Reaches `builders.write_fasta` unchanged, which
        owns the interval semantics and the clipping guarantee.
    :param n_runs: Reaches `builders.write_fasta` unchanged.
    :param reads: ``(name, chrom, pos, cigar, flag)`` tuples placing
        every record explicitly, in place of the seeded random draw.
        Each reaches `builders.aligned_read` as positional arguments,
        and every alignment file gets the same records.
    :param header_contigs: Contigs to declare in the alignment header
        in place of ``chrom_sizes``, which may name one the reference
        does not have. It must cover every contig in ``chrom_sizes`` at
        no less than its length, or the call raises `ValueError`.
    :param peaks: Rows for a peak BED; empty leaves ``peaks`` unset.
    :param negatives: Rows for a background BED; empty leaves
        ``negatives`` unset.
    :param exclusion: Rows for an exclusion BED; empty leaves
        ``exclusion`` unset. The path this factory gives it ends
        ``.bed.gz``.
    :param tracks: One row list per channel, each reaching
        `builders.write_track`; empty leaves ``tracks`` empty. Channel
        ``n`` is written to ``tn.track.parquet``, and the order is the
        channel axis every reader of the set takes.
    :param in_window: The input window span, recorded on the corpus. A
        span that cannot be placed -- ``in_window`` plus jitter on both
        sides exceeding the shortest contig -- raises `ValueError`,
        because every window a test draws from such a corpus would be
        filtered out and the test would pass over nothing.
    :param out_window: The output window span, recorded on the corpus.
        It must not exceed ``in_window``; the output window sits inside
        the input one, and a swapped pair is otherwise only visible as
        a wrong-shaped tensor much later.
    :param max_jitter: The largest offset a draw may be shifted by,
        recorded on the corpus and counted on both sides of the
        placement check above.

    Requesting an alignment or a FASTA index needs ``pysam``, so a call
    that asks for either skips without it. The rest of the surface has
    no such requirement.
    """

    def build(
        name,
        chrom_sizes,
        *,
        n_bams=0,
        n_reads=40,
        seed=0,
        index_fasta=False,
        soft_mask=(),
        n_runs=(),
        header_contigs=None,
        reads=(),
        peaks=(),
        negatives=(),
        exclusion=(),
        tracks=(),
        in_window=None,
        out_window=None,
        max_jitter=0,
    ):
        if n_bams or index_fasta:
            pytest.importorskip("pysam")
        if tracks:
            pytest.importorskip("polars")
        if out_window is not None and in_window is not None and out_window > in_window:
            raise ValueError(
                "out_window {} exceeds in_window {}: the output window "
                "sits inside the input one".format(out_window, in_window)
            )
        if in_window is not None:
            span = in_window + 2 * max_jitter
            shortest = min(chrom_sizes, key=chrom_sizes.get)
            if span > chrom_sizes[shortest]:
                raise ValueError(
                    "a {} bp window with {} bp of jitter needs {} bp and {} "
                    "is {} bp long, so no window could be placed on "
                    "it".format(
                        in_window,
                        max_jitter,
                        span,
                        shortest,
                        chrom_sizes[shortest],
                    )
                )
        if header_contigs is not None:
            uncovered = [
                chrom
                for chrom, size in chrom_sizes.items()
                if header_contigs.get(chrom, -1) < size
            ]
            if uncovered:
                raise ValueError(
                    "header contigs do not cover the reference at {}: {}".format(
                        "its declared length", ", ".join(uncovered)
                    )
                )

        root = tmp_path_factory.mktemp(name)
        corpus = Corpus(
            root,
            chrom_sizes,
            in_window=in_window,
            out_window=out_window,
            max_jitter=max_jitter,
        )
        corpus.fasta = builders.write_fasta(
            corpus.path("genome.fa"),
            chrom_sizes,
            seed=seed,
            soft_mask=soft_mask,
            n_runs=n_runs,
            index=index_fasta,
        )

        if n_bams:
            header = builders.alignment_header(header_contigs or chrom_sizes)
            if reads:
                # Placed where the caller said rather than at a seeded
                # draw, which is what a test asserting on one position
                # needs. Every alignment gets the same records, since
                # the geometry is the point and the file count is not.
                records = [builders.aligned_read(header, *record) for record in reads]
                corpus.bams = [
                    builders.write_bam(
                        corpus.path("s{}.bam".format(index)), records, header
                    )
                    for index in range(n_bams)
                ]
            else:
                corpus.bams = [
                    builders.write_random_bam(
                        corpus.path("s{}.bam".format(index)),
                        header,
                        chrom_sizes,
                        n_reads=n_reads,
                        seed=seed + index,
                    )
                    for index in range(n_bams)
                ]

        if peaks:
            corpus.peaks = builders.write_bed(corpus.path("peaks.bed"), peaks)
        if negatives:
            corpus.negatives = builders.write_bed(
                corpus.path("negatives.bed"), negatives
            )
        if exclusion:
            corpus.exclusion = builders.write_bed(
                corpus.path("exclusion.bed.gz"), exclusion
            )
        corpus.tracks = [
            builders.write_track(corpus.path("t{}.track.parquet".format(channel)), rows)
            for channel, rows in enumerate(tracks)
        ]
        return corpus

    return build


@pytest.fixture(autouse=True)
def _per_test_cache_dir(tmp_path, monkeypatch):
    """Point the cache at a directory this test alone can see.

    Autouse, so this is the value in force for every test body and the
    session directory the module docstring describes is only ever what
    an import observes. One test cannot reach another's cache entries,
    which is the cold-build guarantee at test rather than session
    granularity.
    """

    monkeypatch.setenv("QUADRAT_CACHE_DIR", str(tmp_path / "cache"))
