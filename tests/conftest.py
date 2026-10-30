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
    that rather than built and empty. ``bams`` is an empty list for the
    same reason.

    :param root: The directory holding this corpus's files, as a
        `pathlib.Path`. `Corpus.path` returns members of it as strings,
        which is the form the builders take.
    :param chrom_sizes: The geometry the corpus was built to, copied
        rather than aliased, so a caller mutating its own mapping
        afterwards cannot alter what this corpus reports.

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
        "peaks",
        "negatives",
        "exclusion",
    )

    def __init__(self, root, chrom_sizes):
        self.root = root
        self.chrom_sizes = dict(chrom_sizes)
        self.fasta = None
        self.bams = []
        self.peaks = None
        self.negatives = None
        self.exclusion = None

    def path(self, name):
        """Return a path inside this corpus's directory, as a string."""

        return str(self.root / name)


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

    Window geometry is not a parameter: nothing here renders a window,
    so a suite that needs one supplies it itself.

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
        peaks=(),
        negatives=(),
        exclusion=(),
    ):
        if n_bams or index_fasta:
            pytest.importorskip("pysam")
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
        corpus = Corpus(root, chrom_sizes)
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
        return corpus

    return build
