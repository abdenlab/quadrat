"""The one polars-bio session, and the tables registered in it.

polars-bio holds a single process-global session, so every table name
here carries a ``quadrat_`` prefix and nothing registers a name a
caller might also be using. The prefix is not decoration: cherimoya
registers the same shapes under ``cherimoya_``, and during the
transition both packages may be imported into one process.

The same globalness is why one process runs one pileup at a time. Two
at once would register their alignments under these same names and read
each other's reads, and neither table would look wrong afterwards --
a manifest listing the files its writer wrote is a manifest whose files
all exist, so both pass `require_complete` and training
reads the contamination as signal. `exclusive_pileup` is the
guard. Concurrency is still available through
`pileup_many`'s ``process`` backend, where each
worker is a process with a session of its own.
"""

import contextlib
import importlib
import os
import threading

FASTA_TABLE = "quadrat_fasta"


SIZES_TABLE = "quadrat_chrom_sizes"


#: The two sides of the window-grouping join: the windows a locus set
#: asks for, and one channel's track rows. Fixed names rather than
#: generated ones, because the join registers exactly one of each at a
#: time -- it runs per contig and per channel, and a name per channel
#: would leave as many providers bound as there are channels.
WINDOWS_TABLE = "quadrat_windows"


TRACK_TABLE = "quadrat_track"


#: Prefix of the one table name that is generated rather than fixed:
#: `register_alignments` numbers its tables from it. A constant here
#: rather than a parameter default on that function, because the
#: module docstring above states the `quadrat_` prefix as a property of
#: this package and a caller able to pass `cherimoya_bam` would make
#: that false -- registering precisely the colliding names the prefix
#: exists to prevent.
BAM_TABLE_PREFIX = "quadrat_bam"


#: Modules the ``pileup`` extra installs rather than the base install.
#: Named here so a failed import can state the remedy instead of only
#: the module -- `import_on_first_use` is reached for `joblib` too,
#: which is a hard dependency and wants a different sentence.
_PILEUP_EXTRA = ("giql", "polars", "polars_bio", "pyfaidx")


#: Held for the duration of one pileup. Not a queue: a second pileup in
#: the same process is a bug in the caller rather than work to be
#: scheduled, and blocking would turn a contract violation into a hang.
_pileup_lock = threading.Lock()


# public
class EngineOptionWarning(UserWarning):
    """Emitted when the query engine refuses a setting quadrat asked for.

    The pileup carries on. Every option this covers tunes how the
    engine spends a machine rather than what it computes, so a refused
    one costs time and cannot change a table.

    A warning rather than an exception because the engine is a
    third-party boundary whose option names move between releases, and
    ending a run that would have produced a correct table is the worse
    of the two failures. A caller who disagrees can say so::

        warnings.simplefilter("error", quadrat.EngineOptionWarning)

    Derived from `UserWarning` rather than from `Warning` directly, so
    that a caller already filtering `UserWarning` keeps seeing it.
    """


def fspath_or_none(value):
    """``value`` as a filesystem path string, or ``None`` if it is not one.

    One definition of "is this a path" for the package. Five sites had
    their own -- the three readers in `quadrat.tables`, the two guards
    in `quadrat.reference` and `quadrat.alignments`, and the two pileup
    entry points -- each spelling the probe and the conversion itself,
    and `quadrat.pileup.track`'s own comment documents the five as a
    set. What legitimately differs between them is the refusal, which
    stays with each site: a reader answers ``None`` where nothing is
    checkable, `chrom_lengths` raises `TypeError`, `register_alignments`
    raises `ValueError` naming the suffix rule, and the entry points
    leave a non-path alone so the layer below refuses it in its own
    words. What does not differ is this.

    ``os.fsdecode`` rather than ``os.fspath``, and that is the whole of
    the `bytes` question. A `bytes` path is a path everywhere else in
    Python, both entry points admitted one, and nothing downstream
    honoured it: ``os.fspath`` hands `bytes` straight through, so the
    value reached a ``str``-only provider, a ``str`` concatenation, or
    `str`, which built ``"b'/data/x.track.parquet'.manifest.json"`` out
    of a repr and called it a path. ``os.fsdecode`` answers a `str` for
    all three accepted shapes, and applies ``os.fspath`` first, so one
    call covers them.

    ``None`` rather than raising, and the name says so: ``os.fspath``
    raises for a non-path, and a caller here is usually about to decide
    what its own refusal should be.
    """

    if isinstance(value, (str, bytes, os.PathLike)):
        return os.fsdecode(value)
    return None


@contextlib.contextmanager
def registered_fasta(path):
    """Bind ``path`` as `FASTA_TABLE` for the duration of the block.

    The scope the registration did not have. A FASTA provider is a file
    handle held open inside the process-global DataFusion context, and
    it was recorded as a key in a dict: a value has no teardown hook,
    so nothing ever dropped the table and the provider stayed bound for
    the life of the process. The only documented release said it
    reclaimed the registration and cleared three dict slots.

    A context manager rather than a cache, because the retention bought
    nothing. `FASTA_TABLE` has exactly one reader -- the one query
    `quadrat.reference.chrom_lengths` runs when a reference has no
    usable index -- and the *lengths* that query produces are what the
    caller comes back for, so a second call is answered from the length
    cache and never reaches here. Keeping the provider bound in case of
    a second read meant keeping it bound for a read that cannot happen.

    Yields the engine handle, so a caller needs one call rather than
    two for what is one operation.
    """

    pb = engine()
    pb.register_fasta(path, FASTA_TABLE)
    try:
        yield pb
    finally:
        # Through the DataFusion context for the reasons
        # `quadrat.alignments.deregister_alignments` records, which are
        # the same here and stated there: it does not go through the
        # parser and does not raise on a name that is not bound.
        pb.ctx.deregister_table(FASTA_TABLE)


@contextlib.contextmanager
def registered_frames(frames):
    """Bind ``frames`` by table name for the duration of the block.

    What `registered_fasta` is for a file, for the in-memory frames a
    query is handed: a mapping of table name to polars frame, bound on
    the way in and dropped on the way out.

    Dropped in a ``finally``, and that is the whole reason this exists
    rather than two lines at each call site. The session is
    process-global, as the module docstring above says, so a
    registration that outlives its query stays bound for the life of
    the process -- and the next caller of the same name reads the
    previous one's rows rather than failing.

    Yields the engine handle, so a caller needs one call rather than
    two for what is one operation.

    :param frames: A mapping of table name to polars frame. Iteration
        order is the registration order, which matters only in that a
        failure partway through still drops whatever landed.
    """

    pb = engine()
    try:
        for name, frame in frames.items():
            pb.from_polars(name, frame)
        yield pb
    finally:
        # Through the DataFusion context, for the reasons
        # `quadrat.alignments.deregister_alignments` records: it does
        # not go through the parser and does not raise on a name that
        # is not bound, which is what makes this safe after a
        # registration that refused partway through.
        for name in frames:
            pb.ctx.deregister_table(name)


@contextlib.contextmanager
def exclusive_pileup():
    """Hold this process's one pileup slot, or refuse to start.

    The precondition is stated in the module docstring above and on
    both public entry points. It is enforced here rather than
    documented alone because the failure is silent: two pileups that
    overlap produce two complete-looking tables, each carrying some of
    the other's reads, and nothing downstream can tell.

    Non-blocking deliberately. See `_pileup_lock`.
    """

    if not _pileup_lock.acquire(blocking=False):
        raise RuntimeError(
            "a pileup is already running in this process. polars-bio "
            "holds one process-global session and every pileup registers "
            "its alignments under the same table names, so two at once "
            "would read each other's reads and both tables would still "
            "pass require_complete. Run them one after another, or use "
            "pileup_many(backend='process'), which gives each pileup a "
            "process of its own."
        )
    try:
        yield
    finally:
        _pileup_lock.release()


def import_on_first_use(name):
    """Import a third-party module the first time something needs it.

    A function-local import is forbidden and the remedy named in the
    same breath is a lazy-loading primitive, so this is that primitive.
    It defers nothing of its own -- the deferral belongs to the call
    site, which is why the name says *on first use* rather than
    *lazily*. What it adds is one greppable seam for every import that
    has to stay out of package import time, which is the guarantee
    ``tests/test_quadrat.py`` pins in a child interpreter, and the
    refusal below.

    Availability *is* guarded, unlike what this said while it was
    called ``lazy``. ``pyproject.toml`` puts polars, polars-bio, giql
    and pyfaidx in ``[project.optional-dependencies] pileup``, so a
    base install reaches here with the module genuinely absent -- and a
    bare ``ModuleNotFoundError`` raised from the middle of a pileup
    names neither the extra nor the remedy.

    One accessor rather than one per library. Four one-line functions
    differing only in a module name would be a parallel surface,
    introduced by a change made to remove one.
    """

    try:
        return importlib.import_module(name)
    except ImportError as error:
        if name in _PILEUP_EXTRA:
            raise ImportError(
                "{} is needed to pile up alignments and is not "
                "installed. It belongs to quadrat's `pileup` extra: "
                "install it with `uv sync --all-extras` in a checkout, "
                'or `pip install "quadrat[pileup]"` once the extra\'s '
                "git-pinned dependency reaches an index -- the comment "
                "on `[tool.uv.sources]` in pyproject.toml says when. A "
                "base install reads tracks that already exist and "
                "cannot write new ones.".format(name)
            ) from error
        raise ImportError(
            "{} is declared as a dependency of quadrat and is not "
            "importable, so this environment is incomplete rather than "
            "minimal. Reinstall the package.".format(name)
        ) from error


def engine():
    """Return polars-bio, imported on first use, with its progress bars off.

    .. rubric:: Implementation notes

    ``TQDM_DISABLE`` is the whole of the quieting, and it is set with
    ``setdefault`` so a caller who wants the bars back can have them.

    ``pb.set_loglevel`` is deliberately not called here, and no level
    makes it worth calling. It accepts ``debug``, ``info`` and
    ``warn``/``warning`` and nothing else, so there is no level
    quieter than the ``WARNING`` ``polars_bio`` already sets its own
    logger to at import -- nothing to gain. And it sets the *root*
    logger and calls `logging.basicConfig`, so reaching for it at all
    reconfigures the logging of whatever application imported us in
    order to gain that nothing.

    Quieting polars-bio below ``WARNING`` is therefore a decision to
    suppress a library's warnings, not a default -- and it belongs to
    the caller, as ``logging.getLogger("polars_bio")``, rather than
    here.
    """

    os.environ.setdefault("TQDM_DISABLE", "1")
    return import_on_first_use("polars_bio")


def sql_string(value):
    """One SQL string literal, with quotes doubled.

    Contig names come from a BAM header rather than from a user, but they
    are arbitrary bytes as far as the format is concerned, and a name
    carrying an apostrophe would otherwise end the literal early.
    """

    return "'{}'".format(str(value).replace("'", "''"))
