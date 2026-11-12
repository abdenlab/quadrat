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

Outside a pileup, nothing that registers a table under one of the
fixed names below takes that guard. Two such registrations of one name
running at once in one process would bind and drop each other's
tables, so concurrent calls of anything that makes them are
unsupported for the same reason.
"""

import contextlib
import importlib
import os
import threading

FASTA_TABLE = "quadrat_fasta"


SIZES_TABLE = "quadrat_chrom_sizes"


#: The windows side of the window-grouping join: one contig's windows.
#: A fixed name rather than a generated one, because within one call
#: the join runs once per contig and binds exactly one frame under each
#: side's name at a time. The module docstring says why two calls at
#: once in one process are unsupported.
WINDOWS_TABLE = "quadrat_windows"


#: The track side of the window-grouping join: every channel's rows on
#: one contig in a single frame, with the channel carried as a column
#: the join projects rather than as a registration of its own.
TRACK_TABLE = "quadrat_track"


#: The windows side of the exclusion join: each window's span as a
#: half-open run of exclusion bins, `quadrat.loci.EXCLUSION_BIN_WIDTH`
#: bases wide. Fixed for the reason `WINDOWS_TABLE` is, the join running
#: once per call.
LOCUS_BINS_TABLE = "quadrat_locus_bins"


#: The exclusion side of the exclusion join: each excluded region as a
#: half-open run of the same bins.
EXCLUSION_BINS_TABLE = "quadrat_exclusion_bins"


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
#: the module. Every other module `import_on_first_use` reaches is a
#: hard dependency and wants a different sentence.
_PILEUP_EXTRA = ("giql", "polars_bio")


#: Held for the duration of one pileup. Not a queue: a second pileup in
#: the same process is a bug in the caller rather than work to be
#: scheduled, and blocking would turn a contract violation into a hang.
_pileup_lock = threading.Lock()


#: Which third-party process-global state `engine` has already set, in
#: this process. A plain set rather than SCREAMING_CASE because the
#: binding is constant and the referent is not: `engine` fills it on
#: first use. It exists because `engine` runs on every query rather
#: than once, and one of the things it does is not idempotent the way
#: the rest of it is -- see the write-lock paragraph in its rubric.
_installed = set()


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

    One definition of "is this a path" for the package. What
    legitimately differs between the places that ask is the refusal,
    which stays with each of them: a reader answers ``None`` where
    nothing is checkable, a guard raises the error its own contract
    names, and an entry point leaves a non-path alone so the layer
    below refuses it in its own words. What does not differ is this.

    ``os.fsdecode`` rather than ``os.fspath``, and that is the whole of
    the `bytes` question. A `bytes` path is a path everywhere else in
    Python, and ``os.fspath`` hands `bytes` straight through, so the
    value would reach a ``str``-only provider, a ``str``
    concatenation, or `str`, which builds
    ``"b'/data/x.track.parquet'.manifest.json"`` out of a repr and
    calls it a path. ``os.fsdecode`` answers a `str` for all three
    accepted shapes, and applies ``os.fspath`` first, so one call
    covers them.

    ``None`` rather than raising, and the name says so: ``os.fspath``
    raises for a non-path, and a caller here is usually about to decide
    what its own refusal should be.
    """

    if isinstance(value, (str, bytes, os.PathLike)):
        return os.fsdecode(value)
    return None


def name_list(names, empty, limit=3):
    """Return the first few of ``names`` as message text, with a count of the rest.

    :param names: The names, as any iterable: contigs, paths, or
        whatever else a message has to list without printing all of.
    :param empty: The text that stands for no names at all.
    :param limit: How many to spell out.
    :returns: The names in backticks and comma-separated, followed by
        ``and N more`` when there are more than ``limit``, or ``empty``
        when there are none.
    """

    names = [str(name) for name in names]
    shown = ", ".join("`{}`".format(name) for name in names[:limit])
    if len(names) > limit:
        return "{} and {} more".format(shown, len(names) - limit)
    return shown or empty


@contextlib.contextmanager
def registered_fasta(path):
    """Bind ``path`` as `FASTA_TABLE` for the duration of the block.

    When a reference is scanned, and so registered here, is for
    `quadrat.reference.chrom_lengths` to say.

    Yields the engine handle, so a caller needs one call rather than
    two for what is one operation.

    .. rubric:: Implementation notes

    A context manager rather than a cache, because a registration has
    to be dropped by something and a cached value has no teardown hook
    to drop it with.
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
    the way in and dropped on the way out, however the block exits.

    Yields the engine handle, so a caller needs one call rather than
    two for what is one operation.

    :param frames: A mapping of table name to polars frame. Iteration
        order is the registration order, which matters only in that a
        failure partway through still drops whatever landed.

    .. rubric:: Implementation notes

    Dropped in a ``finally``, and that is the whole reason this exists
    rather than two lines at each call site. The session is
    process-global, as the module docstring above says, so a
    registration that outlives its query stays bound for the life of
    the process -- and the next caller of the same name reads the
    previous one's rows rather than failing.
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

    Availability *is* guarded. ``pyproject.toml`` puts polars-bio and
    giql in ``[project.optional-dependencies] pileup``, so a base install
    reaches here with the module genuinely absent -- and a bare
    ``ModuleNotFoundError`` raised from the middle of a pileup names
    neither the extra nor the remedy. Everything else reached through
    here is a hard dependency, so its absence means an incomplete
    environment rather than a minimal one, and the second message
    below says so.

    One accessor rather than one per library. One-line functions
    differing only in a module name would be a parallel surface.
    """

    try:
        return importlib.import_module(name)
    except ImportError as error:
        if name in _PILEUP_EXTRA:
            raise ImportError(
                "{} is needed to build tracks from alignments and to "
                "join windows against tracks or exclusion lists, and is "
                "not installed. It belongs to quadrat's `pileup` extra: "
                "install it with `uv sync --all-extras` in a checkout, "
                'or `pip install "quadrat[pileup]"` once the extra\'s '
                "git-pinned dependency reaches an index -- the comment "
                "on `[tool.uv.sources]` in pyproject.toml says when. A "
                "base install reads tracks that already exist, one "
                "window at a time; building them and joining "
                "against them both need this.".format(name)
            ) from error
        raise ImportError(
            "{} is declared as a dependency of quadrat and is not "
            "importable, so this environment is incomplete rather than "
            "minimal. Reinstall the package.".format(name)
        ) from error


def engine():
    """Return polars-bio, imported on first use, with its progress bars off.

    .. rubric:: Implementation notes

    ``TQDM_DISABLE`` is most of the quieting, and it is set with
    ``setdefault`` so a caller who wants the bars back can have them.
    It is not all of it, because tqdm reads its ``TQDM_`` overrides in
    the ``envwrap`` decorator applied at import of ``tqdm.std`` -- so
    the variable has to be set before anything imports tqdm, which is
    what the deferred imports at this package's call sites buy.

    The write lock is the other half, and it is a correctness matter
    rather than a cosmetic one. ``tqdm.__new__`` builds its lock
    before it consults ``disable``, and the default lock holds a
    `multiprocessing.RLock` whose construction binds this process's
    start method. A caller who had not yet chosen ``fork`` could then
    no longer choose it, and the residency `quadrat.cache` derives
    from that choice would be settled by this import rather than by
    the program. Installing a `threading.RLock` first means
    ``tqdm.get_lock`` never reaches the default, so importing the
    engine costs the caller nothing it did not ask for.

    That installation happens once per process, and this function runs
    once per query, so "once" has to be arranged rather than assumed.
    ``set_lock`` is tqdm's own published API: a second call would
    discard whatever lock the host application had installed, and a bar
    already live in another thread would hold the previous lock while
    new bars took this one, two locks guarding one stream. The obvious
    guard is the wrong one, because ``tqdm.tqdm.get_lock`` *constructs*
    the default lock when none is set and so would do the very thing
    this avoids; `_installed` records the installation instead.

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
    if "tqdm-lock" not in _installed:
        # Before polars-bio imports tqdm, and once per process; the
        # rubric says why.
        #
        # Guarded although tqdm is a base dependency, so it is absent
        # only from an incomplete environment. A tqdm that cannot be
        # imported cannot build a lock either, so there is nothing left
        # to prevent, and the refusal is left to the polars-bio import
        # below, which imports tqdm itself. The flag is set only on
        # success, so a later call retries rather than recording the
        # absence as done.
        try:
            tqdm = import_on_first_use("tqdm")
        except ImportError:
            pass
        else:
            tqdm.tqdm.set_lock(threading.RLock())
            _installed.add("tqdm-lock")
    return import_on_first_use("polars_bio")


def transpile_intervals(query, *specs):
    """Return a GIQL interval query transpiled to the SQL the engine runs.

    Nothing is kept between calls. The transpilation is pure, so a
    caller whose query never varies holds the result itself.

    :param query: The GIQL text, naming each table by the name its
        frame is registered under.
    :param specs: One ``(table, start column, end column)`` triple per
        table the query treats as intervals. Each is half-open over
        its two columns, on a contig column named ``chrom``, and
        unstranded.
    :returns: The SQL, as a `str`.
    :raises ImportError: if the ``pileup`` extra is not installed.

    .. rubric:: Implementation notes

    Transpiled rather than written, because an interval predicate has
    no operator in the target dialect: giql turns ``INTERSECTS`` into
    the bound comparisons the engine plans as an interval join.
    """

    giql = import_on_first_use("giql")
    return giql.transpile(
        query,
        tables=[
            giql.Table(
                name, chrom_col="chrom", start_col=start, end_col=end, strand_col=None
            )
            for name, start, end in specs
        ],
        dialect="datafusion-bio",
    )


def sql_string(value):
    """One SQL string literal, with quotes doubled.

    Contig names come from a BAM header rather than from a user, but they
    are arbitrary bytes as far as the format is concerned, and a name
    carrying an apostrophe would otherwise end the literal early.
    """

    return "'{}'".format(str(value).replace("'", "''"))
