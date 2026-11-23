"""Tracks the training loader piles up itself, from the alignments it is given.

Provides `Alignments`, a channel stated as the alignments it is piled
up from rather than as a track; `managed_tracks`, which piles one up
into an entry keyed by what decides its contents, or finds the entry
already there; `alignments_key`, that key; `resolve_alignments`,
which replaces every `Alignments` in a track specification by the
tracks it piles up into; and `managed_track_dir_factory`, which makes
a scratch directory for the entries where they are not to persist.

Everything here is package-internal: no name carries a ``# public``
marker and none is re-exported from `quadrat`.
"""

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
from importlib.metadata import version

from quadrat._engine import fspath_or_none, item_list
from quadrat.alignments import bam_index_path
from quadrat.cache import file_stamp, hash_reference, make_scratch_dir
from quadrat.pileup import Aggregate, pileup_alignments
from quadrat.tables import (
    MANIFEST_VERSION,
    commit_directory,
    read_manifest,
    require_complete,
    stage_directory,
    track_paths,
)

#: Folded into every key first, so a managed track key can never equal
#: a key minted for anything else from the same bytes.
_KEY_DOMAIN = b"quadrat-managed-tracks"

#: The ``name`` every entry's pileup writes under, inside the entry. One
#: name for every entry, because the entry's directory is what tells
#: two entries apart.
_TRACK_NAME = "channel"


@dataclasses.dataclass(frozen=True)
class Alignments:
    """A channel stated as the alignments it is piled up from.

    `quadrat.loader.build_training_loader` accepts one wherever it
    accepts a track, and piles it up with `quadrat.pileup_alignments`
    before it reads it, as `managed_tracks` states. An unstranded
    ``aggregate`` makes one channel. A stranded one makes two, the
    ``+`` channel then the ``-`` one, and is a group of its own, so a
    mirrored draw swaps them as it swaps a stranded pair of tracks;
    `resolve_alignments` refuses one given inside a group of more.

    A value: it compares and hashes by its fields, and every field is
    validated when it is constructed. It cannot be subclassed: its
    readers trust its fields as validated, which a subclass could
    override.

    :param paths: The alignments, BAM or SAM, as one path or a `list`
        or `tuple` of them, each a `str`, `bytes` or `os.PathLike`.
        Several are pooled into the same channel or channels, as
        `quadrat.pileup_alignments` pools them. Held as a `tuple` of
        `str`.
    :param aggregate: What the pileup counts, as a `quadrat.Aggregate`.
    :raises TypeError: if ``paths`` is neither a path nor a `list` or
        `tuple` of paths, or ``aggregate`` is not a `quadrat.Aggregate`.
    :raises ValueError: if ``paths`` holds no path.
    """

    paths: object
    aggregate: Aggregate = Aggregate()

    def __init_subclass__(cls, **kwargs):
        raise TypeError(
            "Alignments cannot be subclassed, and {} derives from it. Its "
            "readers trust its fields as validated; build one with the "
            "fields you need instead.".format(cls.__name__)
        )

    def __post_init__(self):
        paths = [fspath_or_none(path) for path in item_list(self.paths)]
        if None in paths:
            raise TypeError(
                "Alignments.paths is one path or a list or tuple of paths, "
                "each a str, bytes or os.PathLike; got {!r}.".format(self.paths)
            )
        if not paths:
            raise ValueError(
                "Alignments.paths holds no path: a channel is piled up from at "
                "least one alignment."
            )
        if not isinstance(self.aggregate, Aggregate):
            raise TypeError(
                "Alignments.aggregate is what the pileup counts, as an "
                "Aggregate; got {}.".format(type(self.aggregate).__name__)
            )
        object.__setattr__(self, "paths", tuple(paths))


def alignments_key(alignments, sequences):
    """Return a hash of everything that decides what an entry's tracks hold.

    Over-specified on purpose, as `quadrat.cache.window_cache_key` is,
    because a stale entry would train on the wrong counts and say
    nothing. It holds the installed version of this package and
    `quadrat.MANIFEST_VERSION`, so an upgrade piles everything up
    again; every alignment's absolute path and its size and
    modification time, in the order given; each alignment's index's
    size and modification time, or that it has none; every field of
    the aggregate; and the reference, as `quadrat.cache.hash_reference`
    folds it in.

    :param alignments: The `Alignments`.
    :param sequences: The reference, as a `str`, `bytes` or
        `os.PathLike`.
    :returns: The key, as a hex `str`.
    :raises TypeError: if ``sequences`` is not a path.
    :raises FileNotFoundError: if an alignment does not exist.
    :raises OSError: for a reference `quadrat.cache.hash_reference`
        refuses.

    .. rubric:: Implementation notes

    An alignment is keyed by `quadrat.cache.file_stamp` rather than by
    its bytes. An alignment runs to tens of gigabytes, and hashing it
    on every call would cost a large fraction of the pileup the key
    exists to skip; a stat costs microseconds and catches every rewrite
    that does not restore the size and the modification time to the
    nanosecond. That is the trade `quadrat.cache.window_cache_key`
    makes for a track.
    """

    reference = fspath_or_none(sequences)
    if reference is None:
        raise TypeError(
            "alignments_key hashes the reference it was given: sequences must "
            "be a str, bytes or os.PathLike, got {}.".format(type(sequences).__name__)
        )

    digest = hashlib.sha256()
    digest.update(_KEY_DOMAIN)
    digest.update(version("quadrat").encode())
    digest.update(str(MANIFEST_VERSION).encode())
    for path in alignments.paths:
        digest.update(os.path.abspath(path).encode())
        digest.update(file_stamp(path).encode())
        index = bam_index_path(path)
        digest.update(b"<no index>" if index is None else file_stamp(index).encode())
    digest.update(
        json.dumps(dataclasses.asdict(alignments.aggregate), sort_keys=True).encode()
    )
    hash_reference(digest, reference)
    return digest.hexdigest()


def managed_tracks(alignments, sequences, track_dir, verbose=False):
    """Return the tracks an `Alignments` piles up into, piling them up where absent.

    The entry is ``track_dir/<key>``, the key being `alignments_key`.
    Where a complete entry is there, its tracks are answered and no
    alignment is read. An entry is complete when its record names this
    key and every track passes `quadrat.require_complete`. Otherwise
    `quadrat.pileup_alignments` writes the tracks into a staging
    sibling, which is swapped in as the entry and recorded as complete
    only once every track is written, so an entry is never complete
    before every track is written. An entry is never authoritative: one
    that is missing or incomplete is piled up again.

    Concurrent calls for one entry pile it up once. A call that finds
    the entry complete answers it without a lock, so reading an entry
    needs no write access to ``track_dir``. One that does not takes an
    exclusive lock on ``track_dir/<key>.lock``, looks again, and holds
    the lock until after it commits, so a second call waits for the
    first and then finds the entry complete. The lock is a ``flock``,
    which serialises processes on one host and on a filesystem that
    honours it.

    :param alignments: The `Alignments`.
    :param sequences: The reference, as `quadrat.pileup_alignments`
        takes it.
    :param track_dir: The managed track directory, as a `str`. Created
        if absent.
    :param verbose: Reaches `quadrat.pileup_alignments` unchanged.
    :returns: The entry's track directories, in channel order, as a
        `list` of `str`.
    :raises OSError: on a miss, if ``track_dir`` cannot be created, or
        the entry cannot be locked, staged or committed there, with no
        entry committed.
    :raises: whatever `alignments_key` or `quadrat.pileup_alignments`
        raises, with no entry committed.
    """

    key = alignments_key(alignments, sequences)
    root = os.path.join(track_dir, key)
    paths = track_paths(os.path.join(root, _TRACK_NAME), alignments.aggregate.unstranded)
    # Without the lock first, as the docstring states.
    if _is_complete(root, key, paths):
        return paths
    os.makedirs(track_dir, exist_ok=True)
    with open(root + ".lock", "a") as lock:
        # Released when the file closes, on every exit from the block.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _is_complete(root, key, paths):
            return paths
        staging = stage_directory(root)
        try:
            pileup_alignments(
                list(alignments.paths),
                sequences,
                os.path.join(staging, _TRACK_NAME),
                aggregate=alignments.aggregate,
                verbose=verbose,
            )
            commit_directory(
                root,
                staging,
                {
                    "version": MANIFEST_VERSION,
                    "files": {},
                    "key": key,
                    "alignments": [os.path.abspath(path) for path in alignments.paths],
                    "aggregate": dataclasses.asdict(alignments.aggregate),
                },
            )
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return paths


@contextlib.contextmanager
def managed_track_dir_factory(track_dir, scratch_dir=None):
    """Yield a factory answering the directory managed track entries go in.

    A directory that is named is answered as it is. Where none is, a
    scratch directory is made when the factory is first called, and
    removed with everything in it when the block exits, however it
    exits, so nothing there persists, and a block that never calls the
    factory makes nothing. A scratch directory belongs to one process:
    two processes each make their own, and each piles up its own copy
    of an entry.

    :param track_dir: The managed track directory, as a `str`, `bytes`
        or `os.PathLike`, or ``None`` for a scratch directory.
    :param scratch_dir: Where a scratch directory is made, as
        `quadrat.cache.make_scratch_dir` takes it.
    :raises TypeError: on entry, if ``track_dir`` is neither ``None``
        nor a path.

    The factory raises what `quadrat.cache.make_scratch_dir` raises.
    """

    named = None
    if track_dir is not None:
        named = fspath_or_none(track_dir)
        if named is None:
            raise TypeError(
                "track_dir is a directory or None, got {!r}.".format(track_dir)
            )
    made = None

    def track_dir_factory():
        nonlocal made
        if named is not None:
            return named
        if made is None:
            made = make_scratch_dir("quadrat-tracks-", scratch_dir)
        return made

    try:
        yield track_dir_factory
    finally:
        if made is not None:
            shutil.rmtree(made, ignore_errors=True)


def resolve_alignments(specifications, sequences, track_dir_factory, verbose=False):
    """Return track specifications with every `Alignments` replaced by its tracks.

    Each `Alignments` is answered by `managed_tracks`, in place, and
    its group grows by the channels it adds beyond one. Every other
    track is answered unchanged and is not looked at. Every group of
    every specification is checked before anything is piled up, so a
    refused specification piles nothing up for any other.

    :param specifications: The specifications, e.g., the signals and
        the controls, each ``(tracks, group_sizes)`` as
        `quadrat.groups.normalize_signal_groups` answers them, where
        ``tracks`` may be ``None``.
    :param sequences: The reference, as `managed_tracks` takes it.
    :param track_dir_factory: A zero-argument callable answering the
        managed track directory, as `managed_track_dir_factory` yields
        one, called only where there is an `Alignments` to pile up.
    :param verbose: Reaches `managed_tracks` unchanged.
    :returns: A `list` of ``(tracks, group_sizes)``, one per
        specification, in the order and the form they were given.
    :raises ValueError: if a stranded `Alignments` shares its group
        with any other track, which would make its two channels part
        of a larger group that a mirrored draw reverses as a whole.
    :raises: whatever `managed_tracks` or ``track_dir_factory`` raises,
        unchanged, with nothing piled up for a later `Alignments`.
    """

    grouped = []
    for tracks, group_sizes in specifications:
        groups = []
        position = 0
        for size in group_sizes if tracks is not None else ():
            group = tracks[position : position + size]
            position += size
            if size > 1 and any(
                isinstance(track, Alignments) and not track.aggregate.unstranded
                for track in group
            ):
                raise ValueError(
                    "a stranded Alignments is a signal group of its own, its + and "
                    "- channels: give it as an entry by itself rather than inside "
                    "a group of {}.".format(size)
                )
            groups.append(group)
        grouped.append((tracks, group_sizes, groups))

    resolved = []
    for tracks, group_sizes, groups in grouped:
        if tracks is None:
            resolved.append((tracks, group_sizes))
            continue
        expanded_tracks = []
        sizes = []
        for group in groups:
            expanded = []
            for track in group:
                if isinstance(track, Alignments):
                    expanded.extend(
                        managed_tracks(
                            track, sequences, track_dir_factory(), verbose=verbose
                        )
                    )
                else:
                    expanded.append(track)
            expanded_tracks.extend(expanded)
            sizes.append(len(expanded))
        resolved.append((expanded_tracks, sizes))
    return resolved


def _is_complete(root, key, paths):
    """Whether ``root`` is a committed entry for ``key`` whose every track is complete."""

    record = read_manifest(root)
    if record is None or record.get("key") != key:
        return False
    try:
        for path in paths:
            require_complete(path)
    except ValueError:
        return False
    return True
