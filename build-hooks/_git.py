"""Derive the distribution's version from the repository's tag history.

A release is named by a ``vMAJOR.MINOR.PATCH`` tag and nothing else is
a release. A commit carrying no tag takes the last tagged release plus
a local identifier naming the commit it is, so every build is
traceable to one without inventing a release number for it. There are
no pre-release or development segments, because there is no release
branch for one to describe.

.. rubric:: Implementation notes

``git`` is invoked as a subprocess rather than through a library, so
the build requires nothing beyond the backend itself and an offline
build cannot fail for want of a wheel to resolve.
"""

import pathlib
import re
import subprocess

# The only tags that name a release. Anything else in the tag
# namespace -- a moved pointer, a `latest`, something pushed by hand --
# is ignored rather than guessed at.
RELEASE_TAG = "v[0-9]*.[0-9]*.[0-9]*"

# `git describe` renders a tagged commit as the tag alone and any other
# as `<tag>-<distance>-g<abbrev>`. The distance is discarded, because a
# count of commits since a release is not a release.
DESCRIBED = re.compile(r"^v(?P<release>\d+\.\d+\.\d+)(?:-\d+-g(?P<commit>[0-9a-f]+))?$")

# What a version falls back to before the first release tag exists, and
# what an unversionable tree reports.
INITIAL_RELEASE = "0.0.0"


def _git(root, *arguments):
    """Return the stdout of a git command in ``root``, or ``None``."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(root)) + arguments,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _is_own_repository(root):
    """Return whether ``root`` is itself the root of a git repository."""

    # `git -C` searches upward, so an unpacked sdist sitting inside an
    # unrelated checkout would otherwise take that repository's tags.
    # The project root and the repository root are the same directory
    # here, so requiring them to match rules that out.
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if toplevel is None:
        return False
    return pathlib.Path(toplevel).resolve() == pathlib.Path(root).resolve()


def _packaged_version(root):
    """Return the version an unpacked sdist already carries, or ``None``."""

    # An sdist has no history to read, but the build that produced it
    # wrote the version it resolved into PKG-INFO, so a wheel built
    # from one reports the version of the tree it came from rather than
    # failing for want of a repository.
    try:
        metadata = (pathlib.Path(root) / "PKG-INFO").read_text()
    except OSError:
        return None
    match = re.search(r"^Version: *(.+)$", metadata, re.MULTILINE)
    return match.group(1).strip() if match else None


def version(root):
    """Return the PEP 440 version of the tree at ``root``.

    A tagged, clean commit is exactly its release. Any other commit in
    a repository is the last release plus a local identifier, i.e. the
    abbreviated commit and ``dirty`` where the tree has uncommitted or
    untracked content. A tree that is not a repository falls back to
    the version an sdist carries, and failing that to the initial
    release.
    """

    if not _is_own_repository(root):
        return _packaged_version(root) or INITIAL_RELEASE

    described = _git(root, "describe", "--tags", "--abbrev=7", "--match", RELEASE_TAG)
    match = DESCRIBED.match(described) if described else None
    if match:
        release, commit = match.group("release"), match.group("commit")
    else:
        # No release tag is reachable, so the commit is all there is to
        # name the build by.
        release = INITIAL_RELEASE
        commit = _git(root, "rev-parse", "--short=7", "HEAD")

    local = [commit] if commit else []
    # Untracked files count as well as modified ones, because the
    # default include is broad enough that one can reach the artifact.
    if _git(root, "status", "--porcelain"):
        local.append("dirty")

    # A period, not the hyphen git uses: PEP 440 separates local
    # segments with periods and normalises anything else to one, so
    # emitting the canonical form keeps the string the build reports
    # identical to the string a resolver compares.
    return "{}+{}".format(release, ".".join(local)) if local else release
