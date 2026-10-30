"""Tests for the version the build backend derives from the repository.

Releasing is pushing a tag, so the derivation is the whole of the
release process and every branch of it is exercised here against a
real repository rather than a mocked one: what git reports is the
thing under test.
"""

from tests import load_build_hook, run_child

version = load_build_hook("_git").version


def git(root, *arguments):
    """Run a git command in ``root`` and return its stdout.

    The identity is supplied per invocation so a machine with no
    configured committer still builds the repositories these tests
    need.
    """

    command = [
        "git",
        "-C",
        str(root),
        "-c",
        "user.email=suite@example.invalid",
        "-c",
        "user.name=Test Suite",
        *arguments,
    ]
    return run_child(command, "git {} failed".format(arguments[0])).stdout.strip()


def repository(root, *, tag=None):
    """Build a one-commit git repository at ``root``, optionally tagged."""

    root.mkdir(parents=True)
    git(root, "init")
    (root / "tracked.txt").write_text("one\n")
    git(root, "add", "tracked.txt")
    git(root, "commit", "-m", "one")
    if tag is not None:
        git(root, "tag", tag)
    return root


def commit(root, message):
    """Add one empty commit to ``root`` and return its abbreviated sha."""

    git(root, "commit", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "--short=7", "HEAD")


def test_version_should_report_the_release_when_the_commit_is_tagged(tmp_path):
    """Test that a tagged, clean commit is exactly its release.

    Given:
        A clean repository whose head commit carries a release tag.
    When:
        The version is derived.
    Then:
        It should be the tag's release and nothing else, because a
        tagged commit is the release rather than a build on top of one.
    """

    # Arrange
    root = repository(tmp_path / "repo", tag="v1.2.3")

    # Act
    derived = version(root)

    # Assert
    assert derived == "1.2.3"


def test_version_should_add_the_commit_when_the_tag_is_behind_the_head(tmp_path):
    """Test that an untagged commit reports the release it follows.

    Given:
        A clean repository with a release tag and a later commit.
    When:
        The version is derived.
    Then:
        It should be the tagged release plus the head's abbreviated
        commit, so the build is traceable to a commit without claiming
        a release of its own.
    """

    # Arrange
    root = repository(tmp_path / "repo", tag="v1.2.3")
    head = commit(root, "two")

    # Act
    derived = version(root)

    # Assert
    assert derived == "1.2.3+{}".format(head)


def test_version_should_add_dirty_when_the_tree_has_uncommitted_content(tmp_path):
    """Test that a modified tracked file marks the version dirty.

    Given:
        A repository tagged at its head with one tracked file modified.
    When:
        The version is derived.
    Then:
        It should append a dirty segment, so an artifact built from a
        tree that no commit describes says so.
    """

    # Arrange
    root = repository(tmp_path / "repo", tag="v1.2.3")
    (root / "tracked.txt").write_text("two\n")

    # Act
    derived = version(root)

    # Assert
    assert derived == "1.2.3+dirty"


def test_version_should_add_dirty_when_the_tree_has_untracked_content(tmp_path):
    """Test that an untracked file marks the version dirty too.

    Given:
        A repository tagged at its head with one untracked file added.
    When:
        The version is derived.
    Then:
        It should append a dirty segment, because the build's default
        include is broad enough that an untracked file can reach the
        artifact.
    """

    # Arrange
    root = repository(tmp_path / "repo", tag="v1.2.3")
    (root / "untracked.txt").write_text("two\n")

    # Act
    derived = version(root)

    # Assert
    assert derived == "1.2.3+dirty"


def test_version_should_report_the_initial_release_when_no_tag_is_reachable(tmp_path):
    """Test that an untagged repository still names its commit.

    Given:
        A clean repository carrying no tags at all.
    When:
        The version is derived.
    Then:
        It should be the initial release plus the head's abbreviated
        commit, so a repository before its first release still
        produces a version that orders and identifies the build.
    """

    # Arrange
    root = repository(tmp_path / "repo")
    head = git(root, "rev-parse", "--short=7", "HEAD")

    # Act
    derived = version(root)

    # Assert
    assert derived == "0.0.0+{}".format(head)


def test_version_should_ignore_a_tag_that_does_not_name_a_release(tmp_path):
    """Test that only a release-shaped tag is read as a release.

    Given:
        A clean repository whose head carries a tag that is not a
        three-part version.
    When:
        The version is derived.
    Then:
        It should fall back to the initial release and the commit,
        because a tag outside the release shape is somebody's pointer
        rather than a version to publish.
    """

    # Arrange
    root = repository(tmp_path / "repo", tag="latest")
    head = git(root, "rev-parse", "--short=7", "HEAD")

    # Act
    derived = version(root)

    # Assert
    assert derived == "0.0.0+{}".format(head)


def test_version_should_read_the_packaged_version_when_there_is_no_repository(tmp_path):
    """Test that an unpacked distribution reports the version it carries.

    Given:
        A directory that is no repository, holding the metadata a
        source distribution ships.
    When:
        The version is derived.
    Then:
        It should be the version recorded in that metadata, so a wheel
        built from a distribution reports the tree it came from rather
        than failing for want of a repository.
    """

    # Arrange
    root = tmp_path / "unpacked"
    root.mkdir()
    (root / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\nName: quadrat\nVersion: 4.5.6+abc1234\n"
    )

    # Act
    derived = version(root)

    # Assert
    assert derived == "4.5.6+abc1234"


def test_version_should_prefer_the_packaged_version_to_an_enclosing_repository(tmp_path):
    """Test that a distribution inside a checkout ignores its tags.

    Given:
        A repository tagged with one release, containing an unpacked
        distribution directory whose metadata names another.
    When:
        The version of that inner directory is derived.
    Then:
        It should be the version the distribution carries, because git
        searches upward and would otherwise hand the distribution a
        release belonging to a project it has nothing to do with.
    """

    # Arrange
    outer = repository(tmp_path / "outer", tag="v9.9.9")
    root = outer / "unpacked"
    root.mkdir()
    (root / "PKG-INFO").write_text("Version: 4.5.6\n")

    # Act
    derived = version(root)

    # Assert
    assert derived == "4.5.6"


def test_version_should_report_the_initial_release_when_nothing_names_a_version(
    tmp_path,
):
    """Test that an unversionable tree still builds.

    Given:
        A directory that is neither a repository nor an unpacked
        distribution.
    When:
        The version is derived.
    Then:
        It should be the initial release, so a build from an exported
        tree produces an artifact rather than an error.
    """

    # Arrange
    root = tmp_path / "bare"
    root.mkdir()

    # Act
    derived = version(root)

    # Assert
    assert derived == "0.0.0"
