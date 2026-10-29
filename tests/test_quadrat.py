"""Tests for the package surface and the distribution built from it.

The package surface is a docstring, a version and an `__all__`, so what
is worth pinning is not behaviour but the wiring
between the source tree and the artifact a user installs: the version
the build backend reads, the export list and the marker convention it
agrees with, what an import of the package costs, and what reaches a
source distribution.

The last of those spawn a subprocess or a real build, because once
pytest is running, its plugins and the suite's own modules have already
populated `sys.modules` and the question cannot be asked.
"""

import ast
import importlib.metadata
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import quadrat
from tests import PROJECT_ROOT, load_build_hook, run_child

# The import names of what the ``pileup`` extra installs. The package
# docstring states the guarantee these are checked against.
PILEUP_MODULES = ("polars", "polars_bio", "giql", "pyfaidx")


def declared_names(node):
    """Return the names one top-level declaration binds."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return set()

    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(
                element.id for element in target.elts if isinstance(element, ast.Name)
            )
    return names


def marked_public(source):
    """Return the names declared under a ``# public`` marker in ``source``.

    A decorated declaration is measured from its first decorator, since
    that is where the marker has to sit to read as marking it.
    """

    lines = source.splitlines()
    names = set()
    for node in ast.parse(source).body:
        decorators = getattr(node, "decorator_list", [])
        first = min([node.lineno] + [item.lineno for item in decorators])
        if first > 1 and lines[first - 2].strip() == "# public":
            names.update(declared_names(node))
    return names


def uv_build(project, out_dir, kind):
    """Build a ``"sdist"`` or ``"wheel"`` from ``project`` and return its path.

    A failed build is reported with its stderr, which
    `subprocess.CalledProcessError` carries but does not print: without
    it a cold uv cache reads as an unexplained non-zero exit.
    """

    command = [
        "uv",
        "build",
        "--offline",
        "--" + kind,
        "--out-dir",
        str(out_dir),
        str(project),
    ]
    run_child(command, "uv build --{} failed".format(kind))

    (archive,) = Path(out_dir).glob("*.tar.gz" if kind == "sdist" else "*.whl")
    return archive


def sdist_members(project, out_dir):
    """Return the member paths of a source distribution built from ``project``.

    The archive's leading version directory is stripped, so a member
    reads as the path it occupies in the project tree.
    """

    with tarfile.open(uv_build(project, out_dir, "sdist")) as handle:
        return {name.split("/", 1)[1] for name in handle.getnames() if "/" in name}


def wheel_metadata(project, out_dir):
    """Return the metadata of a wheel built from ``project``, as text."""

    with zipfile.ZipFile(uv_build(project, out_dir, "wheel")) as handle:
        (name,) = [
            member
            for member in handle.namelist()
            if member.endswith(".dist-info/METADATA")
        ]
        return handle.read(name).decode()


derived_version = load_build_hook("_git").version


class TestDistribution:
    """The package's public surface and its built artifacts."""

    def test___version___should_match_the_installed_distribution(self):
        """Test that the declared version is the one that ships.

        Given:
            An installed distribution of the package.
        When:
            The declared version and the distribution metadata version
            are compared.
        Then:
            It should report the same version from both, so the
            attribute reads the version the build resolved rather than
            carrying a literal that can drift from it.
        """

        # Act
        declared = quadrat.__version__
        packaged = importlib.metadata.version("quadrat")

        # Assert
        assert declared == packaged

    def test___all___should_bind_only_its_names_when_star_imported(self):
        """Test that the export list is the whole public contract.

        Given:
            The package's declared export list.
        When:
            The package is star-imported into a fresh namespace.
        Then:
            It should bind exactly the declared names, so `__all__` is
            enforced rather than documentary.
        """

        # Arrange
        namespace = {}

        # Act
        exec("from quadrat import *", namespace)

        # Assert
        bound = set(namespace) - {"__builtins__"}
        assert bound == set(quadrat.__all__)

    def test___all___should_name_every_symbol_marked_public(self):
        """Test that the marker convention and the export list agree.

        Given:
            Every module of the installed package.
        When:
            The top-level declarations carrying a ``# public`` marker
            are collected across all of them.
        Then:
            It should name exactly the package's export list, so a
            symbol marked public but never re-exported -- or
            re-exported without the marker -- fails here rather than
            drifting until someone reads both files.
        """

        # Arrange
        root = Path(quadrat.__file__).parent

        # Act
        marked = set()
        for path in sorted(root.rglob("*.py")):
            marked |= marked_public(path.read_text())

        # Assert
        assert marked == set(quadrat.__all__)

    @pytest.mark.packaging
    def test_quadrat_should_import_none_of_the_pileup_engine(self):
        """Test that a base install never pays for the pileup engine.

        Given:
            A fresh interpreter in an environment where the optional
            dependencies are installed.
        When:
            The package is imported and the loaded modules inspected.
        Then:
            It should have loaded none of polars, polars-bio, giql or
            pyfaidx, so a caller who already has cut-site tables never
            pays for the engine that builds them.
        """

        # Arrange
        # An absent module is indistinguishable from one the package
        # declined to import, so a base install skips rather than fails.
        for name in PILEUP_MODULES:
            pytest.importorskip(name)

        program = (
            "import sys, quadrat\n"
            "names = {!r}\n"
            "print(' '.join(n for n in names if n in sys.modules))\n".format(
                PILEUP_MODULES
            )
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.split() == []

    @pytest.mark.packaging
    @pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv builds the wheel under test",
    )
    @pytest.mark.packaging
    @pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv builds the wheel under test",
    )
    def test_metadata_should_carry_the_version_the_repository_derives(self, tmp_path):
        """Test that the artifact is versioned by the tag history.

        Given:
            A wheel built from the repository, and the version the
            build hook derives from that same repository.
        When:
            The wheel's metadata version is compared with it.
        Then:
            It should be that version, so the hook the manifest names
            is the one that versioned the artifact rather than some
            default the backend fell back to.
        """

        # Act
        metadata = wheel_metadata(PROJECT_ROOT, tmp_path)

        # Assert
        expected = "Version: {}".format(derived_version(PROJECT_ROOT))
        assert expected in metadata.splitlines()

    def test_metadata_should_declare_the_license_once(self, tmp_path):
        """Test that the built metadata names the licence exactly once.

        Given:
            A wheel built from the repository.
        When:
            Its metadata is inspected.
        Then:
            It should carry a licence expression and no licence
            classifier, because PEP 639 directs tools not to combine the
            two and PyPI rejects a distribution that does.
        """

        # Act
        metadata = wheel_metadata(PROJECT_ROOT, tmp_path)

        # Assert
        assert "License-Expression: MIT" in metadata
        classifiers = [
            line
            for line in metadata.splitlines()
            if line.startswith("Classifier: License")
        ]
        assert classifiers == []

    @pytest.mark.packaging
    @pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv builds the source distribution under test",
    )
    def test_sdist_should_carry_the_test_tree(self, tmp_path):
        """Test that the suite ships inside a source distribution.

        Given:
            A source distribution built from the repository.
        When:
            Its members are inspected.
        Then:
            It should carry the test package including conftest, so the
            suite runs from an sdist exactly as it does from the
            repository.
        """

        # Act
        members = sdist_members(PROJECT_ROOT, tmp_path)

        # Assert
        assert "tests/__init__.py" in members
        assert "tests/conftest.py" in members
        assert "tests/builders.py" in members
        assert "src/quadrat/__init__.py" in members

    @pytest.mark.packaging
    @pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv builds the source distribution under test",
    )
    def test_sdist_should_exclude_notebook_checkpoints_at_any_depth(self, tmp_path):
        """Test that no notebook checkpoint ships, however deep it sits.

        Given:
            A staged tree holding exactly what the sdist declares it
            includes, with checkpoint directories planted one and two
            levels inside those roots. The copy carries no
            ``.gitignore``, whose own checkpoint entry would otherwise
            keep them out of the archive and leave this unable to fail.
        When:
            A source distribution is built from that tree.
        Then:
            It should ship none of them, so the exclusion rests on the
            build configuration the distribution declares rather than on
            an ignore file that is no part of it.
        """

        # Arrange
        staged = tmp_path / "project"
        staged.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy(PROJECT_ROOT / name, staged / name)
        for name in ("src", "tests", "build-hooks"):
            shutil.copytree(
                PROJECT_ROOT / name,
                staged / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        planted = (
            staged / "src" / ".ipynb_checkpoints",
            staged / "src" / "quadrat" / ".ipynb_checkpoints",
            staged / "tests" / "sub" / ".ipynb_checkpoints",
        )
        for directory in planted:
            directory.mkdir(parents=True)
            (directory / "notebook-checkpoint.py").write_text("x = 1\n")

        # Act
        members = sdist_members(staged, tmp_path / "dist")

        # Assert
        assert [name for name in members if ".ipynb_checkpoints" in name] == []
