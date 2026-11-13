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
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import quadrat
from quadrat import tables
from tests import PROJECT_ROOT, load_build_hook, run_child

# What the ``pileup`` extra installs, by import name, against what the
# base install carries but still declines to import at package import
# time. The package docstring states the guarantee both are checked
# against, and the split is the one `pyproject.toml` declares -- which
# `TestDistribution.test_metadata_should_declare_the_partition_it_imports_by`
# pins, so these two tuples cannot drift from it in silence.
EXTRA_MODULES = ("polars_bio", "giql")


DEFERRED_MODULES = ("polars", "pyfaidx")


PILEUP_MODULES = EXTRA_MODULES + DEFERRED_MODULES


# A public call that reaches each deferred module's first import, for a
# child interpreter in which that module will not import. Two calls
# rather than one, because no single entry point reaches all four:
# grouping needs polars, polars-bio and giql, and a sequence read is
# the one that opens a reference.
REACHES = {
    "polars": "tables.track_contigs(42)",
    "polars_bio": (
        "cache.group_track(polars.DataFrame("
        "{'chrom': ['chr1'], 'sig_start': [0], 'sig_end': [8]}), "
        "[polars.DataFrame({'chrom': ['chr1'], 'pos': [1], 'value': [1.0]})])"
    ),
    "pyfaidx": "stores.SequenceStore('absent.fa').window('chr1', 0, 4)",
}
REACHES["giql"] = REACHES["polars_bio"]


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


def declared_partition():
    """Return the installed distribution's requirements as ``(extra, base)`` import names.

    A requirement marked for any extra lands in the first set and every
    other one in the second, each spelled as the module it imports as.
    """

    extra = set()
    base = set()
    for requirement in importlib.metadata.requires("quadrat"):
        name = re.split(r"[\s<>=!;\[]", requirement, maxsplit=1)[0]
        (extra if "extra ==" in requirement else base).add(name.replace("-", "_"))
    return extra, base


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

    def test___doc___should_record_the_declined_signals_and_their_seam(self):
        """Test that the package says which signals it declines, and where.

        Given:
            The package docstring.
        When:
            It is read for the declined signals.
        Then:
            It should name splice junctions, contact maps and
            methylation, and the two functions whose channel split is
            methylation's reopening condition, each of which should
            still exist under the name it is given.
        """

        # Arrange
        text = " ".join(quadrat.__doc__.split())

        # Act
        seams = re.findall(r"``quadrat\.tables\.(\w+)``", text)

        # Assert
        for declined in ("Splice junctions", "contact maps", "Methylation"):
            assert declined in text
        assert seams == ["strand_slices", "track_paths"]
        assert all(callable(getattr(tables, name)) for name in seams)

    def test_metadata_should_declare_the_partition_it_imports_by(self):
        """Test which dependencies the distribution puts behind the extra.

        Given:
            The installed distribution's declared requirements.
        When:
            They are split by whether they are marked for the ``pileup``
            extra.
        Then:
            The engine -- polars-bio and giql -- should be behind it and
            polars and pyfaidx should not, because that line is what
            decides whether a caller is told to install an extra or
            told their install is incomplete.

        Read from the metadata rather than restated here, so the tuples
        above are checked against what `pyproject.toml` declares.
        """

        # Act
        extra, base = declared_partition()

        # Assert
        assert extra == set(EXTRA_MODULES)
        assert set(DEFERRED_MODULES) <= base

    @pytest.mark.packaging
    @pytest.mark.parametrize("name", PILEUP_MODULES)
    def test_deferred_import_should_name_the_extra_only_for_an_extra_module(self, name):
        """Test the remedy a deferred import reports a missing module with.

        Given:
            A fresh interpreter in which one deferred module will not
            import, and the extra the installed metadata declares.
        When:
            A public call that needs that module is made.
        Then:
            It should raise `ImportError` naming the module, and should
            name the ``pileup`` extra exactly when the metadata puts the
            module behind it -- a missing extra module is a minimal
            install the extra completes, and a missing base one is a
            broken install no extra repairs.

        The partition the package reports by and the one it declares
        are two lists, so this is what keeps the first from naming a
        module the second puts in the base install, or the reverse.
        """

        # Arrange
        for other in PILEUP_MODULES:
            if other != name and REACHES[name] == REACHES.get(other):
                pytest.importorskip(other)
        extra, _ = declared_partition()
        program = (
            "import sys\n"
            "sys.modules[{!r}] = None\n"
            "from quadrat import cache, stores, tables\n"
            "polars = None if {!r} == 'polars' else __import__('polars')\n"
            "try:\n"
            "    {}\n"
            "except ImportError as error:\n"
            "    print(error)\n".format(name, name, REACHES[name])
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.startswith(name + " is ")
        assert ("`pileup` extra" in result.stdout) is (name in extra)

    @pytest.mark.packaging
    @pytest.mark.parametrize("name", PILEUP_MODULES)
    def test_quadrat_should_import_none_of_the_pileup_engine(self, name):
        """Test what importing the package costs, one module at a time.

        Given:
            A fresh interpreter, and one module the package defers.
        When:
            The package is imported and the loaded modules inspected.
        Then:
            That module should not have been loaded, so a caller who
            already has tracks never pays for the engine that builds
            them.

        One case per module rather than one case for the set, because
        an absent module is indistinguishable from one the package
        declined to import and so has to be skipped. Checked as a set,
        a base install -- where polars and pyfaidx are installed and
        their deferral is the whole guarantee -- would skip the whole
        check on account of the two that are absent.
        """

        # Arrange
        pytest.importorskip(name)
        program = "import sys, quadrat\nprint({!r} in sys.modules)\n".format(name)

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize("name", PILEUP_MODULES + ("torch", "tangermeme", "tqdm"))
    def test_quadrat_cache_should_import_neither_the_engine_nor_a_tensor_library(
        self, name
    ):
        """Test what importing the window cache costs, one module at a time.

        Given:
            A fresh interpreter, and one module the window cache defers:
            the engine, torch, tangermeme or tqdm.
        When:
            ``quadrat.cache`` is imported and the loaded modules
            inspected.
        Then:
            That module should not have been loaded, so importing the
            module costs nothing that running it will.

        One case per module, for the reason the package-import test
        gives: an absent module has to be skipped, and a skip should
        cost only its own case.

        tqdm is the one that bites rather than merely costs. The
        package quiets the engine's progress bars by setting
        ``TQDM_DISABLE`` before importing polars-bio, and that works
        only while nothing has imported tqdm already -- so a
        module-level torch or tangermeme import here would put the bars
        back on every grouping run.
        """

        # Arrange
        pytest.importorskip(name)
        program = "import sys, quadrat.cache\nprint({!r} in sys.modules)\n".format(name)

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize(
        "name", PILEUP_MODULES + ("torch", "tangermeme", "tqdm", "pandas")
    )
    def test_quadrat_loci_should_import_nothing_it_defers(self, name):
        """Test what importing the locus module costs, one module at a time.

        Given:
            A fresh interpreter, and one module the locus module
            defers: the engine, a frame library, torch, tangermeme or
            tqdm.
        When:
            ``quadrat.loci`` is imported and the loaded modules
            inspected.
        Then:
            That module should not have been loaded.
        """

        # Arrange
        pytest.importorskip(name)
        program = "import sys, quadrat.loci\nprint({!r} in sys.modules)\n".format(name)

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize("name", ("torch", "tangermeme", "tqdm", "pandas"))
    def test_quadrat_should_import_no_tensor_or_frame_library(self, name):
        """Test that the package's own import stays clear of the stream's.

        Given:
            A fresh interpreter, and one library the sampler, the
            dataset and the loader import or reach: torch, tangermeme,
            tqdm or pandas.
        When:
            The package is imported and the loaded modules inspected.
        Then:
            That module should not have been loaded, so the one name
            the package exports from beside the stream costs none of
            what the stream imports.
        """

        # Arrange
        pytest.importorskip(name)
        program = "import sys, quadrat\nprint({!r} in sys.modules)\n".format(name)

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize(
        "name", PILEUP_MODULES + ("torch", "tangermeme", "tqdm", "pandas", "numpy")
    )
    def test_quadrat_groups_should_import_nothing_third_party(self, name):
        """Test what importing the signal-group module costs.

        Given:
            A fresh interpreter, and one third-party module: the
            engine, a frame library, torch, tangermeme, tqdm or numpy.
        When:
            ``quadrat.groups`` is imported and the loaded modules
            inspected.
        Then:
            That module should not have been loaded, so a model that
            imports the group check alone imports no data path with it.
        """

        # Arrange
        pytest.importorskip(name)
        program = "import sys, quadrat.groups\nprint({!r} in sys.modules)\n".format(name)

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize(
        "name", PILEUP_MODULES + ("torch", "tangermeme", "tqdm", "pandas")
    )
    def test_quadrat_pileup_aggregate_should_import_nothing_it_does_not_need(self, name):
        """Test what importing the aggregate record costs.

        Given:
            A fresh interpreter, and one module the record has no use
            for: the engine, a frame library, torch, tangermeme or tqdm.
        When:
            ``quadrat.pileup.aggregate`` is imported, and a record
            constructed and pickled.
        Then:
            That module should not have been loaded, so the record is
            data a base install builds and ships to a worker.
        """

        # Arrange
        pytest.importorskip(name)
        program = (
            "import pickle, sys\n"
            "from quadrat.pileup.aggregate import Aggregate\n"
            "pickle.dumps(Aggregate(position='span', unstranded=True))\n"
            "print({!r} in sys.modules)\n".format(name)
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.strip() == "False"

    @pytest.mark.packaging
    @pytest.mark.parametrize("module", ("sampling", "dataset", "loader"))
    def test_quadrat_sampling_dataset_and_loader_should_import_torch_and_nothing_deferred(
        self, module
    ):
        """Test what importing a module of the stream loads.

        Given:
            A fresh interpreter, and one of the sampler, dataset and
            loader modules.
        When:
            It is imported and the loaded modules inspected.
        Then:
            It should have loaded torch, which each of them subclasses
            or builds on at import, and none of the engine, polars,
            pyfaidx, tangermeme or pandas, which each reaches only when
            it reads.
        """

        # Arrange
        # tqdm is left out on purpose: torch imports it where it is
        # installed, so these modules load it without naming it.
        deferred = PILEUP_MODULES + ("tangermeme", "pandas")
        program = (
            "import sys, quadrat.{}\n"
            "print('torch' in sys.modules)\n"
            "print(sorted(set({!r}) & set(sys.modules)))\n".format(module, deferred)
        )

        # Act
        result = run_child(
            [sys.executable, "-c", program], "the child interpreter failed"
        )

        # Assert
        assert result.stdout.split("\n")[:2] == ["True", "[]"]

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

    @pytest.mark.packaging
    @pytest.mark.skipif(
        shutil.which("uv") is None,
        reason="uv builds the wheel under test",
    )
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
