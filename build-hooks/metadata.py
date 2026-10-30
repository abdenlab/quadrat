"""Supply the ``version`` that ``project.dynamic`` leaves to the build."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import _git
from hatchling.metadata.plugin.interface import MetadataHookInterface


class QuadratMetadataHook(MetadataHookInterface):
    """Resolve the version from the repository rather than from a literal.

    .. rubric:: Implementation notes

    The hook's directory is prepended to ``sys.path`` because hatchling
    executes this file directly rather than importing it as part of a
    package, so a sibling module is not otherwise reachable.
    """

    PLUGIN_NAME = "quadrat-metadata"

    def update(self, metadata):
        metadata["version"] = _git.version(self.root)
