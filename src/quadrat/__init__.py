"""Training data for sequence-to-function models, from alignments.

A sequence-to-function model consumes pairs of a one-hot DNA sequence
over an input window and a measured signal profile over a shorter,
concentric output window. Everything between a set of alignments and a
batch of such pairs is independent of the architecture downstream, and
that is what this package implements.

The vocabulary used throughout:

``locus``
    A genomic coordinate, typically a peak summit or a sampled
    background position.

``window``
    A fixed-length interval centred on a locus, in two nested
    lengths, i.e., an input window supplying sequence and a shorter
    output window supplying signal.

``channel``
    One measured track.

``cut-site table``
    The intermediate representation: a per-contig, position-sorted
    Parquet table of ``(position, value)`` pairs for one channel,
    written once by the pileup and read many times by the loader.

``draw``
    One ``(locus, jitter offset, strand)`` triple the sampler emits, so
    a locus yields many draws under jitter and reverse-complement
    augmentation.

Nothing the ``pileup`` extra installs is imported at package import
time, so a caller who already has cut-site tables pays for none of it.
The engine behind that extra reads alignments through polars-bio and
transpiles its aggregate with giql, and loads what it needs when a
pileup runs.
"""

from importlib.metadata import version

from quadrat.pileup import pileup_alignments, pileup_many
from quadrat.reference import chrom_lengths
from quadrat.tables import CUT_SITE_SUFFIX, read_manifest, require_complete

__all__ = [
    "CUT_SITE_SUFFIX",
    "__version__",
    "chrom_lengths",
    "pileup_alignments",
    "pileup_many",
    "read_manifest",
    "require_complete",
]

# public
__version__ = version("quadrat")
