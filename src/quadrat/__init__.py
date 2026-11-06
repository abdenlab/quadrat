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

``track``
    The intermediate representation: a per-contig, position-sorted
    Parquet table of ``(position, value)`` pairs for one channel,
    written once by the pileup and read many times by the loader.

    One position per read rather than coverage over the read's span,
    so *which* position -- a 5' end for ATAC or DNase, a fragment
    midpoint for ChIP -- is the caller's choice through ``pos_shift``
    and ``neg_shift`` rather than this package's. The format carries
    no assay.

``draw``
    One ``(locus, jitter offset, strand)`` triple the sampler emits, so
    a locus yields many draws under jitter and reverse-complement
    augmentation.

Nothing the ``pileup`` extra installs is imported at package import
time, so a caller who already has tracks pays for none of it.
The engine behind that extra reads alignments through polars-bio and
transpiles its aggregate with giql, and loads what it needs when a
pileup runs.
"""

__all__ = [
    "MANIFEST_VERSION",
    "EngineOptionWarning",
    "TRACK_SUFFIX",
    "__version__",
    "bam_contigs",
    "chrom_lengths",
    "manifest_path",
    "pileup_alignments",
    "pileup_many",
    "read_manifest",
    "require_complete",
    "track_chrom_path",
    "track_contigs",
]

from importlib.metadata import version

from quadrat._engine import EngineOptionWarning
from quadrat.alignments import bam_contigs
from quadrat.pileup import pileup_alignments, pileup_many
from quadrat.reference import chrom_lengths
from quadrat.tables import (
    MANIFEST_VERSION,
    TRACK_SUFFIX,
    manifest_path,
    read_manifest,
    require_complete,
    track_chrom_path,
    track_contigs,
)

# public
__version__ = version("quadrat")
