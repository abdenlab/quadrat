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

    Which records count and what each contributes (e.g., a 5' end
    for ATAC or DNase, a fragment midpoint for ChIP, a 3' end for 3'
    RNA-seq, or coverage over the span) is the caller's choice through
    an `Aggregate` rather than this package's. The format carries no
    assay, only what its values mean.

``draw``
    One ``(locus, jitter offset, strand)`` triple the sampler emits, so
    a locus yields many draws under jitter and reverse-complement
    augmentation.

``peak`` and ``negative``
    The two locus sets a window cache is built over, kept apart
    because they are sampled differently rather than because they are
    read differently: a peak is called, a negative is drawn from what
    the exclusion lists leave.

Nothing the ``pileup`` extra installs is imported at package import
time, so a caller who only reads tracks by coordinate pays for none of
it. The extra is the engine, which reads alignments through polars-bio
and transpiles interval joins with giql, and it is loaded on first use.
Anything that joins intervals through it needs it, e.g., building
tracks from alignments, grouping a track's rows by window, or joining
windows against exclusion lists; reading a track or a reference by
coordinate does not.

Three kinds of signal are declined rather than unbuilt. Splice
junctions and contact maps are not tracks: a junction is a ``(donor,
acceptor, count)`` relation and a contact map a pairwise matrix,
neither has a ``(chrom, pos, value)`` shape, and contact maps already
have a format of their own in ``cooler``. Methylation is declined for
want of a model that trains on it rather than for want of a way to
write it: its two counts per position do not fit one ``value``, but a
sum and a count would fit as two channels. Writing them is what the
channel split cannot do, since ``quadrat.tables.strand_slices`` and
``quadrat.tables.track_paths`` name exactly two channels, ``+`` and
``-``, or one when unstranded; widening that to a list of channels is
the change that would reopen it.
"""

__all__ = [
    "MANIFEST_VERSION",
    "Aggregate",
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
    "validate_signal_groups",
]

from importlib.metadata import version

from quadrat._engine import EngineOptionWarning
from quadrat.alignments import bam_contigs
from quadrat.groups import validate_signal_groups
from quadrat.pileup import Aggregate, pileup_alignments, pileup_many
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
