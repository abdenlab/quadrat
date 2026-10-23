"""Reading alignments: contigs, indexes, and how the genome is batched.

A contig predicate reaches the BAM index as a region query, so a batch
costs only the records it selects rather than a full scan. That is what
makes splitting the genome nearly free, and it is why an unindexed
alignment gets a warning rather than a silent fallback -- the rows are
identical either way and only the time changes.

The refusals here are the asymmetric half. An alignment whose header
names no contig the reference has produces an empty table rather than
an error, which is indistinguishable downstream from a genuinely empty
assay.
"""

import collections
import gzip
import os
import struct
import warnings

from quadrat._engine import _pb

# polars-bio's BAM provider returns ``start`` equal to pysam's 0-based
# ``reference_start`` and ``end`` equal to the exclusive ``reference_end`` on
# the default session, despite its docstring claiming 1-based coordinates.
# ``tests/test_giql_io.py`` pins this against pysam; bump the offset (and the
# test) if a future polars-bio release changes the convention.
_BAM_START_OFFSET = 0


###
# Cut-site pileup
###


def register_alignments(paths, prefix="quadrat_bam"):
    """Register BAM/SAM files as polars-bio tables and return their names."""

    pb = _pb()
    names = []
    for i, path in enumerate(paths):
        if not path.endswith((".bam", ".sam")):
            raise ValueError(
                "The GIQL engine converts BAM/SAM alignments; got {}".format(path)
            )

        name = "{}_{}".format(prefix, i)
        if path.endswith(".sam"):
            pb.register_sam(path, name)
        else:
            pb.register_bam(path, name)
        names.append(name)

    return names


def bam_contigs(path):
    """Reference sequences named in a BAM header, as ``{name: length}``.

    BGZF is gzip-framed, and the header is the first thing in the file,
    so this reads one block however large the alignment is -- 195
    contigs in about a millisecond on an ENCODE ATAC BAM.

    It is read directly rather than through pysam because pysam is a
    development dependency, not a runtime one, and polars-bio's
    ``describe_bam`` reports the column schema rather than the
    reference sequences.

    Returns ``None`` for anything that is not a readable BAM -- SAM,
    which is text and carries no index to restrict, but also a file
    that is missing or truncated. Nothing here diagnoses: the engine
    opens the same file moments later and reports it better, and a
    header that cannot be read only means the pileup is not split.
    """

    if not str(path).endswith(".bam"):
        return None

    try:
        with gzip.open(path, "rb") as handle:
            if handle.read(4) != b"BAM\x01":
                return None
            (length,) = struct.unpack("<i", handle.read(4))
            handle.read(length)
            (count,) = struct.unpack("<i", handle.read(4))

            contigs = collections.OrderedDict()
            for _ in range(count):
                (length,) = struct.unpack("<i", handle.read(4))
                name = handle.read(length)[:-1].decode("utf-8", "replace")
                (size,) = struct.unpack("<i", handle.read(4))
                contigs[name] = size
            return contigs
    except (OSError, EOFError, struct.error):
        # Unreadable, truncated, or not what it claims to be. This is
        # an optimisation, not a validation step: reporting the problem
        # belongs to the engine, which is about to open the same file
        # and already says something better about it. Declining here
        # just means the pileup is not split.
        return None


def _has_bam_index(path):
    """Whether a BAM has an index the engine can restrict a scan with.

    Both of htslib's conventions count -- ``x.bam.bai`` beside the file
    and ``x.bai`` in place of its extension -- though only the former
    appears in practice.

    A ``.csi`` deliberately does not count. polars-bio cannot open a
    CSI-indexed BAM at all, failing with ``invalid BAI header`` on a
    plain scan with no region predicate, so treating one as indexed
    would promise a pushdown that does not exist.
    """

    path = str(path)
    return os.path.exists(path + ".bai") or (
        path.endswith(".bam") and os.path.exists(path[: -len(".bam")] + ".bai")
    )


def _warn_unindexed_split(alignments, available, batches):
    """Say what splitting an alignment without an index will cost.

    A contig predicate reaches the BAM index as a region query, which
    is what makes a batch cost only the records it selects. Without an
    index the predicate is still correct -- it is an ordinary ``WHERE``
    clause, and the rows come back identical -- but it is no longer
    free: measured against a 2.4 GB hg38 ATAC BAM, one contig costs
    9.61 s where the indexed read costs 0.23 s, against 16.57 s for the
    whole file. So each batch is roughly 0.58 of a full scan, and four
    batches take about 2.3 times one ungrouped pass.

    This reports rather than clamps, because the batch count is only
    ever above one when something needs it to be -- :func:`~quadrat.pileup.plan.plan_pileup` returns the smallest count that reaches the best
    worker count, and never fewer than the per-job scan cap asks for.
    Falling back to a single batch would therefore trade a slow run for
    a dead one on exactly the machines that chose to split, and where
    the cap is what chose it the single batch is the arrangement that
    has been observed to die outright rather than merely crawl.

    A file whose header could not be read is left out: its branch is
    never restricted, so it pays nothing either way.
    """

    if len(batches) < 2:
        return

    missing = [
        alignment
        for alignment, known in zip(alignments, available.values())
        if known is not None and not _has_bam_index(alignment)
    ]
    if not missing:
        return

    warnings.warn(
        "{} has no .bai beside it, so each of the {} contig "
        "groups scans the whole file -- about {:.1f}x the time of one "
        "ungrouped pass. Memory is still bounded. Run `samtools index` "
        "to make the split free.".format(
            ", ".join(missing), len(batches), 0.58 * len(batches)
        ),
        UserWarning,
        stacklevel=3,
    )


def _require_reference_contigs(lengths, available, alignments, sequences):
    """Raise unless every readable alignment header names a reference contig.

    An alignment that shares no contig name with the reference
    contributes nothing: every record is dropped by the join against
    the chrom sizes, and the pileup writes a table with no rows in it
    that :func:`require_complete` then passes, because a manifest
    listing no files is a manifest whose files all exist. Training reads
    it as a genome with no signal.

    The usual cause is a naming convention rather than a wrong genome --
    an Ensembl-headered BAM calling its contigs ``1``, ``2``, ``MT``
    against a UCSC reference calling them ``chr1``, ``chr2``, ``chrM``.

    Checked per alignment rather than across the pool, because a pool
    mixing conventions is the quiet case: the other branches keep the
    batches non-empty, so the mismatched one simply counts nothing and
    the run looks clean.

    A header that could not be read is exempt. Diagnosing an unreadable
    file belongs to the engine, which opens it moments later and says
    something better about it, and a SAM input has no header to read
    here at all.
    """

    for alignment, known in zip(alignments, available.values()):
        if known is None or any(chrom in lengths for chrom in known):
            continue

        raise ValueError(
            "none of the {} contigs in {} appear in {} -- "
            "the alignment names {} and the reference names {}. A pileup "
            "would write an empty table. Use a reference whose contig "
            "names match the alignment, or re-header the "
            "alignment.".format(
                len(known),
                alignment,
                sequences,
                ", ".join(list(known)[:3]),
                ", ".join(list(lengths)[:3]),
            )
        )


def _contig_batches(lengths, available, groups):
    """Split contigs into ``groups`` batches of similar total length.

    ``lengths`` is the reference's ``{chrom: length}``; ``available`` is
    ``{table: {chrom: length}}`` for each alignment, from
    :func:`bam_contigs`, with ``None`` for a table whose contigs are
    unknown.

    Each batch is a ``{table: [chrom, ...]}`` mapping rather than one
    contig list, because the alignments pooled into a channel need not
    share a header. Taking the intersection across them would silently
    drop whatever only one of them has; restricting each branch
    separately keeps every record that an ungrouped pileup would have
    counted. A table whose contigs are unknown gets the whole batch and
    relies on the join to discard the rest, which is what an ungrouped
    pileup does anyway.

    Longest contig first into the emptiest batch -- enough to keep one
    batch from being most of the genome, which is all the balance that
    is needed given the reference is dominated by a few large contigs.
    """

    groups = max(1, int(groups))
    keep = [
        chrom
        for chrom in lengths
        if any(known is None or chrom in known for known in available.values())
    ]
    if not keep:
        return []

    groups = min(groups, len(keep))
    batches = [[] for _ in range(groups)]
    load = [0] * groups
    for chrom in sorted(keep, key=lambda c: -lengths[c]):
        index = load.index(min(load))
        batches[index].append(chrom)
        load[index] += lengths[chrom]

    return [
        {
            table: [chrom for chrom in batch if known is None or chrom in known]
            for table, known in available.items()
        }
        for batch in batches
    ]
