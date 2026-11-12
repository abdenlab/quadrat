"""Reading a reference: its contig lengths.

Lengths come from the FASTA index when one exists and from a scan when
it does not, because the pileup needs them before it reads a single
alignment -- they are what bounds a position to its contig and what a
contig batch is built from.
"""

import collections
import os
import warnings

from quadrat._engine import (
    FASTA_TABLE,
    fspath_or_none,
    import_on_first_use,
    registered_fasta,
)

# Here rather than in `_engine`, which held it and never read it: every
# key is written and read by this module alone, and so was the reset
# below. A stringly-keyed global in a module that does not touch it is
# a cross-module coupling that buys nothing.
#
# `chrom_lengths` holds a ``(key, lengths)`` pair rather than the
# mapping alone, with the key `_fasta_key` derives. One slot rather than
# two, because the two are one fact: published in separate statements
# they could be written by two threads in opposite orders, leaving one
# genome's lengths filed under the other's key -- permanently, and
# passing the validity check for every later caller of that path.
#
# `lower_case` rather than SCREAMING, because the binding is constant
# and the referent is not: it is a live registry, which is the case
# §5's own Don't example names.
cache = {
    "chrom_lengths": None,
}


# public
def chrom_lengths(sequences):
    """Return an ordered ``chrom -> length`` mapping for a FASTA file.

    Read from the faidx index, which stores each contig's length directly,
    so no sequence is touched. The polars-bio table scan below computes the
    same thing with ``LENGTH(sequence)``, which means materializing the
    genome: measured on hg38 that is 4.8 s and a 10.2 GB peak, against
    0.00 s and 0.03 GB from the index, for a byte-identical mapping in the
    same order. It dominated the memory profile of an otherwise lazy
    pipeline.

    The scan is the fallback for a FASTA with no usable ``.fai`` beside
    it that cannot be indexed in place. pyfaidx is also the reference
    definition here, in the sense that bam2bw and tangermeme take their
    lengths the same way.

    The lengths of one reference at a time are kept, keyed on its path,
    modification time and size. A call for that reference while its
    file is unchanged is answered from them, with no index read and no
    scan. A call for any other reference, or for the same path after
    its file changed, reads the lengths afresh, by a scan where there
    is no usable index, and keeps them in place of the ones held. A
    mapping with no contig in it is answered and not kept, so a
    reference whose index was read while still being written is read
    again by the next call rather than answered as empty from then on.

    Two calls that scan at once in one process are unsupported;
    `quadrat._engine` says why.

    :param sequences: FASTA path, as a `str`, `bytes` or
        `os.PathLike`.
    :returns: An ordered ``chrom -> length`` mapping, copied, so a
        caller who drops an entry does not change which contigs a
        later pileup in this process counts.
    :raises TypeError: If ``sequences`` is neither. An in-memory
        sequence mapping is a planned capability rather than a
        supported one, and the plan is prose: the exception for an
        argument of the wrong type is `TypeError`, which is what
        `quadrat.require_complete` raises for the same mistake and
        what a caller guarding argument validation writes.
    """

    # Normalized through the one normalizer and refused here. The
    # package had five spellings of this probe and the conversion is
    # the part that does not differ between them; the message does,
    # and this one is about sequence dictionaries rather than about
    # paths, which is why it stays.
    path = fspath_or_none(sequences)
    if path is None:
        raise TypeError(
            "The GIQL engine reads sequences from a FASTA file; got {}. "
            "In-memory sequence dictionaries are not supported.".format(
                type(sequences).__name__
            )
        )
    sequences = path

    # One read of the slot, and one write of it. The pair is read into
    # a local so a concurrent write cannot change what this call
    # validated between the check and the use, and the key travels with
    # the value so the two cannot be made to disagree.
    key = _fasta_key(sequences)
    cached = cache["chrom_lengths"]
    if cached is not None and cached[0] == key:
        lengths = cached[1]
    else:
        lengths = _index_lengths(sequences)

        if lengths is None:
            warnings.warn(
                "{} has no usable FASTA index, so its contig lengths are "
                "being read by scanning every sequence. Upstream measured "
                "that at 4.8 s and a 10.2 GB peak on hg38, against 0.00 s "
                "and 0.03 GB from an index. The lengths are the same either "
                "way -- only the cost differs -- but the scan is in no term "
                "of the memory model the fan-out plans against, so under "
                "pileup_many(backend='process') every worker pays it and "
                "none of it was reserved. Run `samtools faidx` on the "
                "reference, or put it somewhere writable so pyfaidx can "
                "index it in place.".format(sequences),
                UserWarning,
                stacklevel=2,
            )
            with registered_fasta(sequences) as pb:
                frame = pb.sql(
                    "SELECT name, LENGTH(sequence) AS length FROM {}".format(FASTA_TABLE)
                ).collect()
            lengths = collections.OrderedDict(
                zip(frame["name"].to_list(), frame["length"].to_list())
            )

        # Only a mapping that names a contig; the docstring says why.
        if lengths:
            cache["chrom_lengths"] = (key, lengths)
    # A copy of what this call resolved, not of the global, and not the
    # cached mapping itself. Two reasons, and the first is the one that
    # bites: reading the slot back hands this caller whatever the slot
    # holds now, which under two threads on two references is the other
    # genome's lengths -- on the path where this call did the work. The
    # second is that `pileup_alignments` builds both the chrom-sizes
    # frame and the contig batches from this, so a caller who inspected
    # the lengths and dropped an entry would change which contigs a
    # later pileup in the same process counts -- and the short table
    # that resulted would still pass `require_complete`.
    return collections.OrderedDict(lengths)


def clear_cache():
    """Forget the cached chromosome lengths.

    The cache invalidates itself when the file at a path changes, so
    there is no staleness for this to cure; what it buys is the memory
    the mapping holds.

    No registration is released by it: `quadrat._engine.registered_fasta`
    drops its own on the way out of its block.
    """

    cache["chrom_lengths"] = None


def _fasta_key(sequences):
    """Return the key a cached read of this FASTA is filed under.

    `chrom_lengths` states the key. It is not the path alone because a
    caller who regenerates or replaces a reference at the same path
    within one process would otherwise be handed the previous genome's
    lengths -- which then decide which reads are kept and which contigs
    the batches are built from, so the stale answer is not merely stale
    but silently changes what a table contains.

    .. rubric:: Implementation notes

    A path that cannot be stat'd is keyed on the path alone. There is
    nothing better to key on, and the read that follows fails on the
    same path.
    """

    try:
        stat = os.stat(sequences)
    except OSError:
        return (sequences,)
    return (sequences, stat.st_mtime_ns, stat.st_size)


def _index_lengths(sequences):
    """Contig lengths from the FASTA index, or None if there is none usable.

    None means exactly "no index this process can read or build": the
    file or its index is absent or unreadable (`OSError`, which covers
    pyfaidx's `FastaNotFoundError` and `IndexNotFoundError`), the index
    cannot be built in place (`FastaIndexingError`), or pyfaidx is not
    installed. Those are the cases the scan is the documented fallback
    for.

    Everything else propagates, and the narrowness is the point. An
    `except Exception` spanning the import, the open, the reads *and*
    the close is the easier thing to write and buys a full-genome scan
    for a corrupt index, a pyfaidx version break and a failure to
    close a handle alike -- identically, and silently. The cost of
    narrowing it is that a
    pyfaidx failure outside the set above now ends the run instead of
    slowing it, which is the right way round for a fallback that is
    three orders of magnitude more expensive than the path it replaces.
    """

    try:
        pyfaidx = import_on_first_use("pyfaidx")
    except ImportError:
        return None
    try:
        fasta = pyfaidx.Fasta(sequences)
    except (OSError, pyfaidx.FastaIndexingError):
        return None
    try:
        return collections.OrderedDict(
            (chrom, len(fasta[chrom])) for chrom in fasta.keys()
        )
    finally:
        fasta.close()
