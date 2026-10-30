"""Synthetic genomics inputs for the test suite.

Not a ``test_`` module, so pytest does not collect it. Everything here
is a plain function taking an output path and the geometry to render at
it, so one builder serves every fixture that needs that shape of input
rather than each fixture carrying its own copy.

`pysam` is imported under a guard, so this module imports without it.
"""

import gzip
import os
import random

try:
    import pysam
except ImportError:
    # Left unbound rather than re-raised, because the suite's `pysam`
    # fixture owns the skip and nothing reaches a builder below without
    # requesting it.
    pysam = None


def write_fasta(
    path, chrom_sizes, *, seed=0, soft_mask=(), n_runs=(), index=False, line_width=60
):
    """Write a random FASTA over ``chrom_sizes`` and return its path.

    ``soft_mask`` and ``n_runs`` are iterables of ``(chrom, start,
    stop)`` half-open intervals, lowercased and replaced with ``N``
    respectively, so masked sequence is something a caller can ask for
    rather than something a real reference happens to carry. An
    interval running past either end of its contig is clipped to it,
    and every contig is written at exactly its declared length. Where
    the two overlap ``n_runs`` wins, so a position in both reads as an
    upper-case ``N`` rather than a lower-case base.

    ``index`` writes a ``.fai`` beside the file. It needs pysam, so it
    is off by default.

    .. rubric:: Implementation notes

    Both coordinates are clamped at both ends rather than left to
    slice truncation, which diverges from the stated contract at every
    negative one. A negative start truncates to an empty slice, so
    ``bases[-5:3]`` would mask nothing instead of masking from the
    contig's start; a negative stop is instead counted back from the
    end, so ``bases[0:-10]`` is a slice of almost the whole contig —
    which lowercases positions no interval named and, where the
    replacement is built from the clamped width rather than from the
    slice, deletes bases from a contig whose length is a guarantee.

    The masks are applied in the order the contract states rather than
    merged, so the precedence is the loop order and nothing has to
    reconcile an overlap. A caller wanting the other precedence has
    both inputs and can subtract.
    """

    rng = random.Random(seed)
    sequences = {}
    for chrom, size in chrom_sizes.items():
        sequences[chrom] = [rng.choice("ACGT") for _ in range(size)]

    # Both loops clip to the contig; see this function's docstring.
    for chrom, start, stop in soft_mask:
        bases = sequences[chrom]
        start = min(max(start, 0), len(bases))
        stop = min(max(stop, 0), len(bases))
        bases[start:stop] = [base.lower() for base in bases[start:stop]]
    for chrom, start, stop in n_runs:
        bases = sequences[chrom]
        start = min(max(start, 0), len(bases))
        stop = min(max(stop, 0), len(bases))
        bases[start:stop] = ["N"] * max(0, stop - start)

    with open(path, "w") as handle:
        for chrom, size in chrom_sizes.items():
            handle.write(">{}\n".format(chrom))
            bases = "".join(sequences[chrom])
            for offset in range(0, size, line_width):
                handle.write(bases[offset : offset + line_width] + "\n")

    if index:
        pysam.faidx(path)
    return path


def alignment_header(chrom_sizes):
    """Build a coordinate-sorted header over ``chrom_sizes``.

    A contig the reference does not declare is accepted without
    complaint, so a header-versus-reference mismatch is constructible.
    """

    return pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": chrom, "LN": size} for chrom, size in chrom_sizes.items()],
        }
    )


def aligned_read(header, name, chrom, pos, cigar="20M", flag=0, mapq=30):
    """Build one aligned segment with a dummy query and qualities.

    ``chrom=None`` builds an unplaced record: no reference id and no
    start, but the CIGAR it was given either way. Query length follows
    the CIGAR, so soft clips and deletions stay self-consistent, and
    falls back to 20 bases for a CIGAR that consumes no query at all.

    A ``chrom`` the header does not declare builds a half-placed
    record: no reference id, because the header resolves an unknown
    contig to one, while the start and the CIGAR are kept. That is how
    a header-versus-reference mismatch reaches a reader.
    """

    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.flag = flag
    if chrom is None:
        read.reference_id = -1
        read.reference_start = -1
    else:
        read.reference_id = header.get_tid(chrom)
        read.reference_start = pos
    read.cigarstring = cigar

    length = read.infer_query_length() or 20
    read.query_sequence = "A" * length
    read.query_qualities = pysam.qualitystring_to_array("I" * length)
    read.mapping_quality = mapq
    return read


def write_bam(path, reads, header):
    """Sort ``reads`` into ``path``, index it, and return the path.

    The file is coordinate-sorted and indexed on return whatever order
    the reads arrive in, and no intermediate survives beside it.

    .. rubric:: Implementation notes

    Written unsorted and sorted afterwards rather than sorted in
    memory, because the index is what the sort exists to make valid,
    and a BAM whose order disagrees with its index is a failure mode
    worth being unable to produce by accident.
    """

    staging = path + ".unsorted.bam"
    with pysam.AlignmentFile(staging, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.sort("-o", path, staging)
    pysam.index(path)
    os.remove(staging)
    return path


def write_random_bam(path, header, chrom_sizes, *, n_reads=40, seed=0, read_length=20):
    """Write a BAM of ``n_reads`` randomly placed records, half per strand.

    Every contig holds at least one read once ``n_reads`` reaches the
    contig count and, once ``n_reads`` exceeds it, the first contig
    holds more than any other, so a per-contig count is not the same
    number everywhere. A read may start anywhere a record of
    ``read_length`` still ends inside its contig, so no stretch of a
    contig is systematically uncovered.

    Raises `ValueError` for a contig shorter than ``read_length``, which
    cannot hold a read at all without running past the end the header
    declares.

    .. rubric:: Implementation notes

    The skew is deliberate: a uniform draw over contigs would satisfy
    neither the per-contig floor nor the unevenness, and a suite that
    splits work per contig needs both at once to be worth testing
    against.
    """

    rng = random.Random(seed)
    chroms = list(chrom_sizes)
    short = [chrom for chrom in chroms if chrom_sizes[chrom] < read_length]
    if short:
        raise ValueError(
            "contig shorter than the read length of {}: {}".format(
                read_length, ", ".join(short)
            )
        )

    reads = []
    for index in range(n_reads):
        if index < len(chroms):
            chrom = chroms[index]
        else:
            offset = index - len(chroms)
            chrom = chroms[0] if offset % 3 else chroms[(offset // 3) % len(chroms)]
        span = chrom_sizes[chrom] - read_length + 1
        reads.append(
            aligned_read(
                header,
                "r{}".format(index),
                chrom,
                rng.randrange(0, span),
                cigar="{}M".format(read_length),
                flag=16 if index % 2 else 0,
            )
        )
    return write_bam(path, reads, header)


def write_bed(path, rows):
    """Write ``(chrom, start, stop)`` rows, gzipped when ``path`` says so.

    Exclusion lists are conventionally gzipped and locus files
    conventionally are not, and both readers have to cope, so the
    suffix decides rather than a flag.
    """

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as handle:
        for row in rows:
            chrom, start, stop = row[:3]
            handle.write("{}\t{}\t{}\n".format(chrom, start, stop))
    return path


def intervals(chrom, start, stop, step, width):
    """Return ``(chrom, start, stop)`` rows every ``step`` bases over a range."""

    return [(chrom, pos, pos + width) for pos in range(start, stop, step)]
