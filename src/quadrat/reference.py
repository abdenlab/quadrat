"""Reading a reference: contig lengths and sequence by coordinate.

Lengths come from the FASTA index when one exists and from a scan when
it does not, because the pileup needs them before it reads a single
alignment -- they are what bounds a cut site to its contig and what a
contig batch is built from.
"""

import collections

from quadrat._engine import _FASTA_TABLE, _cache, _pb

###
# Sequences
###


def _register_fasta(sequences):
    """Register ``sequences`` (a FASTA path) once, dropping stale caches."""

    if not isinstance(sequences, str):
        raise NotImplementedError(
            "The GIQL engine reads sequences from a "
            "FASTA file; in-memory sequence dictionaries are not supported."
        )

    pb = _pb()
    if _cache["fasta"] != sequences:
        pb.register_fasta(sequences, _FASTA_TABLE)
        _cache["fasta"] = sequences
        _cache["chrom_lengths"] = None
        _cache["sequences"] = {}
    return pb


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

    The scan remains the fallback for a FASTA with no ``.fai`` beside it
    that cannot be indexed in place. pyfaidx is also the reference
    definition here, in the sense that bam2bw and tangermeme take their
    lengths the same way.
    """

    if not isinstance(sequences, str):
        raise NotImplementedError(
            "The GIQL engine reads sequences from a "
            "FASTA file; in-memory sequence dictionaries are not supported."
        )

    if _cache.get("chrom_lengths_path") != sequences:
        _cache["chrom_lengths"] = None

    if _cache["chrom_lengths"] is None:
        lengths = None
        try:
            import pyfaidx

            fasta = pyfaidx.Fasta(sequences)
            try:
                lengths = collections.OrderedDict(
                    (chrom, len(fasta[chrom])) for chrom in fasta.keys()
                )
            finally:
                fasta.close()
        except Exception:
            lengths = None

        if lengths is None:
            pb = _register_fasta(sequences)
            frame = pb.sql(
                "SELECT name, LENGTH(sequence) AS length FROM {}".format(_FASTA_TABLE)
            ).collect()
            lengths = collections.OrderedDict(
                zip(frame["name"].to_list(), frame["length"].to_list())
            )

        _cache["chrom_lengths"] = lengths
        _cache["chrom_lengths_path"] = sequences
    return _cache["chrom_lengths"]


def _prefetch_sequences(sequences, chroms):
    """Load every chromosome in ``chroms`` with a single FASTA scan."""

    pb = _register_fasta(sequences)
    missing = [str(c) for c in chroms if str(c) not in _cache["sequences"]]
    if not missing:
        return

    frame = pb.sql(
        "SELECT name, sequence FROM {} WHERE name IN ({})".format(
            _FASTA_TABLE, ", ".join("'{}'".format(c) for c in missing)
        )
    ).collect()
    for name, sequence in zip(frame["name"].to_list(), frame["sequence"].to_list()):
        _cache["sequences"][name] = sequence


def _chrom_sequence(sequences, chrom):
    """Return the residues of one chromosome as a Python string (cached)."""

    chrom = str(chrom)
    if chrom not in _cache["sequences"]:
        _prefetch_sequences(sequences, [chrom])
    if chrom not in _cache["sequences"]:
        raise KeyError(chrom)
    return _cache["sequences"][chrom]
