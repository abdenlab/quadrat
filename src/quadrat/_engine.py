"""The one polars-bio session, and the tables registered in it.

polars-bio holds a single process-global session, so every table name
here carries a ``quadrat_`` prefix and nothing registers a name a
caller might also be using. The prefix is not decoration: cherimoya
registers the same shapes under ``cherimoya_``, and during the
transition both packages may be imported into one process.

The same globalness is why a pileup worker runs one job at a time.
Two concurrent pileups would register their alignments under the same
names and read each other's reads.
"""

import os

_FASTA_TABLE = "quadrat_fasta"


_SIZES_TABLE = "quadrat_chrom_sizes"


_WINDOWS_TABLE = "quadrat_windows"


_CUTS_TABLE = "quadrat_cuts"


_LOCUS_BINS_TABLE = "quadrat_locus_bins"


_EXCLUSION_BINS_TABLE = "quadrat_exclusion_bins"


_cache = {
    "fasta": None,
    "chrom_lengths": None,
    "chrom_lengths_path": None,
    "sequences": {},
    "sql": {},
}


def clear_cache():
    """Forget cached chromosome lengths, sequences and transpiled queries."""

    _cache["fasta"] = None
    _cache["chrom_lengths"] = None
    _cache["chrom_lengths_path"] = None
    _cache["sequences"] = {}
    _cache["sql"] = {}


###
# Engine access
###


def _pb():
    """Import polars-bio lazily with quiet defaults."""

    os.environ.setdefault("TQDM_DISABLE", "1")
    import polars_bio as pb

    try:
        pb.set_loglevel("error")
    except Exception:
        pass
    return pb


def _giql_sql(key, query, tables):
    """Transpile a GIQL query for the datafusion-bio target once per process."""

    if key not in _cache["sql"]:
        from giql import transpile

        _cache["sql"][key] = transpile(query, tables=tables, dialect="datafusion-bio")
    return _cache["sql"][key]


def _interval_tables(*specs):
    from giql import Table

    return [
        Table(name, chrom_col="chrom", start_col=start, end_col=end, strand_col=None)
        for name, start, end in specs
    ]


def _register_frame(pb, name, frame):
    pb.from_polars(name, frame)


def _explain(pb, sql):
    """Return the physical plan of ``sql`` as one string."""

    import polars

    plan = pb.sql("EXPLAIN " + sql).collect()
    return (
        "\n".join(
            str(v)
            for col in plan.get_columns()
            for v in col.to_list()
            if isinstance(v, str)
        )
        if isinstance(plan, polars.DataFrame)
        else str(plan)
    )


def _sql_string(value):
    """One SQL string literal, with quotes doubled.

    Contig names come from a BAM header rather than from a user, but they
    are arbitrary bytes as far as the format is concerned, and a name
    carrying an apostrophe would otherwise end the literal early.
    """

    return "'{}'".format(str(value).replace("'", "''"))
