# quadrat

Sequence-to-function training data, straight from BAM.

A quadrat is the fixed-size frame a field ecologist drops on the ground to tally everything inside it, moved across a landscape by a sampling design chosen separately from the frame. This package is that: a fixed-length window, placed at loci by a sampling scheme, tallying the cut sites that fall inside.

Everything between a set of alignments and a batch of `(sequence, signal)` pairs is independent of the model that consumes them. `quadrat` implements that span — the pileup, the window cache, the sampler and the loader — so any sequence-to-function architecture can be trained without reimplementing it.

## Status

Under construction. The data path is being extracted from [cherimoya](https://github.com/conradbzura/cherimoya); see [#1](https://github.com/abdenlab/quadrat/issues/1) for the specification and the slices it lands in.

## Install

Nothing is published yet. Once the data path lands, the two installs will be:

```bash
pip install quadrat            # stream windows over cut-site tables
pip install "quadrat[pileup]"  # also build those tables from alignments
```

The `pileup` extra adds the engine that builds those tables: [polars-bio](https://github.com/biodatageeks/polars-bio) and [giql](https://github.com/abdenlab/giql) among its dependencies. The base install is enough for a caller who already has cut-site tables; `quadrat`'s package docstring states the import guarantee that makes it so.

The extra resolves with `uv` and not yet with pip, because one of its dependencies is pinned to a git revision that only `uv` reads — the comment on `[tool.uv.sources]` in `pyproject.toml` says why and when that changes.

## Development

```bash
uv sync --locked --all-extras                     # install, pileup engine included
uv run pytest -q                                  # run the suite
uv run ruff format --check src tests build-hooks   # check formatting
uv run ruff check src tests build-hooks            # lint
```

The version is the repository's rather than a literal in the tree: a `vMAJOR.MINOR.PATCH` tag names a release, and a commit between tags reports the last release plus the commit it is, so releasing is pushing a tag and nothing in the tree changes for one. A build from a tree that is not a checkout — an unpacked source distribution — reports the version that distribution was built with.

`uv sync --locked --no-dev` installs what `pip install quadrat` will, which is the base install the import guarantee above is about; CI runs it as a job of its own.

## License

MIT.
