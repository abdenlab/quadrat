Reference
=========

The published surface, in the order a caller meets it: alignments become tracks, and a track is verified before it is read.

The pileup
----------

.. autofunction:: quadrat.pileup_alignments

.. autofunction:: quadrat.pileup_many

.. autoclass:: quadrat.EngineOptionWarning

The reference
-------------

.. autofunction:: quadrat.chrom_lengths

Alignments
----------

.. autofunction:: quadrat.bam_contigs

Tracks
------

.. autodata:: quadrat.TRACK_SUFFIX
   :no-value:

.. autofunction:: quadrat.track_contigs

.. autofunction:: quadrat.require_complete

The track format
----------------

What a writer other than this package must satisfy for its tables to be readable here, and what a reader may rely on.

A track is a **directory** -- it keeps the ``.track.parquet`` name, but it holds one Parquet file per contig and nothing else, because polars refuses to scan a directory of mixed extensions and so does anything else pointed at the dataset. The completeness record sits *beside* it, at ``quadrat.manifest_path``, for the same reason.

Each file carries three columns, and the dtypes are part of the contract rather than an artifact of how this package writes them:

=========  ===========  ===========================================
column     dtype        contents
=========  ===========  ===========================================
``chrom``  ``Utf8``     the contig, repeated on every row
``pos``    ``Int64``    0-based position, ascending within the file
``value``  ``Float64``  the count at that position, scaled
=========  ===========  ===========================================

One file per contig the table holds at least one row for, and none for a contig it does not -- so an absent contig and an empty one are the same thing here, which is what makes the completeness record necessary. The filename is what ``quadrat.track_chrom_path`` derives. Rows are sorted by ``pos`` within each file, which is what a windowed read depends on; the whole directory globbed back into one frame is therefore sorted by contig and then by position.

``chrom`` is kept inside each file although the filename already implies it. It costs nothing once dictionary-encoded, and it is what keeps a globbed scan of the whole directory filterable on ``chrom``.

The dtypes are deliberately not narrowed. ``pos`` as ``Int32`` and ``value`` as ``UInt16`` *double* the file on disk, because sorted ``Int64`` positions delta-encode far better than ``Int32`` ones; a reader that wants them narrow can narrow them in memory, where it is free.

``quadrat.read_manifest`` documents the record's fields.

.. autodata:: quadrat.MANIFEST_VERSION
   :no-value:

.. autofunction:: quadrat.manifest_path

.. autofunction:: quadrat.track_chrom_path

.. autofunction:: quadrat.read_manifest

The package
-----------

.. autodata:: quadrat.__version__
   :no-value:

Not documented here
-------------------

``quadrat.pileup`` also exports ``BACKENDS`` and ``plan_pileup``. Both are in that subpackage's ``__all__`` and carry no ``# public`` marker, which is deliberate: they are the subpackage's wildcard surface rather than the package's supported one, and this page documents the latter. ``BACKENDS`` is the legal domain of ``pileup_many``'s ``backend`` argument and ``plan_pileup`` answers what to run; the two constants that used to sit beside them were dropped, their effect being stated in the prose that needs it.
