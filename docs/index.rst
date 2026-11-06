quadrat
=======

Training data for sequence-to-function models, from alignments.

A sequence-to-function model consumes pairs of a one-hot DNA sequence over an input window and a measured signal profile over a shorter, concentric output window. Everything between a set of alignments and a batch of such pairs is independent of the architecture downstream, and that is what this package implements.

This build documents the published surface only -- the symbols marked ``# public`` in the source. Everything else is internal and may change without notice while the package is pre-1.0.

.. toctree::
   :maxdepth: 2

   reference
