Changelog
=========

All notable changes to X-DeID3D are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[0.1.0] - 2024-XX-XX
--------------------

Initial release of X-DeID3D.

Added
~~~~~

Core Features
^^^^^^^^^^^^^

- Protocol-based anonymizer interface with registry
- Comprehensive metric system (identity, quality, temporal, explainability)
- Evaluation pipeline with multiple modes
- Spherical heatmap generation with kernel regression
- Colored mesh export to PLY format
- Camera pose sampling utilities
- Mesh extraction from volumetric data
- PTI projector for image-to-latent projection

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

- ``xdeid3d evaluate`` commands for running evaluations
- ``xdeid3d generate`` commands for visualizations
- ``xdeid3d benchmark`` commands for performance testing
- ``xdeid3d utils`` commands for data preparation
- ``xdeid3d viewer`` for experiments browser
- ``xdeid3d gui`` for interactive evaluation

Web Interfaces
^^^^^^^^^^^^^^

- Flask-based experiments viewer
- FastAPI-based interactive GUI

Documentation
^^^^^^^^^^^^^

- Sphinx documentation with API reference
- Installation and quickstart guides
- Contributing guidelines

Built-in Components
^^^^^^^^^^^^^^^^^^^

Anonymizers:

- InsightFace adapter (default)
- Identity (passthrough) anonymizer
- Blur anonymizer

Metrics:

- Cosine similarity
- Euclidean distance
- De-identification rate
- PSNR, SSIM, LPIPS, FID
- Temporal identity consistency
- Optical flow consistency
- Jitter metric
- Angular variance

[Unreleased]
------------

Planned features for future releases.

- Additional anonymizer adapters
- Extended metrics suite
- Improved GPU utilization
- Batch processing optimization
