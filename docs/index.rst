X-DeID3D Documentation
=======================

**X-DeID3D** is a 3D explainability framework for face anonymization evaluation.
It provides tools for evaluating anonymization performance across viewing angles,
generating 3D visualizations, and benchmarking anonymization methods.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/concepts
   user_guide/evaluation
   user_guide/visualization
   user_guide/cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/anonymizers
   api/metrics
   api/evaluation
   api/visualization
   api/synthesis
   api/config

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Features
--------

- **Protocol-based design**: Extensible interfaces for anonymizers and metrics
- **Spherical evaluation**: Evaluate performance across viewing angles on S²
- **3D visualizations**: Generate heatmap meshes and spherical projections
- **Multiple metrics**: Identity, quality, temporal, and explainability metrics
- **CLI tools**: Comprehensive command-line interface for all operations
- **Web interfaces**: Experiments viewer and interactive GUI


Quick Example
-------------

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline
   from xdeid3d.metrics import get_metric
   from xdeid3d.visualization import HeatmapGenerator

   # Run evaluation
   pipeline = EvaluationPipeline()
   results = pipeline.evaluate_directory("data/")

   # Generate heatmap
   generator = HeatmapGenerator(resolution=72)
   for sample in results.samples:
       generator.add_score(sample.yaw, sample.pitch, sample.score)

   heatmap = generator.generate()
   heatmap.save("output/heatmap.png")


Installation
------------

.. code-block:: bash

   pip install xdeid3d

   # With optional dependencies
   pip install xdeid3d[full]


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
