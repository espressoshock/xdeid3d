Quickstart Guide
================

This guide will help you get started with X-DeID3D in minutes.

Basic Usage
-----------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

After installation, the ``xdeid3d`` command is available:

.. code-block:: bash

   # Show available commands
   xdeid3d --help

   # Show system info
   xdeid3d info

   # List available metrics
   xdeid3d utils list-metrics

   # List available anonymizers
   xdeid3d utils list-anonymizers

Evaluating Anonymization
------------------------

Single Image Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate a single image pair
   xdeid3d evaluate single original.jpg anonymized.jpg

   # With specific metrics
   xdeid3d evaluate single original.jpg anonymized.jpg \
       --metrics cosine_similarity,psnr,ssim

Directory Evaluation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate all images in directories
   xdeid3d evaluate run \
       --input data/original \
       --anonymized data/anonymized \
       --output results/

Video Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate video pairs
   xdeid3d evaluate video original.mp4 anonymized.mp4 \
       --output results/video_eval.json

Generating Visualizations
-------------------------

Heatmaps
~~~~~~~~

.. code-block:: bash

   # Generate heatmap from results
   xdeid3d generate heatmap results.json \
       --colormap magma \
       --projection polar

Colored Meshes
~~~~~~~~~~~~~~

.. code-block:: bash

   # Create colored 3D mesh
   xdeid3d generate mesh results.json \
       --mesh-file head.ply \
       --colormap viridis

Reports
~~~~~~~

.. code-block:: bash

   # Generate HTML report
   xdeid3d generate report results.json \
       --output report.html

Python API
----------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline
   from xdeid3d.utils.io import load_image

   # Create pipeline
   pipeline = EvaluationPipeline()

   # Load images
   original = load_image("original.jpg")
   anonymized = load_image("anonymized.jpg")

   # Evaluate
   result = pipeline.evaluate_single(original, anonymized)

   # Print results
   for metric, value in result.metrics.items():
       print(f"{metric}: {value:.4f}")

Custom Metrics
~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.metrics import get_metric

   # Get a specific metric
   psnr = get_metric("psnr")

   # Compute score
   score = psnr.compute(original, anonymized)
   print(f"PSNR: {score:.2f}")

Generating Heatmaps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import HeatmapGenerator, save_figure

   # Create generator
   gen = HeatmapGenerator(resolution=72)
   gen.set_metric_name("identity_distance")

   # Add evaluation data
   for yaw, pitch, score in evaluation_data:
       gen.add_score(yaw, pitch, score)

   # Generate and save
   heatmap = gen.generate()
   rgb = heatmap.to_rgb(colormap="magma")
   save_figure(rgb, "heatmap.png")

Web Interfaces
--------------

Experiments Viewer
~~~~~~~~~~~~~~~~~~

Browse and compare experiments:

.. code-block:: bash

   # Start viewer
   xdeid3d viewer -d ./experiments -p 5001

   # Open http://localhost:5001 in browser

Interactive GUI
~~~~~~~~~~~~~~~

Upload images and run evaluations interactively:

.. code-block:: bash

   # Start GUI
   xdeid3d gui -p 8000

   # Open http://localhost:8000 in browser

Configuration
-------------

Create a configuration file:

.. code-block:: bash

   # Generate default config
   xdeid3d init --format yaml -o config.yaml

Use configuration in evaluations:

.. code-block:: bash

   xdeid3d evaluate run \
       --config config.yaml \
       --input data/

Next Steps
----------

- Read the :doc:`user_guide/concepts` to understand the core concepts
- Explore the :doc:`api/metrics` for available evaluation metrics
- Learn about :doc:`user_guide/visualization` for advanced visualizations
- Check :doc:`api/anonymizers` for implementing custom anonymizers
