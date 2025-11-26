Evaluation Guide
================

This guide covers how to use X-DeID3D for evaluating face anonymization.

Evaluation Pipeline
-------------------

The ``EvaluationPipeline`` is the main entry point for running evaluations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline, EvaluationConfig

   # Create pipeline with configuration
   config = EvaluationConfig(
       metrics=["cosine_similarity", "psnr", "ssim"],
       device="cuda",
       batch_size=16,
   )
   pipeline = EvaluationPipeline(config)

   # Evaluate directory
   results = pipeline.evaluate_directory(
       original_dir="data/original/",
       anonymized_dir="data/anonymized/",
   )

   # Access results
   print(f"Total samples: {results.total_samples}")
   print(f"Summary: {results.summary}")

Single Sample Evaluation
------------------------

For evaluating individual image pairs:

.. code-block:: python

   from xdeid3d.utils.io import load_image

   original = load_image("original.jpg")
   anonymized = load_image("anonymized.jpg")

   result = pipeline.evaluate_single(
       original,
       anonymized,
       yaw=0.0,
       pitch=1.57,  # frontal view
   )

   for metric, score in result.metrics.items():
       print(f"{metric}: {score:.4f}")

Video Evaluation
----------------

For video files with temporal metrics:

.. code-block:: python

   from xdeid3d.evaluation import TemporalMode

   # Enable temporal mode
   pipeline.set_mode(TemporalMode())

   results = pipeline.evaluate_video(
       "original.mp4",
       "anonymized.mp4",
   )

   # Temporal metrics included automatically
   print(f"TIC: {results.summary['temporal_identity_consistency']}")

Metric Selection
----------------

Choosing metrics:

.. code-block:: python

   # Use preset groups
   config = EvaluationConfig(metrics="standard")  # PSNR, SSIM, LPIPS
   config = EvaluationConfig(metrics="identity")  # cosine, euclidean
   config = EvaluationConfig(metrics="all")       # all available

   # Or specify individually
   config = EvaluationConfig(
       metrics=["cosine_similarity", "psnr", "ssim", "lpips"]
   )

Callbacks
---------

Monitor evaluation progress:

.. code-block:: python

   from xdeid3d.evaluation import ProgressCallback

   callback = ProgressCallback()
   pipeline.add_callback(callback)

   results = pipeline.evaluate_directory(...)

Export Results
--------------

Save evaluation results:

.. code-block:: python

   # Save as JSON
   results.save("results.json")

   # Save as NPZ (for heatmap generation)
   results.save_npz("results.npz")

   # Export summary CSV
   results.to_csv("summary.csv")

Command Line Usage
------------------

.. code-block:: bash

   # Basic evaluation
   xdeid3d evaluate run -i data/original -a data/anonymized -o results/

   # With specific metrics
   xdeid3d evaluate run -i data/ --metrics cosine,psnr,ssim

   # Video evaluation
   xdeid3d evaluate video original.mp4 anonymized.mp4
