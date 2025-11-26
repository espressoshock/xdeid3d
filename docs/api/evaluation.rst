Evaluation API
==============

The evaluation module provides the pipeline for running comprehensive
anonymization evaluations across datasets and viewing angles.

Data Structures
---------------

.. autoclass:: xdeid3d.evaluation.Sample
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.EvaluationResult
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.MetricStats
   :members:
   :show-inheritance:

Evaluation Modes
----------------

.. autoclass:: xdeid3d.evaluation.EvaluationMode
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.SingleSampleMode
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.AggregateMode
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.TemporalMode
   :members:
   :show-inheritance:

Pipeline
--------

.. autoclass:: xdeid3d.evaluation.EvaluationPipeline
   :members:
   :show-inheritance:

Configuration
-------------

.. autoclass:: xdeid3d.evaluation.EvaluationConfig
   :members:
   :show-inheritance:

Callbacks
---------

.. autoclass:: xdeid3d.evaluation.EvaluationCallback
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.ProgressCallback
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.evaluation.LoggingCallback
   :members:
   :show-inheritance:

Example Usage
-------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline, EvaluationConfig

   # Create pipeline
   config = EvaluationConfig(
       metrics=["cosine_similarity", "psnr", "ssim"],
       device="cuda",
   )
   pipeline = EvaluationPipeline(config)

   # Run on directory
   results = pipeline.evaluate_directory(
       "data/original/",
       "data/anonymized/",
   )

   # Print summary
   print(results.summary)

Single Sample Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline
   from xdeid3d.utils.io import load_image

   pipeline = EvaluationPipeline()

   original = load_image("original.jpg")
   anonymized = load_image("anonymized.jpg")

   result = pipeline.evaluate_single(
       original,
       anonymized,
       yaw=0.5,
       pitch=1.2,
   )

Video Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.evaluation import EvaluationPipeline, TemporalMode

   pipeline = EvaluationPipeline()
   pipeline.set_mode(TemporalMode())

   results = pipeline.evaluate_video(
       "original.mp4",
       "anonymized.mp4",
   )

   # Temporal metrics are automatically included
   print(f"TIC: {results.summary['temporal_identity_consistency']}")
