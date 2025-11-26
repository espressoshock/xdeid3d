Metrics API
===========

The metrics module provides evaluation metrics for face anonymization,
organized by category: identity, quality, temporal, and explainability.

Protocol
--------

.. autoclass:: xdeid3d.metrics.MetricProtocol
   :members:
   :show-inheritance:

Base Classes
------------

.. autoclass:: xdeid3d.metrics.BaseMetric
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.metrics.PairwiseMetric
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.metrics.SequenceMetric
   :members:
   :show-inheritance:

Registry
--------

.. autofunction:: xdeid3d.metrics.register_metric

.. autofunction:: xdeid3d.metrics.get_metric

.. autofunction:: xdeid3d.metrics.list_metrics

Identity Metrics
----------------

Metrics for evaluating identity protection effectiveness.

Cosine Similarity
~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.CosineSimilarityMetric
   :members:
   :show-inheritance:

Euclidean Distance
~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.EuclideanDistanceMetric
   :members:
   :show-inheritance:

De-identification Rate
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.DeIdentificationRateMetric
   :members:
   :show-inheritance:

Quality Metrics
---------------

Metrics for evaluating visual quality preservation.

PSNR
~~~~

.. autoclass:: xdeid3d.metrics.PSNRMetric
   :members:
   :show-inheritance:

SSIM
~~~~

.. autoclass:: xdeid3d.metrics.SSIMMetric
   :members:
   :show-inheritance:

LPIPS
~~~~~

.. autoclass:: xdeid3d.metrics.LPIPSMetric
   :members:
   :show-inheritance:

FID
~~~

.. autoclass:: xdeid3d.metrics.FIDMetric
   :members:
   :show-inheritance:

Temporal Metrics
----------------

Metrics for evaluating video consistency.

Temporal Identity Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.TemporalIdentityConsistency
   :members:
   :show-inheritance:

Optical Flow Consistency
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.OpticalFlowConsistency
   :members:
   :show-inheritance:

Jitter Metric
~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.JitterMetric
   :members:
   :show-inheritance:

Explainability Metrics
----------------------

Metrics for 3D explainability analysis.

Angular Variance
~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.metrics.AngularVarianceMetric
   :members:
   :show-inheritance:

Creating Custom Metrics
-----------------------

Implement the :class:`MetricProtocol` or extend :class:`BaseMetric`:

.. code-block:: python

   from xdeid3d.metrics import BaseMetric, register_metric
   import numpy as np

   class MyMetric(BaseMetric):
       \"\"\"Custom evaluation metric.\"\"\"

       @property
       def name(self) -> str:
           return "my_metric"

       @property
       def higher_is_better(self) -> bool:
           return True  # or False for distance-like metrics

       def compute(
           self,
           original: np.ndarray,
           anonymized: np.ndarray,
           **kwargs,
       ) -> float:
           # Compute metric
           return score

   # Register
   register_metric("my_metric", MyMetric)
