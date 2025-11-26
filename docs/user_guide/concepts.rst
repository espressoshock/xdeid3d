Core Concepts
=============

This section explains the core concepts behind X-DeID3D.

3D Explainability
-----------------

X-DeID3D evaluates face anonymization performance across viewing angles
on the unit sphere S². This allows us to:

- Identify angles where anonymization fails
- Visualize performance as 3D heatmaps
- Quantify robustness across viewpoints

Spherical Coordinate System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use spherical coordinates (yaw, pitch) where:

- **Yaw** (θ): Horizontal angle, typically [-π, π]
- **Pitch** (φ): Vertical angle, typically [0, π] where π/2 is frontal

Kernel Regression
~~~~~~~~~~~~~~~~~

To create smooth heatmaps from sparse evaluation samples, we use
Nadaraya-Watson kernel regression with a Gaussian RBF kernel:

.. math::

   \hat{f}(x) = \frac{\sum_i K_h(x, x_i) y_i}{\sum_i K_h(x, x_i)}

where :math:`K_h` is the Gaussian kernel with bandwidth :math:`h`.

Protocol-Based Design
---------------------

X-DeID3D uses Python's Protocol (PEP 544) for interfaces:

- **AnonymizerProtocol**: Interface for anonymization methods
- **MetricProtocol**: Interface for evaluation metrics

This allows duck-typing while maintaining type safety.

Registry Pattern
----------------

Components are registered in a central registry for discovery:

.. code-block:: python

   from xdeid3d.anonymizers import register_anonymizer

   @register_anonymizer("my_method")
   class MyAnonymizer:
       ...

Components can also be registered via entry points in ``pyproject.toml``.

Evaluation Modes
----------------

X-DeID3D supports different evaluation modes:

Single Sample Mode
~~~~~~~~~~~~~~~~~~

Evaluate individual image pairs with full metric computation.

Aggregate Mode
~~~~~~~~~~~~~~

Batch evaluation with statistical aggregation (mean, std, min, max).

Temporal Mode
~~~~~~~~~~~~~

Video evaluation with temporal consistency metrics.

Metrics Categories
------------------

Identity Metrics
~~~~~~~~~~~~~~~~

Measure how well the original identity is hidden:

- Cosine similarity between embeddings
- Euclidean distance in embedding space
- De-identification rate (threshold-based)

Quality Metrics
~~~~~~~~~~~~~~~

Measure visual quality preservation:

- PSNR (pixel-level)
- SSIM (structural)
- LPIPS (perceptual)
- FID (distribution)

Temporal Metrics
~~~~~~~~~~~~~~~~

Measure video consistency:

- Temporal Identity Consistency (TIC)
- Optical flow smoothness
- Jitter detection

Explainability Metrics
~~~~~~~~~~~~~~~~~~~~~~

Analyze performance variation:

- Angular variance
- Regional statistics
