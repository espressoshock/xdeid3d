Configuration API
=================

The config module provides Pydantic-based configuration schemas
for all X-DeID3D components.

Main Configuration
------------------

.. autoclass:: xdeid3d.config.XDeID3DConfig
   :members:
   :show-inheritance:

Section Configurations
----------------------

Evaluation Config
~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.config.EvaluationConfig
   :members:
   :show-inheritance:

Anonymizer Config
~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.config.AnonymizerConfig
   :members:
   :show-inheritance:

Visualization Config
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.config.VisualizationConfig
   :members:
   :show-inheritance:

Heatmap Config
~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.config.HeatmapConfig
   :members:
   :show-inheritance:

Loading Configuration
---------------------

.. code-block:: python

   from xdeid3d.config import XDeID3DConfig

   # Load from file
   config = XDeID3DConfig.from_file("config.yaml")

   # Or from dictionary
   config = XDeID3DConfig(**config_dict)

   # Access sections
   print(config.evaluation.metrics)
   print(config.visualization.colormap)

Configuration File Format
-------------------------

YAML Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # xdeid3d_config.yaml

   evaluation:
     metrics:
       - cosine_similarity
       - psnr
       - ssim
     device: cuda
     batch_size: 16
     num_workers: 4

   anonymizer:
     name: insightface
     model_path: null  # Auto-download
     swap_model: inswapper_128

   visualization:
     colormap: magma
     dpi: 150
     format: png

   heatmap:
     resolution: 72
     bandwidth: 0.5
     projection: rectangular

JSON Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "evaluation": {
       "metrics": ["cosine_similarity", "psnr"],
       "device": "cuda"
     },
     "visualization": {
       "colormap": "viridis",
       "dpi": 300
     }
   }

Environment Variables
---------------------

Configuration can also be set via environment variables:

.. code-block:: bash

   export XDEID3D_DEVICE=cuda
   export XDEID3D_CACHE_DIR=~/.cache/xdeid3d
   export XDEID3D_LOG_LEVEL=INFO

Validation
----------

All configurations are validated using Pydantic:

.. code-block:: python

   from xdeid3d.config import XDeID3DConfig
   from pydantic import ValidationError

   try:
       config = XDeID3DConfig(
           evaluation={"metrics": ["invalid_metric"]},
       )
   except ValidationError as e:
       print(f"Invalid configuration: {e}")
