Visualization API
=================

The visualization module provides tools for generating heatmaps, meshes,
and figures from evaluation results.

Heatmaps
--------

Heatmap Generator
~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.visualization.HeatmapGenerator
   :members:
   :show-inheritance:

Heatmap Data
~~~~~~~~~~~~

.. autoclass:: xdeid3d.visualization.SphericalHeatmap
   :members:
   :show-inheritance:

Projection Functions
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xdeid3d.visualization.create_2d_heatmap

.. autofunction:: xdeid3d.visualization.create_polar_heatmap

.. autofunction:: xdeid3d.visualization.create_mollweide_projection

Mesh Export
-----------

.. autoclass:: xdeid3d.visualization.MeshExporter
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.visualization.ColoredMesh
   :members:
   :show-inheritance:

PLY I/O
~~~~~~~

.. autofunction:: xdeid3d.visualization.read_ply

.. autofunction:: xdeid3d.visualization.write_ply

Figure Generation
-----------------

.. autofunction:: xdeid3d.visualization.create_metric_plot

.. autofunction:: xdeid3d.visualization.create_distribution_plot

.. autofunction:: xdeid3d.visualization.create_comparison_figure

.. autofunction:: xdeid3d.visualization.create_summary_figure

.. autofunction:: xdeid3d.visualization.save_figure

Example Usage
-------------

Generating Heatmaps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import HeatmapGenerator
   import numpy as np

   # Create generator
   generator = HeatmapGenerator(resolution=72)
   generator.set_metric_name("identity_distance")

   # Add data points
   for yaw, pitch, score in evaluation_data:
       generator.add_score(yaw, pitch, score)

   # Generate heatmap
   heatmap = generator.generate()

   # Different projections
   rgb_rect = heatmap.to_rgb(colormap="magma")
   polar = create_polar_heatmap(heatmap)
   mollweide = create_mollweide_projection(heatmap)

   # Save
   save_figure(rgb_rect, "heatmap_rect.png")

Creating Colored Meshes
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import MeshExporter

   # Create exporter
   exporter = MeshExporter(
       colormap="magma",
       bandwidth=0.5,  # Kernel bandwidth for interpolation
   )

   # Add evaluation scores
   for sample in results.samples:
       exporter.add_score(sample.yaw, sample.pitch, sample.score)

   # Export with vertex coloring
   mesh = exporter.export_ply(
       "output.ply",
       vertices,
       faces,
   )

Plotting Metrics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import (
       create_metric_plot,
       create_distribution_plot,
       save_figure,
   )
   import numpy as np

   # Time series
   scores = np.array([0.8, 0.75, 0.82, 0.79, 0.85])
   ts_img = create_metric_plot(
       scores,
       title="Identity Score Over Time",
       ylabel="Score",
   )
   save_figure(ts_img, "timeseries.png")

   # Distribution
   dist_img = create_distribution_plot(
       scores,
       title="Score Distribution",
       bins=20,
   )
   save_figure(dist_img, "distribution.png")
