Visualization Guide
===================

X-DeID3D provides powerful visualization tools for analyzing evaluation results.

Spherical Heatmaps
------------------

Generate heatmaps showing metric values across viewing angles.

Creating Heatmaps
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import HeatmapGenerator

   # Create generator
   gen = HeatmapGenerator(resolution=72)
   gen.set_metric_name("identity_distance")

   # Add data points from evaluation
   for sample in results.samples:
       gen.add_score(sample.yaw, sample.pitch, sample.score)

   # Generate heatmap
   heatmap = gen.generate()

Projections
~~~~~~~~~~~

Different projection methods for visualization:

.. code-block:: python

   from xdeid3d.visualization import (
       create_2d_heatmap,
       create_polar_heatmap,
       create_mollweide_projection,
       save_figure,
   )

   # Rectangular (equirectangular)
   rect = create_2d_heatmap(heatmap, colormap="magma")
   save_figure(rect, "heatmap_rect.png")

   # Polar projection
   polar = create_polar_heatmap(heatmap, colormap="viridis")
   save_figure(polar, "heatmap_polar.png")

   # Mollweide projection
   mollweide = create_mollweide_projection(heatmap)
   save_figure(mollweide, "heatmap_mollweide.png")

Colormaps
~~~~~~~~~

Available colormaps:

- ``magma``, ``viridis``, ``plasma``, ``inferno`` (perceptually uniform)
- ``coolwarm``, ``RdBu`` (diverging)
- ``jet``, ``rainbow`` (classic)

Colored Meshes
--------------

Create 3D meshes with metric values mapped to vertex colors.

.. code-block:: python

   from xdeid3d.visualization import MeshExporter

   # Create exporter
   exporter = MeshExporter(
       colormap="magma",
       bandwidth=0.5,
   )

   # Add scores
   for sample in results.samples:
       exporter.add_score(sample.yaw, sample.pitch, sample.score)

   # Export to PLY
   mesh = exporter.export_ply(
       "output.ply",
       vertices,
       faces,
   )

The resulting PLY file can be viewed in:

- MeshLab
- Blender
- ParaView
- Any PLY-compatible viewer

Metric Plots
------------

Time Series
~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import create_metric_plot

   scores = results.get_metric_values("psnr")
   img = create_metric_plot(
       scores,
       title="PSNR Over Time",
       ylabel="PSNR (dB)",
   )
   save_figure(img, "psnr_timeseries.png")

Distributions
~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import create_distribution_plot

   img = create_distribution_plot(
       scores,
       title="PSNR Distribution",
       bins=30,
   )
   save_figure(img, "psnr_distribution.png")

Summary Figures
~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.visualization import create_summary_figure

   summary = {name: results.summary[name]["mean"] for name in results.metrics}
   img = create_summary_figure(summary, title="Evaluation Summary")
   save_figure(img, "summary.png")

Command Line
------------

Generate visualizations from the command line:

.. code-block:: bash

   # Heatmap from results
   xdeid3d generate heatmap results.json -c magma --projection polar

   # Colored mesh
   xdeid3d generate mesh results.json --mesh-file head.ply

   # All figures
   xdeid3d generate figures results.json -o figures/

   # HTML report
   xdeid3d generate report results.json
