Command Line Interface
======================

X-DeID3D provides a comprehensive CLI for all operations.

Global Options
--------------

.. code-block:: bash

   xdeid3d [OPTIONS] COMMAND [ARGS]...

Options:

- ``-v, --verbose``: Increase verbosity (use -vv for debug)
- ``-q, --quiet``: Suppress non-error output
- ``--device``: Device to use (cpu, cuda, cuda:0)
- ``--version``: Show version and exit
- ``--help``: Show help message

Commands Overview
-----------------

.. code-block:: bash

   xdeid3d evaluate    # Run evaluations
   xdeid3d generate    # Generate visualizations
   xdeid3d benchmark   # Run benchmarks
   xdeid3d utils       # Utility commands
   xdeid3d viewer      # Launch experiments viewer
   xdeid3d gui         # Launch interactive GUI
   xdeid3d config      # Manage configuration
   xdeid3d info        # Show system information

Evaluate Commands
-----------------

Run Evaluation
~~~~~~~~~~~~~~

.. code-block:: bash

   xdeid3d evaluate run [OPTIONS]

   Options:
     -i, --input         Input directory (original images)
     -a, --anonymized    Anonymized images directory
     -o, --output        Output file path
     -c, --config        Configuration file
     --metrics           Comma-separated metric names
     --mode              Evaluation mode (single, aggregate, temporal)
     --device            Device to use
     --batch-size        Batch size for processing
     --workers           Number of data loading workers

   Example:
     xdeid3d evaluate run -i data/orig -a data/anon -o results.json

Single Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xdeid3d evaluate single ORIGINAL ANONYMIZED [OPTIONS]

   Arguments:
     ORIGINAL      Path to original image
     ANONYMIZED    Path to anonymized image

   Options:
     --metrics     Metrics to compute
     --yaw         Yaw angle (radians)
     --pitch       Pitch angle (radians)

Video Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   xdeid3d evaluate video ORIGINAL ANONYMIZED [OPTIONS]

   Options:
     -o, --output       Output file path
     --sample-rate      Frame sampling rate

Generate Commands
-----------------

Heatmap
~~~~~~~

.. code-block:: bash

   xdeid3d generate heatmap INPUT_FILE [OPTIONS]

   Options:
     -o, --output       Output file path
     -c, --colormap     Colormap name
     -r, --resolution   Grid resolution
     --projection       Projection type (rectangular, polar, mollweide)
     --metric           Metric to visualize

Mesh
~~~~

.. code-block:: bash

   xdeid3d generate mesh INPUT_FILE [OPTIONS]

   Options:
     -o, --output       Output PLY file
     -c, --colormap     Colormap for coloring
     --bandwidth        Kernel bandwidth
     --mesh-file        Base mesh to color

Figures
~~~~~~~

.. code-block:: bash

   xdeid3d generate figures INPUT_FILE [OPTIONS]

   Options:
     -o, --output-dir   Output directory
     --format           Output format (png, pdf, svg)
     --dpi              Resolution

Project
~~~~~~~

.. code-block:: bash

   xdeid3d generate project IMAGE GENERATOR [OPTIONS]

   Options:
     -o, --output       Output directory
     --method           Projection method (latent, pti)
     --steps            Optimization steps
     --lr               Learning rate

Benchmark Commands
------------------

.. code-block:: bash

   xdeid3d benchmark anonymizer NAME [OPTIONS]
   xdeid3d benchmark metric NAME [OPTIONS]
   xdeid3d benchmark throughput [OPTIONS]
   xdeid3d benchmark compare [OPTIONS]

Utility Commands
----------------

.. code-block:: bash

   xdeid3d utils list-anonymizers    # List registered anonymizers
   xdeid3d utils list-metrics        # List registered metrics
   xdeid3d utils check-models        # Check/download models
   xdeid3d utils convert             # Convert file formats
   xdeid3d utils extract-frames      # Extract video frames
   xdeid3d utils clear-cache         # Clear model cache

Web Interfaces
--------------

Viewer
~~~~~~

.. code-block:: bash

   xdeid3d viewer [OPTIONS]

   Options:
     -d, --experiments-dir    Experiments directory
     -p, --port               Port number (default: 5001)
     --host                   Host to bind (default: 0.0.0.0)

GUI
~~~

.. code-block:: bash

   xdeid3d gui [OPTIONS]

   Options:
     -p, --port    Port number (default: 8000)
     --host        Host to bind (default: 0.0.0.0)

Configuration
-------------

.. code-block:: bash

   xdeid3d init [OPTIONS]           # Create config file
   xdeid3d config show              # Show current config
   xdeid3d config validate FILE     # Validate config file
