Installation
============

Requirements
------------

X-DeID3D requires Python 3.8 or later.

Core Requirements
~~~~~~~~~~~~~~~~~

- Python >= 3.8
- NumPy >= 1.20
- Pydantic >= 2.0
- Click >= 8.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For full functionality:

- PyTorch >= 1.11 (GPU acceleration)
- OpenCV (image/video processing)
- Pillow (image I/O)
- InsightFace (face detection/recognition)
- LPIPS (perceptual metrics)
- Trimesh (mesh processing)
- Matplotlib (figure generation)
- Flask (experiments viewer)
- FastAPI + uvicorn (interactive GUI)

Installation Methods
--------------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install xdeid3d

   # With visualization support
   pip install xdeid3d[viz]

   # With full functionality
   pip install xdeid3d[full]

   # Development installation
   pip install xdeid3d[dev]

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-org/xdeid3d.git
   cd xdeid3d

   # Install in editable mode
   pip install -e .

   # With development dependencies
   pip install -e ".[dev]"

Optional Extras
---------------

The following extras are available:

- ``viz``: Visualization dependencies (matplotlib, trimesh)
- ``metrics``: Full metrics support (lpips, insightface)
- ``web``: Web interfaces (flask, fastapi, uvicorn)
- ``full``: All optional dependencies
- ``dev``: Development tools (pytest, black, mypy)

.. code-block:: bash

   pip install xdeid3d[viz,metrics]

GPU Support
-----------

For GPU acceleration, ensure you have CUDA installed and PyTorch with CUDA support:

.. code-block:: bash

   # Install PyTorch with CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Model Downloads
---------------

Some features require downloading pre-trained models. Models are automatically
downloaded on first use, or you can pre-download them:

.. code-block:: bash

   # Download all required models
   xdeid3d utils check-models --download

   # Check model status
   xdeid3d utils check-models

Models are cached in ``~/.xdeid3d/models/`` by default. You can change this
with the ``XDEID3D_CACHE_DIR`` environment variable.

Verifying Installation
----------------------

.. code-block:: bash

   # Check version and dependencies
   xdeid3d info

   # Run basic test
   python -c "import xdeid3d; print(xdeid3d.__version__)"

Troubleshooting
---------------

ImportError: No module named 'torch'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch is an optional dependency. Install it for GPU support:

.. code-block:: bash

   pip install torch torchvision

InsightFace initialization fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

InsightFace requires model files. Ensure they are downloaded:

.. code-block:: bash

   xdeid3d utils check-models --download

CUDA out of memory
~~~~~~~~~~~~~~~~~~

Reduce batch size or use CPU:

.. code-block:: bash

   xdeid3d evaluate run --device cpu --batch-size 1 ...
