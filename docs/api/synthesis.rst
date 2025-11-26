Synthesis API
=============

The synthesis module provides camera utilities, mesh extraction,
rendering, and image projection capabilities.

Camera Utilities
----------------

Camera Pose Samplers
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.synthesis.CameraPoseSampler
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.GaussianCameraSampler
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.UniformCameraSampler
   :members:
   :show-inheritance:

Camera Functions
~~~~~~~~~~~~~~~~

.. autofunction:: xdeid3d.synthesis.create_look_at_matrix

.. autofunction:: xdeid3d.synthesis.fov_to_intrinsics

.. autofunction:: xdeid3d.synthesis.create_camera_matrix

Mesh Extraction
---------------

.. autoclass:: xdeid3d.synthesis.MarchingCubesExtractor
   :members:
   :show-inheritance:

.. autofunction:: xdeid3d.synthesis.extract_mesh_from_volume

.. autofunction:: xdeid3d.synthesis.convert_sdf_to_mesh

Rendering
---------

.. autoclass:: xdeid3d.synthesis.BasicRenderer
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.RenderConfig
   :members:
   :show-inheritance:

.. autofunction:: xdeid3d.synthesis.render_mesh_to_image

Image Projection
----------------

Projector Protocol
~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.synthesis.ProjectorProtocol
   :members:
   :show-inheritance:

Projector Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.synthesis.BaseProjector
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.LatentProjector
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.PTIProjector
   :members:
   :show-inheritance:

Projector Factory
~~~~~~~~~~~~~~~~~

.. autofunction:: xdeid3d.synthesis.create_projector

Data Classes
~~~~~~~~~~~~

.. autoclass:: xdeid3d.synthesis.ProjectionResult
   :members:
   :show-inheritance:

.. autoclass:: xdeid3d.synthesis.ProjectionConfig
   :members:
   :show-inheritance:

Example Usage
-------------

Camera Pose Sampling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.synthesis import GaussianCameraSampler, UniformCameraSampler

   # Gaussian sampling around front view
   gaussian_sampler = GaussianCameraSampler(
       mean_yaw=0.0,
       mean_pitch=0.0,
       std_yaw=0.3,
       std_pitch=0.2,
   )
   poses = gaussian_sampler.sample(batch_size=10)

   # Uniform sampling
   uniform_sampler = UniformCameraSampler(
       yaw_range=(-1.0, 1.0),
       pitch_range=(-0.5, 0.5),
   )
   poses = uniform_sampler.sample(batch_size=10)

Mesh Extraction
~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.synthesis import MarchingCubesExtractor
   import numpy as np

   # Create volume (e.g., from neural field)
   volume = np.random.randn(64, 64, 64)

   # Extract mesh
   extractor = MarchingCubesExtractor(level=0.0)
   mesh = extractor.extract(volume)

   # Access results
   print(f"Vertices: {mesh.vertices.shape}")
   print(f"Faces: {mesh.faces.shape}")

Image Projection
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xdeid3d.synthesis import create_projector, ProjectionConfig

   # Load generator (e.g., StyleGAN)
   generator = load_generator("model.pkl")

   # Create projector
   projector = create_projector(
       generator,
       method="pti",  # or "latent"
       device="cuda",
   )

   # Configure projection
   config = ProjectionConfig(
       num_steps=1000,
       learning_rate=0.1,
   )

   # Project image
   result = projector.project(target_image, config)

   # Use results
   latent_code = result.latent
   reconstruction = result.reconstruction
