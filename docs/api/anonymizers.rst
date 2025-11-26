Anonymizers API
===============

The anonymizers module provides a protocol-based interface for face anonymization
methods. Users can implement custom anonymizers by following the :class:`AnonymizerProtocol`
or use built-in adapters.

Protocol
--------

.. autoclass:: xdeid3d.anonymizers.AnonymizerProtocol
   :members:
   :show-inheritance:

Base Classes
------------

.. autoclass:: xdeid3d.anonymizers.BaseAnonymizer
   :members:
   :show-inheritance:

Registry
--------

.. autofunction:: xdeid3d.anonymizers.register_anonymizer

.. autofunction:: xdeid3d.anonymizers.get_anonymizer

.. autofunction:: xdeid3d.anonymizers.list_anonymizers

Built-in Adapters
-----------------

InsightFace Adapter
~~~~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.anonymizers.InsightFaceAdapter
   :members:
   :show-inheritance:

Identity Adapter
~~~~~~~~~~~~~~~~

.. autoclass:: xdeid3d.anonymizers.IdentityAnonymizer
   :members:
   :show-inheritance:

Blur Adapter
~~~~~~~~~~~~

.. autoclass:: xdeid3d.anonymizers.BlurAnonymizer
   :members:
   :show-inheritance:

Creating Custom Anonymizers
---------------------------

To create a custom anonymizer, implement the :class:`AnonymizerProtocol`:

.. code-block:: python

   from xdeid3d.anonymizers import AnonymizerProtocol, register_anonymizer
   import numpy as np

   class MyAnonymizer:
       \"\"\"Custom anonymizer implementation.\"\"\"

       @property
       def name(self) -> str:
           return "my_anonymizer"

       def initialize(self, device: str = "cuda") -> None:
           # Load models, initialize resources
           pass

       def anonymize(
           self,
           image: np.ndarray,
           face_bbox: tuple = None,
           **kwargs,
       ) -> np.ndarray:
           # Apply anonymization
           return anonymized_image

   # Register the anonymizer
   register_anonymizer("my_anonymizer", MyAnonymizer)

Using Entry Points
~~~~~~~~~~~~~~~~~~

Anonymizers can also be registered via entry points in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."xdeid3d.anonymizers"]
   my_anonymizer = "my_package.anonymizer:MyAnonymizer"
