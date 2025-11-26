Contributing
============

We welcome contributions to X-DeID3D! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/xdeid3d.git
      cd xdeid3d

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # or `venv\Scripts\activate` on Windows

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Style
----------

We use the following tools for code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run all checks:

.. code-block:: bash

   # Format code
   black xdeid3d/ tests/

   # Lint
   ruff check xdeid3d/ tests/

   # Type check
   mypy xdeid3d/

Testing
-------

We use pytest for testing:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=xdeid3d

   # Run specific test file
   pytest tests/test_metrics.py

   # Run tests matching pattern
   pytest -k "test_psnr"

Adding New Features
-------------------

Custom Anonymizers
~~~~~~~~~~~~~~~~~~

1. Create your anonymizer class implementing ``AnonymizerProtocol``
2. Register it in ``xdeid3d/anonymizers/__init__.py``
3. Add tests in ``tests/test_anonymizers.py``
4. Document in ``docs/api/anonymizers.rst``

Custom Metrics
~~~~~~~~~~~~~~

1. Create your metric class extending ``BaseMetric``
2. Register it in ``xdeid3d/metrics/__init__.py``
3. Add tests in ``tests/test_metrics.py``
4. Document in ``docs/api/metrics.rst``

Pull Request Process
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/my-feature

2. Make your changes with clear commits
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

Use conventional commits:

- ``feat:`` New feature
- ``fix:`` Bug fix
- ``docs:`` Documentation changes
- ``test:`` Test additions/changes
- ``refactor:`` Code refactoring
- ``chore:`` Build/tooling changes

Example:

.. code-block::

   feat: Add new identity metric using ArcFace embeddings

   - Implement ArcFaceMetric class
   - Add embedding cache for efficiency
   - Include unit tests and documentation

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

   # View at docs/_build/html/index.html

Reporting Issues
----------------

When reporting issues, please include:

- Python version
- X-DeID3D version
- Operating system
- Minimal reproducible example
- Full error traceback

Code of Conduct
---------------

Please be respectful and constructive in all interactions.
We aim to maintain a welcoming and inclusive community.
