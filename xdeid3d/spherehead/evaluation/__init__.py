# Copyright 2024 X-DeID3D Authors
# SPDX-License-Identifier: Apache-2.0
"""
SphereHead Evaluation Module.

Provides tools for evaluating 3D face synthesis and anonymization systems
using the SphereHead rendering engine.

Main Components:
    - BaseEvaluator: Abstract base class for custom evaluators
    - SyntheticEvaluator: Main synthetic evaluation tool
    - MeshEvaluator: 3D mesh-based evaluation with heatmaps
    - gpu_utils: GPU selection and memory management

Example:
    >>> from xdeid3d.spherehead.evaluation import BaseEvaluator, SyntheticEvaluator
    >>>
    >>> class MyEvaluator(BaseEvaluator):
    ...     def evaluate(self, sample_data):
    ...         return compute_score(sample_data['image'])
    >>>
    >>> evaluator = SyntheticEvaluator(
    ...     network_path='checkpoint.pkl',
    ...     custom_evaluator=MyEvaluator()
    ... )
    >>> results = evaluator.run()
"""

from xdeid3d.spherehead.evaluation.synthetic import (
    BaseEvaluator,
    create_samples,
    load_network_pkl_cpu_safe,
)

from xdeid3d.spherehead.evaluation.gpu_utils import (
    get_best_gpu_device,
    clear_gpu_memory,
)

__all__ = [
    # Base classes
    "BaseEvaluator",
    # Utility functions
    "create_samples",
    "load_network_pkl_cpu_safe",
    "get_best_gpu_device",
    "clear_gpu_memory",
]
