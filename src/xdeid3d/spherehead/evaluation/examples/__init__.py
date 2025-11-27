# Copyright 2024 X-DeID3D Authors
# SPDX-License-Identifier: Apache-2.0
"""
Example evaluators for the SphereHead evaluation framework.

Provides template implementations showing how to create custom evaluators:
    - GazeDirectionEvaluator: Scores based on gaze direction
    - LandmarkQualityEvaluator: Scores based on facial landmark quality
    - CompositeEvaluator: Combines multiple evaluators
    - ExamplePoseEvaluator: Scores based on viewing angle

Use these as templates for implementing your own evaluators.
"""

from xdeid3d.spherehead.evaluation.examples.custom_evaluator import (
    GazeDirectionEvaluator,
    LandmarkQualityEvaluator,
    CompositeEvaluator,
)

from xdeid3d.spherehead.evaluation.examples.pose_evaluator import (
    ExamplePoseEvaluator,
)

__all__ = [
    "GazeDirectionEvaluator",
    "LandmarkQualityEvaluator",
    "CompositeEvaluator",
    "ExamplePoseEvaluator",
]
