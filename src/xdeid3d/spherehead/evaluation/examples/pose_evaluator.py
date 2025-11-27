#!/usr/bin/env python3
"""
Example pose-based evaluator for the Synthetic Evaluation Tool.

This evaluator simulates neural network scores based on viewing angle, 
giving highest scores to frontal views and lower scores to profile/back views.
This is an example implementation that demonstrates how to create custom evaluators.
"""

import numpy as np
from typing import Dict, Any

# Import the base class
from xdeid3d.spherehead.evaluation.synthetic import BaseEvaluator


class ExamplePoseEvaluator(BaseEvaluator):
    """
    Example evaluator that simulates neural network scores based on viewing angle.
    Highest scores for frontal views, lower for profile/back views.
    
    This demonstrates how to create a custom evaluator by implementing the
    BaseEvaluator interface. The scoring is purely synthetic and is meant
    to simulate what a real neural network might output.
    """
    
    def __init__(self, **kwargs):
        """Initialize the evaluator."""
        super().__init__(**kwargs)
        self.total_samples = 0
        self.score_sum = 0.0
    
    def evaluate(self, sample_data: Dict[str, Any]) -> float:
        """
        Evaluate based on pose parameters.
        
        Args:
            sample_data: Dictionary containing:
                - 'pose': Camera pose parameters with 'yaw' and 'pitch'
                - Other fields (image, depth, etc.) are available but not used
        
        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        pose = sample_data['pose']
        yaw = pose['yaw']
        pitch = pose['pitch']
        
        # Normalize yaw to [0, 2*pi]
        yaw_norm = yaw % (2 * np.pi)
        
        # Front view is around pi/2, back view is around 3*pi/2
        # Create a score that's highest at front view
        front_score = np.cos(yaw_norm - np.pi/2) ** 2
        
        # Add some pitch variation
        pitch_factor = 0.8 + 0.2 * np.cos(2 * (pitch - np.pi/2))
        
        # Add deterministic "noise" based on angle to make it more interesting
        # but consistent across calls
        angle_variation = 0.05 * np.sin(5 * yaw_norm) * np.cos(3 * pitch)
        
        # Final score in [0, 1]
        score = np.clip(front_score * pitch_factor + angle_variation, 0, 1)
        
        # Track statistics
        self.total_samples += 1
        self.score_sum += score
        
        return score
    
    def finalize(self, all_evaluations: list) -> None:
        """Print summary statistics after all evaluations."""
        if self.total_samples > 0:
            avg_score = self.score_sum / self.total_samples
            print(f"\nExamplePoseEvaluator Summary:")
            print(f"  Total samples evaluated: {self.total_samples}")
            print(f"  Average score: {avg_score:.3f}")


# Usage example:
# python synth_eval/evaluate_synthetic.py --seeds=0-3 --evaluator=synth_eval/example_pose_evaluator.py