#!/usr/bin/env python3
"""
Example custom evaluators for the Synthetic Evaluation Tool.

This file demonstrates how to create custom evaluators by implementing
the BaseEvaluator interface. You can use these as templates for your
own evaluators.
"""

import numpy as np
from typing import Dict, Any, List
import torch

# Import the base class
from xdeid3d.spherehead.evaluation.synthetic import BaseEvaluator


class GazeDirectionEvaluator(BaseEvaluator):
    """
    Evaluator that scores based on gaze direction.
    
    This example shows how to:
    - Use initialization parameters
    - Track state across evaluations
    - Implement complex scoring logic
    """
    
    def __init__(self, target_gaze_x=0.0, target_gaze_y=0.0, sensitivity=2.0, **kwargs):
        """
        Initialize evaluator with target gaze direction.
        
        Args:
            target_gaze_x: Target horizontal gaze direction (-1 to 1)
            target_gaze_y: Target vertical gaze direction (-1 to 1)
            sensitivity: How sensitive the scoring is to deviations
        """
        super().__init__(**kwargs)
        self.target_gaze_x = float(target_gaze_x)
        self.target_gaze_y = float(target_gaze_y)
        self.sensitivity = float(sensitivity)
        
        # Track statistics
        self.total_samples = 0
        self.gaze_history = []
    
    def initialize(self, generator_config: Dict[str, Any]) -> None:
        """Initialize with generator configuration."""
        print(f"GazeDirectionEvaluator initialized with:")
        print(f"  Target gaze: ({self.target_gaze_x:.2f}, {self.target_gaze_y:.2f})")
        print(f"  Sensitivity: {self.sensitivity}")
        print(f"  Generator config: {generator_config.get('cfg', 'unknown')}")
    
    def evaluate(self, sample_data: Dict[str, Any]) -> float:
        """
        Evaluate based on estimated gaze direction.
        
        In a real implementation, you would:
        1. Run gaze detection on the image
        2. Compare detected gaze with target
        3. Return a score based on similarity
        
        This is a mock implementation based on pose.
        """
        pose = sample_data['pose']
        yaw = pose['yaw']
        pitch = pose['pitch']
        
        # Mock gaze estimation based on pose
        # In reality, you'd use a gaze detection model here
        estimated_gaze_x = np.sin(yaw - np.pi/2) * 0.5
        estimated_gaze_y = np.cos(pitch - np.pi/2) * 0.3
        
        # Calculate distance from target
        gaze_distance = np.sqrt(
            (estimated_gaze_x - self.target_gaze_x)**2 + 
            (estimated_gaze_y - self.target_gaze_y)**2
        )
        
        # Convert distance to score
        score = np.exp(-self.sensitivity * gaze_distance)
        
        # Track statistics
        self.total_samples += 1
        self.gaze_history.append({
            'frame': sample_data['frame_idx'],
            'gaze_x': estimated_gaze_x,
            'gaze_y': estimated_gaze_y,
            'score': score
        })
        
        return float(score)
    
    def finalize(self, all_evaluations: List[Dict[str, Any]]) -> None:
        """Finalize and print statistics."""
        if self.gaze_history:
            avg_gaze_x = np.mean([g['gaze_x'] for g in self.gaze_history])
            avg_gaze_y = np.mean([g['gaze_y'] for g in self.gaze_history])
            avg_score = np.mean([g['score'] for g in self.gaze_history])
            
            print("\n" + "="*50)
            print("GAZE EVALUATION SUMMARY")
            print("="*50)
            print(f"Total samples evaluated: {self.total_samples}")
            print(f"Average gaze direction: ({avg_gaze_x:.3f}, {avg_gaze_y:.3f})")
            print(f"Target gaze direction: ({self.target_gaze_x:.3f}, {self.target_gaze_y:.3f})")
            print(f"Average score: {avg_score:.3f}")
            print("="*50)


class LandmarkQualityEvaluator(BaseEvaluator):
    """
    Evaluator that scores based on facial landmark quality.
    
    This example shows how to:
    - Use image data directly
    - Implement quality-based scoring
    - Handle missing features gracefully
    """
    
    def __init__(self, min_confidence=0.5, required_landmarks=None, **kwargs):
        """
        Initialize evaluator.
        
        Args:
            min_confidence: Minimum confidence threshold for landmarks
            required_landmarks: List of required landmark names
        """
        super().__init__(**kwargs)
        self.min_confidence = float(min_confidence)
        self.required_landmarks = required_landmarks or ['nose', 'left_eye', 'right_eye', 'mouth']
        
        # Track quality metrics
        self.quality_scores = []
        self.landmark_detections = []
    
    def evaluate(self, sample_data: Dict[str, Any]) -> float:
        """
        Evaluate based on facial landmark detection quality.
        
        In a real implementation, you would:
        1. Run facial landmark detection on the image
        2. Check confidence scores and landmark visibility
        3. Return quality score
        
        This is a mock implementation.
        """
        image = sample_data['image']
        pose = sample_data['pose']
        
        # Mock landmark detection based on pose
        # Frontal views should have better landmark visibility
        yaw_norm = pose['yaw'] % (2 * np.pi)
        frontal_score = np.cos(yaw_norm - np.pi/2) ** 2
        
        # Simulate landmark confidences
        landmark_confidences = {
            'nose': 0.9 * frontal_score + 0.1,
            'left_eye': 0.85 * frontal_score + 0.15,
            'right_eye': 0.85 * frontal_score + 0.15,
            'mouth': 0.8 * frontal_score + 0.2,
            'left_ear': 0.7 * (1 - frontal_score) + 0.1,
            'right_ear': 0.7 * (1 - frontal_score) + 0.1
        }
        
        # Calculate quality score
        quality_score = 0.0
        detected_count = 0
        
        for landmark in self.required_landmarks:
            if landmark in landmark_confidences:
                confidence = landmark_confidences[landmark]
                if confidence >= self.min_confidence:
                    quality_score += confidence
                    detected_count += 1
        
        # Normalize score
        if len(self.required_landmarks) > 0:
            quality_score /= len(self.required_landmarks)
        
        # Bonus for detecting all required landmarks
        if detected_count == len(self.required_landmarks):
            quality_score *= 1.1
        
        # Store metrics
        self.quality_scores.append(quality_score)
        self.landmark_detections.append({
            'frame': sample_data['frame_idx'],
            'confidences': landmark_confidences,
            'detected': detected_count,
            'score': quality_score
        })
        
        return np.clip(quality_score, 0, 1)
    
    def finalize(self, all_evaluations: List[Dict[str, Any]]) -> None:
        """Finalize and print quality statistics."""
        if self.quality_scores:
            print("\n" + "="*50)
            print("LANDMARK QUALITY EVALUATION SUMMARY")
            print("="*50)
            print(f"Total samples evaluated: {len(self.quality_scores)}")
            print(f"Average quality score: {np.mean(self.quality_scores):.3f}")
            print(f"Min quality score: {np.min(self.quality_scores):.3f}")
            print(f"Max quality score: {np.max(self.quality_scores):.3f}")
            
            # Count frames with all landmarks detected
            all_detected = sum(1 for d in self.landmark_detections 
                             if d['detected'] == len(self.required_landmarks))
            print(f"Frames with all landmarks: {all_detected}/{len(self.landmark_detections)} "
                  f"({100*all_detected/len(self.landmark_detections):.1f}%)")
            print("="*50)


class CompositeEvaluator(BaseEvaluator):
    """
    Example of a composite evaluator that combines multiple evaluation criteria.
    
    This shows how to:
    - Combine multiple evaluation methods
    - Weight different criteria
    - Create complex evaluation pipelines
    """
    
    def __init__(self, weights=None, **kwargs):
        """
        Initialize composite evaluator.
        
        Args:
            weights: Dictionary of weights for different criteria
        """
        super().__init__(**kwargs)
        self.weights = weights or {
            'pose': 0.3,
            'gaze': 0.3,
            'quality': 0.4
        }
        
        # Initialize sub-evaluators
        self.pose_eval = None  # Will use DefaultPoseEvaluator
        self.gaze_eval = GazeDirectionEvaluator(target_gaze_x=0, target_gaze_y=0)
        self.quality_eval = LandmarkQualityEvaluator(min_confidence=0.6)
        
        self.composite_scores = []
    
    def initialize(self, generator_config: Dict[str, Any]) -> None:
        """Initialize all sub-evaluators."""
        # Import here to avoid circular dependency
        from synth_eval.evaluate_synthetic_improved import DefaultPoseEvaluator
        self.pose_eval = DefaultPoseEvaluator()
        
        # Initialize sub-evaluators
        self.pose_eval.initialize(generator_config)
        self.gaze_eval.initialize(generator_config)
        self.quality_eval.initialize(generator_config)
        
        print(f"\nCompositeEvaluator initialized with weights:")
        for criterion, weight in self.weights.items():
            print(f"  {criterion}: {weight:.2f}")
    
    def evaluate(self, sample_data: Dict[str, Any]) -> float:
        """Evaluate using multiple criteria."""
        # Get individual scores
        pose_score = self.pose_eval.evaluate(sample_data)
        gaze_score = self.gaze_eval.evaluate(sample_data)
        quality_score = self.quality_eval.evaluate(sample_data)
        
        # Weighted combination
        composite_score = (
            self.weights['pose'] * pose_score +
            self.weights['gaze'] * gaze_score +
            self.weights['quality'] * quality_score
        )
        
        # Normalize by total weight
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            composite_score /= total_weight
        
        self.composite_scores.append({
            'frame': sample_data['frame_idx'],
            'pose': pose_score,
            'gaze': gaze_score,
            'quality': quality_score,
            'composite': composite_score
        })
        
        return composite_score
    
    def finalize(self, all_evaluations: List[Dict[str, Any]]) -> None:
        """Finalize all sub-evaluators and print combined statistics."""
        # Finalize sub-evaluators
        self.pose_eval.finalize(all_evaluations)
        self.gaze_eval.finalize(all_evaluations)
        self.quality_eval.finalize(all_evaluations)
        
        # Print composite statistics
        if self.composite_scores:
            print("\n" + "="*50)
            print("COMPOSITE EVALUATION SUMMARY")
            print("="*50)
            print(f"Total samples: {len(self.composite_scores)}")
            
            # Average scores for each component
            avg_pose = np.mean([s['pose'] for s in self.composite_scores])
            avg_gaze = np.mean([s['gaze'] for s in self.composite_scores])
            avg_quality = np.mean([s['quality'] for s in self.composite_scores])
            avg_composite = np.mean([s['composite'] for s in self.composite_scores])
            
            print(f"\nAverage scores:")
            print(f"  Pose:      {avg_pose:.3f} (weight: {self.weights['pose']:.2f})")
            print(f"  Gaze:      {avg_gaze:.3f} (weight: {self.weights['gaze']:.2f})")
            print(f"  Quality:   {avg_quality:.3f} (weight: {self.weights['quality']:.2f})")
            print(f"  Composite: {avg_composite:.3f}")
            print("="*50)


# Example usage:
# python synth_eval/evaluate_synthetic_improved.py --seeds=0-9 \
#     --evaluator=synth_eval/example_custom_evaluator.py \
#     --evaluator_args target_gaze_x=0.2 target_gaze_y=-0.1 sensitivity=3.0

# Or for composite evaluator with custom weights:
# python synth_eval/evaluate_synthetic_improved.py --seeds=0-9 \
#     --evaluator=synth_eval/example_custom_evaluator.py \
#     --evaluator_args weights.pose=0.5 weights.gaze=0.2 weights.quality=0.3