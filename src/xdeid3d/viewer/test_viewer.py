#!/usr/bin/env python3
"""
Test script to verify the experiments viewer is working correctly
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.utils.experiment_reader import ExperimentReader

def test_experiment_reader():
    """Test the experiment reader functionality"""
    print("Testing Experiment Reader...")
    
    # Initialize reader
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
    reader = ExperimentReader(experiments_dir)
    
    # Test getting all experiments
    experiments = reader.get_all_experiments()
    print(f"✓ Found {len(experiments)} experiments")
    
    # Test getting specific experiment (if any exist)
    if experiments:
        first_exp = experiments[0]
        exp_info = reader.get_experiment_info(first_exp['cfg'], first_exp['seed'])
        print(f"✓ Successfully loaded experiment: {exp_info['cfg']}-{exp_info['seed']}")
        print(f"  Files found: {list(exp_info['files'].keys())}")
        print(f"  Metrics: {list(exp_info['metrics'].keys())}")
    
    # Test metrics summary
    summary = reader.get_metrics_summary()
    print(f"✓ Metrics summary computed for {len(summary)} metrics")
    
    print("\n✅ All tests passed!")
    return True

def test_flask_import():
    """Test that Flask can be imported"""
    print("\nTesting Flask import...")
    try:
        import flask
        import flask_cors
        print("✓ Flask and flask-cors imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Flask: {e}")
        print("  Please run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    print("SphereHead Experiments Viewer - Test Suite")
    print("=" * 50)
    
    # Run tests
    flask_ok = test_flask_import()
    reader_ok = test_experiment_reader()
    
    if flask_ok and reader_ok:
        print("\n🎉 All tests passed! The viewer is ready to use.")
        print("\nTo start the viewer, run:")
        print("  ./launch_viewer.sh")
        print("\nOr manually:")
        print("  python backend/app.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)