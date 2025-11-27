#!/usr/bin/env python3
"""
Test script to check the actual structure of 3d_shape_data.npy
"""

import numpy as np
import sys
import os

def analyze_shape_data(filepath):
    """Analyze the structure of 3d_shape_data.npy"""
    print(f"Analyzing: {filepath}")
    
    try:
        # Try loading with allow_pickle
        data = np.load(filepath, allow_pickle=True)
        
        print(f"Type: {type(data)}")
        print(f"Shape: {getattr(data, 'shape', 'N/A')}")
        print(f"Dtype: {getattr(data, 'dtype', 'N/A')}")
        
        # If it's an object array, try to access the item
        if hasattr(data, 'item'):
            try:
                item = data.item()
                print(f"Item type: {type(item)}")
                if isinstance(item, dict):
                    print(f"Dict keys: {list(item.keys())}")
                    for key, value in list(item.items())[:5]:  # Show first 5 items
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: ndarray with shape {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
            except:
                pass
        
        # If it's a structured array
        if hasattr(data, 'dtype') and data.dtype.names:
            print(f"Field names: {data.dtype.names}")
            
        # Show array info
        if isinstance(data, np.ndarray):
            print(f"Array info:")
            print(f"  Min: {np.min(data) if data.size > 0 else 'N/A'}")
            print(f"  Max: {np.max(data) if data.size > 0 else 'N/A'}")
            print(f"  Mean: {np.mean(data) if data.size > 0 else 'N/A'}")
            
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Test with a sample file
    test_file = "experiments/Head/1/3d_shape_data.npy"
    
    if os.path.exists(test_file):
        analyze_shape_data(test_file)
    else:
        print(f"File not found: {test_file}")
        print("Looking for other shape data files...")
        
        # Find any shape data files
        for root, dirs, files in os.walk("experiments"):
            for file in files:
                if file == "3d_shape_data.npy":
                    filepath = os.path.join(root, file)
                    print(f"\nFound: {filepath}")
                    analyze_shape_data(filepath)
                    break  # Just analyze one file
            else:
                continue
            break