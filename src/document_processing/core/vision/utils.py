"""
Vision Utilities Module
--------------------

Utility functions for vision processing modules, providing helper functions
for device management, image processing, and other common operations.

Key Features:
- Device optimization
- Common image operations
- Shared helper functions

Technical Components:
1. Device Management:
   - CUDA/MPS/CPU detection
   - Optimal device selection
   - Device capability checking

Author: Keith Satuku
Version: 1.0.0
License: MIT
"""

import torch


def get_optimal_device():
    """Get the optimal available device for computation.
    
    Returns:
        str: Device identifier ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu" 