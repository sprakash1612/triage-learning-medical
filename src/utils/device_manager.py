"""
Device management utilities
"""

import torch


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get appropriate device (CUDA, MPS, or CPU)
    
    Args:
        gpu_id: GPU ID to use
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    return device