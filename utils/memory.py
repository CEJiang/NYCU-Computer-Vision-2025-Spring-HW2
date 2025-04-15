"""memory.py - Utilities for clearing CUDA memory during training."""

import gc
import torch


def clear_memory():
    """
    Clears unused memory from GPU and triggers Python garbage collection.

    This helps reduce memory fragmentation and out-of-memory errors
    during training/validation cycles in PyTorch.
    """
    gc.collect()
    torch.cuda.empty_cache()
