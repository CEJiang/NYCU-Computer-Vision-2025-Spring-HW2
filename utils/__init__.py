"""
__init__.py

Utility module initializer for the digit detection project.

This file exposes core utilities across the project, including:
- Visualization functions for loss curves and predictions
- Custom loss functions for bounding box regression (DIoU and CIoU)
- Memory cleanup for large-scale training

These functions are imported from:
- visualization.py
- memory.py
- losses.py
"""
from .visualization import plot_loss_accuracy, visualize_batch, visualize_sample, plot_val_map_curve
from .memory import clear_memory
from .losses import ciou_loss, diou_loss

__all__ = [
    "plot_loss_accuracy",
    'plot_val_map_curve',
    'visualize_batch',
    'visualize_sample',
    "clear_memory",
    "ciou_loss",
    "diou_loss"
]
