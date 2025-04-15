"""
visualization.py

Utilities for visualizing training progress and detection results.
Includes:
- Loss and mAP curve plotting
- Single and batch image prediction visualizations
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import torchvision.transforms.functional as F

def plot_loss_accuracy(train_losses, val_losses,
                       train_accuracies=None, val_accuracies=None,
                       save_fig=True, output_path="training_curve.png"):
    """
    Plot and optionally save training/validation loss and accuracy curves.

    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        train_accuracies (list of float, optional): Training accuracy.
        val_accuracies (list of float, optional): Validation accuracy.
        save_fig (bool): Whether to save the figure.
        output_path (str): Path to save the output plot.
    """
    num_plots = 2 if train_accuracies and val_accuracies else 1
    _, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    axes[0].plot(train_losses, label="Train Loss", marker='o')
    axes[0].plot(val_losses, label="Validation Loss", marker='o')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid()

    if num_plots == 2:
        axes[1].plot(train_accuracies, label="Train Acc", marker='o')
        axes[1].plot(val_accuracies, label="Val Acc", marker='o')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].legend()
        axes[1].grid()

    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path)
        print(f"Training curve saved as {output_path}")
    plt.close()

def visualize_prediction(
    image,
    boxes,
    labels,
    scores=None,
    gt_boxes=None,
    gt_labels=None,
    label_map=None,
    threshold=0.3,
    show=True,
    save_path=None
):
    """
    Visualize a single image with predicted and ground truth bounding boxes.

    Args:
        image (Tensor): Image tensor [3, H, W], normalized.
        boxes (Tensor): Predicted bounding boxes (N, 4).
        labels (Tensor): Predicted class labels.
        scores (Tensor, optional): Confidence scores for predicted boxes.
        gt_boxes (Tensor, optional): Ground truth boxes (M, 4).
        gt_labels (Tensor, optional): Ground truth labels (M,).
        label_map (dict, optional): Mapping from label index to class name.
        threshold (float): Score threshold for displaying predicted boxes.
        show (bool): Whether to display the image.
        save_path (str, optional): File path to save the visualization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(-1, 1, 1)
    denorm_image = (image * std + mean).clamp(0, 1)
    pil_image = F.to_pil_image(denorm_image.cpu()).convert("RGB")

    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.imshow(pil_image)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue

        x_min, y_min, x_max, y_max = box.tolist()
        label = labels[i].item()
        score = scores[i].item() if scores is not None else None

        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        axis.add_patch(rect)

        label_str = label_map[label] if label_map and label in label_map else str(label)
        if score is not None:
            label_str += f" ({score:.2f})"

        axis.text(x_min, max(y_min - 5, 0), label_str,
                  bbox={"facecolor": 'yellow', "alpha": 0.5}, fontsize=10, color='black')

    if gt_boxes is not None and gt_labels is not None:
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            x_min, y_min, x_max, y_max = gt_box.tolist()
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            axis.add_patch(rect)
            gt_text = f"GT: {label_map[gt_label.item()] if label_map else gt_label.item()}"
            axis.text(x_min, y_max + 5, gt_text,
                      color='white', backgroundcolor='red', fontsize=10)

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved prediction image to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def visualize_batch(images, outputs, targets=None,
                    max_samples=5, label_map=None,
                    show=True, save_dir=None):
    """
    Visualize a batch of predictions and optionally ground truth annotations.

    Args:
        images (list of Tensor): Batch of image tensors.
        outputs (list of dict): List of model outputs per image.
        targets (list of dict, optional): List of ground truth annotations.
        max_samples (int): Maximum number of images to visualize.
        label_map (dict, optional): Label to class name mapping.
        show (bool): Whether to show the image.
        save_dir (str, optional): Directory to save output visualizations.
    """
    for i in range(min(max_samples, len(images))):
        image = images[i]
        output = outputs[i]
        boxes = output['boxes'].cpu()
        labels = output['labels'].cpu()
        scores = output['scores'].cpu()

        gt_boxes = targets[i]['boxes'].cpu() if targets else None
        gt_labels = targets[i]['labels'].cpu() if targets else None

        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"pred_{i}.png")

        visualize_prediction(
            image, boxes, labels, scores,
            gt_boxes=gt_boxes, gt_labels=gt_labels,
            label_map=label_map, show=show,
            save_path=save_path
        )

def visualize_sample(image, target, title=None, save_path=None, show=True):
    """
    Visualize a single training sample with ground truth boxes.

    Args:
        image (Tensor): Image tensor [3, H, W], normalized.
        target (dict): Ground truth dict with 'boxes' and 'labels'.
        title (str, optional): Title of the plot.
        save_path (str, optional): Path to save the image.
        show (bool): Whether to show the plot.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(-1, 1, 1)
    denorm_image = (image * std + mean).clamp(0, 1)

    img = F.to_pil_image(denorm_image.cpu()).convert("RGB")
    boxes = target["boxes"]
    labels = target["labels"]

    _, axis = plt.subplots(1, figsize=(6, 6), dpi=100)
    axis.imshow(img)

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        axis.add_patch(rect)
        axis.text(x_min, y_max + 5, str(label.item()),
                  color='black', backgroundcolor='lime', fontsize=10)

    if title:
        axis.set_title(title, fontsize=12)

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def plot_val_map_curve(val_maps, output_path="val_map_curve.png"):
    """
    Plot validation mAP@0.5:0.95 per epoch and save the result.

    Args:
        val_maps (list of float): List of mAP values per epoch.
        output_path (str): Path to save the curve plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_maps) + 1), val_maps,
             marker='o', label="val mAP@0.5:0.95")
    plt.title("Validation mAP@0.5:0.95 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Validation mAP curve saved to {output_path}")
