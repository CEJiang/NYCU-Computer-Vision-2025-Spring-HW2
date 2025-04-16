import torch
import math


def iou_loss(box1, box2, eps=1e-7):
    """Standard IoU loss with squared term."""
    inter = (torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])).clamp(0) * \
            (torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])).clamp(0)
    
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    loss = 1 - iou ** 2
    return loss


def diou_loss(box1, box2, eps=1e-7):
    """Distance IoU Loss."""
    inter = (torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])).clamp(0) * \
            (torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])).clamp(0)
    
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    center_x1 = (box1[:, 0] + box1[:, 2]) / 2
    center_y1 = (box1[:, 1] + box1[:, 3]) / 2
    center_x2 = (box2[:, 0] + box2[:, 2]) / 2
    center_y2 = (box2[:, 1] + box2[:, 3]) / 2
    center_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    xc1 = torch.min(box1[:, 0], box2[:, 0])
    yc1 = torch.min(box1[:, 1], box2[:, 1])
    xc2 = torch.max(box1[:, 2], box2[:, 2])
    yc2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2 + eps

    loss = 1 - iou + center_dist / c2
    return loss


def ciou_loss(box1, box2, eps=1e-7):
    """Complete IoU Loss."""
    inter = (torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0])).clamp(0) * \
            (torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1])).clamp(0)
    
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    center_x1 = (box1[:, 0] + box1[:, 2]) / 2
    center_y1 = (box1[:, 1] + box1[:, 3]) / 2
    center_x2 = (box2[:, 0] + box2[:, 2]) / 2
    center_y2 = (box2[:, 1] + box2[:, 3]) / 2
    center_dist = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

    xc1 = torch.min(box1[:, 0], box2[:, 0])
    yc1 = torch.min(box1[:, 1], box2[:, 1])
    xc2 = torch.max(box1[:, 2], box2[:, 2])
    yc2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2 + eps

    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    loss = 1 - iou + center_dist / c2 + alpha * v
    return loss
