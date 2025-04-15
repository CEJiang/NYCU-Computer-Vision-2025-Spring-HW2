"""
losses.py

Implements bounding box regression losses for object detection:
- IoU Loss
- DIoU Loss
- CIoU Loss
"""
import torch


def iou_loss(pred, target, eps=1e-7):
    """
    Computes the IoU and related geometric components between predicted and target boxes.

    Args:
        pred (Tensor): Predicted bounding boxes (N, 4) in [x1, y1, x2, y2] format.
        target (Tensor): Ground truth boxes (N, 4) in [x1, y1, x2, y2] format.
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple: Geometric values (center_x, center_y, width, height) for predicted and target boxes,
               along with IoU score.
    """

    # 
    pred_x1, pred_y1, pred_x2, pred_y2 = pred[:,
                                              0], pred[:, 1], pred[:, 2], pred[:, 3]
    target_x1, target_y1, target_x2, target_y2 = target[:,
                                                        0], target[:, 1], target[:, 2], target[:, 3]

    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    pred_widthidth = pred_x2 - pred_x1
    pred_height = pred_y2 - pred_y1

    target_center_x = (target_x1 + target_x2) / 2
    target_center_y = (target_y1 + target_y2) / 2
    target_width = target_x2 - target_x1
    target_height = target_y2 - target_y1

    # get top-left and bottom-right coordinates of intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # calculate two areas
    pred_area = pred_widthidth * pred_height
    target_area = target_width * target_height

    # calculate inter area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
        (inter_y2 - inter_y1).clamp(min=0)

    #calculate union area
    union_area = pred_area + target_area - inter_area

    iou = inter_area / (union_area + eps)

    return (
        pred_center_x, pred_center_y, pred_widthidth, pred_height,
        target_center_x, target_center_y, target_width, target_height,
        iou
    )


def ciou_loss(pred, target, eps=1e-7):
    """
    Computes Complete IoU (CIoU) loss between predicted and ground truth boxes.

    Args:
        pred (Tensor): Predicted bounding boxes (N, 4) in [x1, y1, x2, y2] format.
        target (Tensor): Ground truth boxes (N, 4) in [x1, y1, x2, y2] format.
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: CIoU loss for each predicted box.
    """
    pred_center_x, pred_center_y, pred_width, pred_height, target_center_x, target_center_y, target_width, target_height, iou = iou_loss(
        pred, target, eps)

    # Center distance term (DIoU)
    center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
    enclosing_x1 = torch.min(pred[:, 0], target[:, 0])
    enclosing_y1 = torch.min(pred[:, 1], target[:, 1])
    enclosing_x2 = torch.max(pred[:, 2], target[:, 2])
    enclosing_y2 = torch.max(pred[:, 3], target[:, 3])
    enclosing_diag = (enclosing_x2 - enclosing_x1) ** 2 + \
        (enclosing_y2 - enclosing_y1) ** 2
    diou_term = center_dist / (enclosing_diag + eps)

    # Aspect ratio consistency term
    aspect_ratio_term = (4 / (torch.pi ** 2)) * (torch.atan(target_width / (target_height + eps)) -
                                 torch.atan(pred_width / (pred_height + eps))) ** 2
    with torch.no_grad():
        alpha = aspect_ratio_term / (1 - iou + aspect_ratio_term + eps)
    ciou_term = alpha * aspect_ratio_term

    return 1 - iou + diou_term + ciou_term


def diou_loss(pred, target, eps=1e-7):
    """
    Computes Distance IoU (DIoU) loss between predicted and ground truth boxes.

    Args:
        pred (Tensor): Predicted bounding boxes (N, 4) in [x1, y1, x2, y2] format.
        target (Tensor): Ground truth boxes (N, 4) in [x1, y1, x2, y2] format.
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: DIoU loss for each predicted box.
    """
    pred_center_x, pred_center_y, _, _, target_center_x, target_center_y, _, _, iou = iou_loss(
        pred, target, eps)

    center_dist = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
    enclosing_x1 = torch.min(pred[:, 0], target[:, 0])
    enclosing_y1 = torch.min(pred[:, 1], target[:, 1])
    enclosing_x2 = torch.max(pred[:, 2], target[:, 2])
    enclosing_y2 = torch.max(pred[:, 3], target[:, 3])
    enclosing_diag = (enclosing_x2 - enclosing_x1) ** 2 + \
        (enclosing_y2 - enclosing_y1) ** 2

    return 1 - iou + center_dist / (enclosing_diag + eps)
