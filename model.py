from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead


def faster_rcnn_resnet50(num_classes, device, args):
    """Faster R-CNN (ResNet-50 FPN).

    Pretrained, customizable detection model with mode-based thresholds.

    Args:
        num_classes (int): Output classes.
        device (torch.device): Model device.
        args (Namespace): Arguments with `mode`.

    Returns:
        FasterRCNN: PyTorch model.
    """
    weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    anchor_generator = AnchorGenerator(
        sizes=((4,), (8,), (12,), (24,), (48,)),
        aspect_ratios=((0.5, 1.0, 1.5, 2.0),) * 5
    )

    num_anchors = anchor_generator.num_anchors_per_location()[0]

    model.rpn.anchor_generator = anchor_generator

    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    if args.mode in ("train", "validate"):
        model.roi_heads.score_thresh = 0.5
        model.roi_heads.nms_thresh = 0.5
    elif args.mode == "test":
        model.roi_heads.score_thresh = 0.7
        model.roi_heads.nms_thresh = 0.3

    model.roi_heads.positive_fraction = 0.25

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.rpn.head = RPNHead(in_channels=256, num_anchors=num_anchors)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)


def faster_rcnn_resnet101(num_classes, device, args):
    """Faster R-CNN (ResNet-101 FPN).

    Customizable detection model with pretrained backbone and mode-based thresholds.

    Args:
        num_classes (int): Output classes.
        device (torch.device): Model device.
        args (Namespace): Arguments with `mode`.

    Returns:
        FasterRCNN: PyTorch model.
    """
    backbone = resnet_fpn_backbone(
        'resnet101',
        pretrained=True,
        trainable_layers=5)

    anchor_generator = AnchorGenerator(
        sizes=((4,), (8,), (12,), (24,), (48,)),
        aspect_ratios=((0.25, 0.5, 1.0),) * 5
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    if args.mode in ("train", "validate"):
        model.roi_heads.score_thresh = 0.5
        model.roi_heads.nms_thresh = 0.5
    elif args.mode == "test":
        model.roi_heads.score_thresh = 0.7
        model.roi_heads.nms_thresh = 0.3

    model.roi_heads.positive_fraction = 0.25

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)
