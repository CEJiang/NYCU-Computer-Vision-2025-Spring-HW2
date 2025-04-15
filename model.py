from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead


def faster_rcnn_resnet50(num_classes, device, args):
    """
    Builds a Faster R-CNN model with a ResNet-50 FPN backbone.

    The model is initialized from torchvision's pretrained weights and customized with:
        - Custom anchor generator (sizes and aspect ratios)
        - Custom RPN head with matching number of anchors
        - Multi-scale RoI Align
        - Custom Fast R-CNN predictor for classification
        - Inference thresholds adjusted based on mode (train / validate / test)

    Args:
        num_classes (int): Number of output classes including background.
        device (torch.device): The device to place the model on.
        args (Namespace): Argument object with a 'mode' attribute.

    Returns:
        FasterRCNN: A PyTorch Faster R-CNN model ready for training or inference.
    """
    weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # anchor_generator = AnchorGenerator(
    #     sizes=((4,), (8,), (12,), (24,), (48,)),
    #     aspect_ratios=((0.25, 0.5, 1.0),) * 5
    # )
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
    """
    Builds a Faster R-CNN model with a custom ResNet-101 FPN backbone.

    The backbone is built using torchvision's resnet101 with FPN and 5 trainable layers.
    The model is customized with:
        - Custom anchor generator
        - Multi-scale RoI Align
        - Fast R-CNN predictor with modified output head
        - Inference thresholds set based on mode

    Args:
        num_classes (int): Number of output classes including background.
        device (torch.device): The device to place the model on.
        args (Namespace): Argument object with a 'mode' attribute ('train', 'validate', or 'test').

    Returns:
        FasterRCNN: A PyTorch Faster R-CNN model ready for training or inference.
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
