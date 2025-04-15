"""
dataset.py

Custom PyTorch Dataset for digit detection in COCO format.

Supports:
- Loading image and annotation pairs
- Mapping image IDs to filenames
- Sorting images by filename
"""
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class DigitDetectionDataset(Dataset):
    """
    PyTorch Dataset class for digit detection using COCO-style annotations.

    Args:
        img_dir (str): Directory containing the images.
        annotation_file (str): Path to the COCO-format JSON annotation file.
        transforms (callable, optional): Optional image transformations (e.g., augmentations).

    Attributes:
        coco (COCO): COCO object for evaluation or advanced access.
        image_info (list): List of dicts with image ID and filename.
        annotations (dict): Mapping from image ID to bounding box annotations.
        id_to_filename (dict): Mapping from image ID to filename.
    """

    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        self.coco = COCO(annotation_file)

        with open(annotation_file, encoding='utf-8') as file:
            coco = json.load(file)

        self.image_info = []
        self.annotations = {}
        self.id_to_filename = {}

        for img in coco["images"]:
            img_id = img["id"]
            file_name = img["file_name"]
            self.image_info.append({"id": img_id, "file_name": file_name})
            self.id_to_filename[img_id] = file_name

        self.image_info.sort(key=lambda x: x["file_name"])

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            bbox = ann["bbox"]  # [x, y, w, h]
            label = ann["category_id"]  # 1~10
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append({"bbox": bbox, "label": label})

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x_min, y_min, width, height = ann["bbox"]
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann["label"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(img_id, dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
