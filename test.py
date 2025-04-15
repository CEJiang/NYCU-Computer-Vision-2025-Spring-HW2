"""
test.py

Inference and post-processing script for digit detection using a trained Faster R-CNN model.

Supports:
- Loading and running inference on test images
- Exporting predictions in COCO JSON format (Task 1)
- Post-processing predictions to digit strings and saving results as CSV (Task 2)
"""
import os
import json
import csv
import torch
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset

from train import collate_fn, get_transform


class TestDataset(Dataset):
    """
    A dataset class for loading test images for inference.

    Args:
        image_dir (str): Path to the test image directory.

    Returns:
        Each sample is a tuple of:
            - image (Tensor): Transformed input image tensor.
            - file_name (str): Original filename (used for image ID extraction).
    """
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.loader = default_loader
        self.transform = get_transform(train=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.loader(self.image_paths[idx])
        file_name = os.path.basename(self.image_paths[idx])
        image = self.transform(image)
        return image, file_name


def load_data(data_path, batch_size):
    """
    Loads test data into a DataLoader for inference.

    Args:
        data_path (str): Path to the dataset root (must contain 'test' folder).
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: A DataLoader for the test dataset.
    """
    test_dataset = TestDataset(os.path.join(data_path, "test"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)
    return test_loader


def test_model(device, model, args):
    """
    Runs inference on the test set using the best model checkpoint and saves predictions.

    Args:
        device (torch.device): Device to run inference on.
        model (nn.Module): The trained detection model.
        args (Namespace): Arguments containing config (e.g., save path, data path, batch size).

    Saves:
        - pred.json: COCO-format detection results for Task 1.
        - pred.csv: Post-processed digit predictions for Task 2.
    """
    model_path = os.path.join(args.save, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    test_loader = load_data(args.data_path, args.batch_size)

    results_task1 = []

    with torch.no_grad():
        for images, file_names in tqdm(test_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for file_name, output in zip(file_names, outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                image_id = int(os.path.splitext(file_name)[0])

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    results_task1.append({
                        "image_id": image_id,
                        'bbox': [float(x_min), float(y_min), \
                                 float(x_max - x_min), float(y_max - y_min)],
                        "score": float(score),
                        "category_id": int(label)
                    })

    os.makedirs(args.save, exist_ok=True)
    pred_json_path = os.path.join(args.save, "pred.json")
    with open(pred_json_path, "w", encoding="utf-8") as file:
        json.dump(results_task1, file)
        print(f"Task 1 saved to {pred_json_path}")

    test_task2(
        pred_json_path, os.path.join(
            args.data_path, "test"), os.path.join(
            args.save, "pred.csv"))


def test_task2(input_json_path, test_image_dir, output_csv_path):
    """
    Converts detection results in pred.json to digit predictions (Task 2).

    Args:
        input_json_path (str): Path to JSON file containing detection results.
        test_image_dir (str): Path to the directory containing test images.
        output_csv_path (str): Output CSV path for saving Task 2 results.

    Task 2 Logic:
        - Group detections by image_id
        - Sort digits from left to right by x-coordinate
        - Combine digit class predictions into a number string
        - Convert to integer; if conversion fails, use -1
    """
    with open(input_json_path, 'r', encoding="utf-8") as file:
        predictions = json.load(file)

    image_predictions = {}
    for pred in predictions:
        image_id = pred['image_id']
        if image_id not in image_predictions:
            image_predictions[image_id] = []
        image_predictions[image_id].append(pred)

    image_files = sorted(os.listdir(test_image_dir))
    all_image_ids = set()
    for file_name in image_files:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_id = int(os.path.splitext(file_name)[0])
                all_image_ids.add(image_id)
            except ValueError:
                print(
                    f"Warning: Could not extract image_id from filename: {file_name}")

    results_task2 = []
    for image_id in sorted(list(all_image_ids)):
        detections = image_predictions.get(image_id, [])
        if not detections:
            results_task2.append([image_id, -1])
        else:
            detections.sort(key=lambda x: x['bbox'][0])
            predicted_number_str = ""
            for detection in detections:
                predicted_number_str += str(detection['category_id'] - 1)
            try:
                predicted_number = int(predicted_number_str)
                results_task2.append([image_id, predicted_number])
            except ValueError:
                results_task2.append([image_id, -1])

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(results_task2)
    print(f"Task 2 predictions saved to {output_csv_path}")
