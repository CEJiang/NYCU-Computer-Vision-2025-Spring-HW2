"""
train.py

Main training script for digit detection using Faster R-CNN with ResNet-50 backbone.

This script handles:
- Dataset loading and transformation
- Model initialization with custom anchor settings
- Training and validation loops with support for DIoU / CIoU loss integration
- Checkpoint saving and metric plotting
"""
import os
import time
import json
import tempfile
from contextlib import redirect_stdout
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pycocotools.cocoeval import COCOeval
import torch

from dataset import DigitDetectionDataset

from utils import plot_loss_accuracy
from utils import clear_memory
from utils import visualize_sample
from utils import visualize_batch
from utils import plot_val_map_curve
from utils import diou_loss, ciou_loss


def collate_fn(batch):
    """
    Transposes the batch (list of tuples) into a tuple of lists, which is
    required for handling variable-length targets like bounding boxes.

    Returns:
        A tuple of (images, targets).
    """
    return tuple(zip(*batch))


def get_transform(train):
    """
     Returns a composition of image transformations for training or evaluation.

    Args:
        train (bool): Whether the transformations are for training mode.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations including
        random color jitter, Gaussian blur (only if training), and normalization.
    """
    transforms = [T.ToTensor()]

    if train:
        transforms += [
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ], p=0.3),

            T.RandomApply([
                T.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8))
            ], p=0.3)
        ]

    transforms += [
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]

    return T.Compose(transforms)


def load_data(data_path, batch_size):
    """
    Loads and prepares the training and validation datasets for digit detection.

    Args:
        data_path (str): Path to the dataset directory containing
                         'train' and 'valid' subfolders and annotations.
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - num_classes (int): The number of classes including background
                                 (fixed as 11 for digits 0-9 + background).
    """
    train_dataset = DigitDetectionDataset(
        img_dir=f"{data_path}/train",
        annotation_file=f"{data_path}/train.json",
        transforms=get_transform(train=True)
    )
    val_dataset = DigitDetectionDataset(
        img_dir=f"{data_path}/valid",
        annotation_file=f"{data_path}/valid.json",
        transforms=get_transform(train=False)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, 11


class DetectionTrainer:
    """
    A training and evaluation engine for object detection models.

    This class encapsulates training, validation, checkpoint saving/loading,
    and performance tracking (loss and mAP) for models such as Faster R-CNN.

    Methods:
        - train(): Performs one epoch of training with optional loss blending (DIoU/CIoU).
        - validate(): Evaluates model on validation set and computes COCO mAP.
        - save_checkpoint(): Saves model, optimizer, and training state.
        - load_checkpoint(): Loads saved training state from checkpoint.
    """
    def __init__(self, model, optimizer, scheduler, device, save_path):
        """
        Initializes the training engine with model, optimizer, scheduler, and other configurations.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            optimizer (Optimizer): The optimizer used for updating model parameters.
            scheduler (LRScheduler): Learning rate scheduler.
            device (torch.device): The device on which training and evaluation will run.
            save_path (str): Directory path to save best model weights and result images.

        Attributes:
            best_loss (float): Tracks the best validation loss.
            best_map (float): Tracks the best mAP achieved.
            train_losses (list): Stores training loss per epoch.
            val_losses (list): Stores validation loss per epoch.
            val_maps (list): Stores validation mAP per epoch.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.best_loss = float("inf")
        self.best_map = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        os.makedirs(save_path + "/Images", exist_ok=True)

    def train(self, dataloader, epoch, args):
        """
        Trains the model for one epoch with optional DIoU/CIoU loss integration.

        Args:
            dataloader (DataLoader): DataLoader providing training data batches.
            epoch (int): Current epoch index (used for loss schedule).
            args (Namespace): Arguments containing the `loss_type` setting
                            ('original', 'diou', or 'ciou').

        Returns:
            float: Average training loss for the current epoch.

        Notes:
            - Uses AMP (Automatic Mixed Precision) with GradScaler and autocast.
            - Blends DIoU or CIoU into the bounding box regression loss
            progressively after epoch 6 using a linear weight schedule.
            - Performs loss scaling and optimizer stepping via AMP.
        """
        self.model.train()
        total_loss = 0.0
        start = time.time()
        scaler = torch.amp.GradScaler()  # 建立 GradScaler

        for batch_idx, (images, targets) in enumerate(dataloader):
            if batch_idx % 100 == 1:
                start = time.time()
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            with torch.amp.autocast(enabled=True, device_type=self.device.type):  # 使用 autocast
                loss_dict = self.model(images, targets)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(images)
                self.model.train()

                pred_boxes = []
                gt_boxes = []
                for out, tgt in zip(outputs, targets):
                    if out['boxes'].size(0) > 0 and tgt['boxes'].size(0) > 0:
                        num = min(out['boxes'].size(0), tgt['boxes'].size(0))
                        pred_boxes.append(out['boxes'][:num])
                        gt_boxes.append(tgt['boxes'][:num])

                if pred_boxes and gt_boxes:
                    pred_boxes = torch.cat(pred_boxes, dim=0)
                    gt_boxes = torch.cat(gt_boxes, dim=0)
                    original_loss = loss_dict['loss_box_reg']
                    alpha = 0.0 if epoch < 6 else min(
                        0.1 + 0.05 * (epoch - 6), 0.5)

                    if args.loss_type == "diou":
                        loss_dict['loss_box_reg'] = (
                            1 - alpha) * original_loss + \
                        alpha * diou_loss(pred_boxes, gt_boxes).mean()
                    elif args.loss_type == "ciou":
                        loss_dict['loss_box_reg'] = (
                            1 - alpha) * original_loss + \
                                alpha * ciou_loss(pred_boxes, gt_boxes).mean()
                    elif args.loss_type == "original":
                        loss_dict['loss_box_reg'] = original_loss

                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()  # 縮放損失並反向傳播
            scaler.step(self.optimizer)  # 更新模型參數
            scaler.update()  # 更新縮放因子

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f"  ├─ Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f} \
                        | Time: {time.time() - start:.2f}s")

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """
        Runs validation for one epoch, 
        computes loss and COCO-style mAP, 
        and saves prediction visualizations.

        Args:
            dataloader (DataLoader): DataLoader for the validation dataset.
            epoch (int): Current epoch number, used for naming saved results and logging.

        Returns:
            tuple:
                - avg_loss (float or None): 
                        Average validation loss across all batches, or None if unavailable.
                - map5095 (float or None): 
                        COCO mAP score at IoU thresholds [0.50:0.95], or None if no predictions.

        Workflow:
            - Sets the model to evaluation mode.
            - Collects sample predictions and targets for visualization (up to 5 images).
            - Runs inference and formats results for COCO evaluation.
            - Optionally computes validation loss.
            - Computes COCO evaluation metrics and logs results to file.

        Notes:
            - The function uses `@torch.no_grad()` for efficiency.
            - Visual outputs are saved to `Images/val_preds_epoch{epoch}`.
            - COCO mAP logs are appended to `map_log.txt` under the specified save path.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        coco_results = []
        coco_gt = dataloader.dataset.coco
        sample_images, sample_outputs, sample_targets = ([], [], [])

        for images, targets in dataloader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device)
                        for k, v in t.items()} for t in targets]
            outputs = self.model(images)

            for output, target, img in zip(outputs, targets, images):
                image_id = int(target['image_id'].item())
                if len(sample_images) < 5:
                    sample_images.append(img.cpu())
                    sample_outputs.append({k: v.cpu()
                                          for k, v in output.items()})
                    sample_targets.append({k: v.cpu()
                                          for k, v in target.items()})

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(x_min), float(y_min), \
                                 float(x_max - x_min), float(y_max - y_min)],
                        'score': float(score)
                    })

        visualize_batch(
            sample_images,
            sample_outputs,
            targets=sample_targets,
            save_dir=os.path.join(
                self.save_path,
                f'Images/val_preds_epoch{epoch + 1}'),
            show=False
        )

        try:
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device)
                            for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                if isinstance(loss_dict, dict):
                    total_loss += sum((loss for loss in loss_dict.values())).item()
                    count += 1
        except BaseException:
            pass

        avg_loss = total_loss / count if count > 0 else None
        if avg_loss:
            print(f'[Epoch {epoch + 1}] Val Loss: {avg_loss:.4f}')

        map5095 = None
        if coco_results:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as file:
                json.dump(coco_results, file)
                file.flush()
                coco_dt = coco_gt.loadRes(file.name)

            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()

            log_file = os.path.join(self.save_path, "map_log.txt")
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(f"\n[Epoch {epoch + 1}]\n")
                with redirect_stdout(file):
                    coco_eval.summarize()

            map5095 = coco_eval.stats[0]
            print(f'[Epoch {epoch + 1}] mAP@0.5:0.95 = {map5095:.4f}')
        else:
            print(f'[Epoch {epoch + 1}] No predictions for mAP evaluation.')

        return (avg_loss, map5095)

    def save_checkpoint(self, epoch, is_best=False):
        """
        Saves the current training checkpoint, including model and optimizer states.

        Args:
            epoch (int): The current epoch number, to be stored in the checkpoint.
            is_best (bool, optional): Whether the current model is the best so far.
                                    If True, saves an additional copy as 'best_model.pth'.

        Saves:
            - latest_checkpoint.pth: 
                    Contains model, optimizer, scheduler states, and training history.
            - best_model.pth: 
                    (if is_best=True) Contains only the model weights.
        """
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_map": self.best_map,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_maps": self.val_maps
        }

        torch.save(ckpt, os.path.join(self.save_path, "latest_checkpoint.pth"))
        print("Checkpoint saved.")
        if is_best:

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.save_path,
                    "best_model.pth"))
            print("Best model updated.")

    def load_checkpoint(self, path):
        """
        Loads a saved checkpoint and restores model, optimizer, and scheduler states.

        Args:
            path (str): Path to the checkpoint file to be loaded.

        Returns:
            int: The epoch number to resume training from (epoch + 1).
                Returns 0 if no checkpoint is found at the given path.

        Restores:
            - Model weights
            - Optimizer state
            - Scheduler state
            - Best validation loss and mAP
            - Training and validation loss/mAP history
        """
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.best_loss = checkpoint.get("best_loss", float("inf"))
            self.best_map = checkpoint.get("best_map", 0.0)
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            self.val_maps = checkpoint.get("val_maps", [])
            print(
                f"Resumed from epoch {checkpoint['epoch'] + 1} | Best mAP: {self.best_map:.4f}")
            return checkpoint["epoch"] + 1
        return 0


def validate_model(device, net, val_loader, args):
    """
    Loads the latest model checkpoint and runs validation on the validation set.

    Args:
        device (torch.device): The device to run inference on.
        net (nn.Module): The model to be evaluated.
        val_loader (DataLoader): DataLoader providing validation data.
        args (Namespace): Parsed arguments containing configuration, including save path.

    Workflow:
        - Loads model weights from 'latest_checkpoint.pth'.
        - Initializes a DetectionTrainer with dummy optimizer/scheduler.
        - Runs `validate()` once using the loaded model.
    """
    ckpt_path = os.path.join(args.save, "latest_checkpoint.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)

    trainer = DetectionTrainer(
        net,
        optimizer=None,
        scheduler=None,
        device=device,
        save_path=args.save)

    print("Running validation on validation set only...")
    _, _ = trainer.validate(val_loader, epoch=999)


def train_model(device, net, optimizer, train_loader,
                val_loader, scheduler, args):
    """
    Runs full training loop across all epochs, 
        including checkpointing, validation, and visualization.

    Args:
        device (torch.device): The device to run training on.
        net (nn.Module): The model to be trained.
        optimizer (Optimizer): Optimizer for model parameter updates.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        scheduler (LRScheduler): Learning rate scheduler.
        args (Namespace): 
            Parsed arguments containing training config (epochs, save path, loss type, etc.).

    Workflow:
        - Visualizes a few training samples.
        - Initializes DetectionTrainer and optionally resumes from checkpoint.
        - For each epoch:
            - Trains and validates the model.
            - Updates training logs and learning rate.
            - Saves checkpoint if current mAP is the best so far.
        - Saves loss curve and validation mAP curve plots.
    """
    sample_images, sample_targets = next(iter(train_loader))
    for i in range(min(3, len(sample_images))):
        visualize_sample(
            sample_images[i],
            sample_targets[i],
            title=f"Sample {i}")

    trainer = DetectionTrainer(net, optimizer, scheduler, device, args.save)
    start_epoch = trainer.load_checkpoint(
        os.path.join(args.save, "latest_checkpoint.pth"))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = trainer.train(train_loader, epoch, args)
        val_loss, val_map = trainer.validate(val_loader, epoch)

        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(
            val_loss if val_loss is not None else float("nan"))
        trainer.val_maps.append(val_map)

        scheduler.step()
        torch.cuda.empty_cache()

        if val_map is not None and val_map > trainer.best_map:
            trainer.best_map = val_map
            is_best = True
        else:
            is_best = False

        trainer.save_checkpoint(epoch, is_best=is_best)
        clear_memory()
        print(f"Epoch Time: {time.time() - start_time:.2f}s")

    plot_loss_accuracy(trainer.train_losses, trainer.val_losses, save_fig=True,
                       output_path=os.path.join(args.save, "training_curve.png"))
    plot_val_map_curve(
        trainer.val_maps,
        output_path=os.path.join(
            args.save,
            "val_map_curve.png"))
