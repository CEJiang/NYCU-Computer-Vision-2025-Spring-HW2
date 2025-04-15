"""
main.py

Main entry point for training, validating, or testing a Faster R-CNN model
on a digit detection task using PyTorch.
"""
import argparse

from torch.backends import cudnn
from torch.optim import lr_scheduler
import torch

from train import load_data, train_model, validate_model
from model import faster_rcnn_resnet50
from test import test_model


def main():
    """
    Parses command-line arguments, initializes model, and runs training,
    validation, or testing based on the selected mode.

    The script supports:
    - Customizable training hyperparameters (batch size, epochs, lr, decay)
    - Loss scheduling with optional DIoU/CIoU integration
    - Saving checkpoints and results to a specified directory
    """
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Digit Detection with Faster R-CNN")

    parser.add_argument(
        "data_path",
        type=str,
        help="Root path to dataset folder containing 'train', 'valid', and 'test' subfolders."
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=8,
        help="Batch size for training and validation."
    )
    parser.add_argument(
        "--epoch", "-e",
        type=int,
        default=15,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for optimizer."
    )
    parser.add_argument(
        "--eta_min", "-em",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine annealing scheduler."
    )
    parser.add_argument(
        "--decay", "-d",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) coefficient."
    )
    parser.add_argument(
        "--save_path", "-s",
        type=str,
        default="saved_models",
        help="Directory to save model checkpoints and output results."
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["train", "validate", "test"],
        default="train",
        help="Execution mode: train the model, validate on validation set, or test on test set."
    )
    parser.add_argument(
        "--loss_type", "-l",
        type=str,
        choices=["original", "diou", "ciou"],
        default="original",
        help="Loss function for box regression: 'original' (SmoothL1), 'diou', or 'ciou'."
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    train_loader, val_loader, num_classes = load_data(
        args.data_path, args.batch_size)

    net = faster_rcnn_resnet50(
        num_classes=num_classes,
        device=device,
        args=args)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=args.eta_min)

    if args.mode == 'train':
        train_model(device=device,
                    net=net,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    scheduler=scheduler,
                    args=args)
    elif args.mode == 'validate':
        validate_model(device=device,
                       net=net,
                       val_loader=val_loader,
                       args=args)
    else:
        test_model(device=device, model=net, args=args)


if __name__ == "__main__":
    main()
