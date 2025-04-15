"""
main.py

Trains, validates, or tests a Faster R-CNN model for digit detection.
"""
import argparse

from torch.backends import cudnn
from torch.optim import lr_scheduler
import torch

from train import load_data, train_model, validate_model
from model import faster_rcnn_resnet50
from test import test_model


def main():
    """Main entry point: parses arguments and runs the selected mode."""
    cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Digit Detection with Faster R-CNN")

    parser.add_argument("data_path", type=str, help="Root path to dataset.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size.")
    parser.add_argument("-e", "--epoch", type=int, default=15, help="Epochs.")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.")
    parser.add_argument(
        "-em",
        "--eta_min",
        type=float,
        default=1e-6,
        help="Min LR (cosine annealing).")
    parser.add_argument(
        "-d",
        "--decay",
        type=float,
        default=1e-4,
        help="Weight decay.")
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="saved_models",
        help="Save directory.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=[
            "train",
            "validate",
            "test"],
        default="train",
        help="Execution mode.")
    parser.add_argument(
        "-l",
        "--loss_type",
        type=str,
        choices=[
            "original",
            "diou",
            "ciou"],
        default="original",
        help="Box loss type.")

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
        train_model(device=device, net=net, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, args=args)
    elif args.mode == 'validate':
        validate_model(
            device=device,
            net=net,
            val_loader=val_loader,
            args=args)
    else:
        test_model(device=device, model=net, args=args)


if __name__ == "__main__":
    main()
