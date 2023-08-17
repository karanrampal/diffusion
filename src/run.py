#!/usr/bin/env python
"""Run training script"""

import argparse

import torch
from torch.optim import Adam

from dataloaders.dataloader import get_dataloader
from models.unet import Unet
from trainers.trainer import CustomTrainer
from utils.utils import Params


def args_parser() -> argparse.Namespace:
    """CLI args parser"""
    parser = argparse.ArgumentParser(description="Parse CLI")
     # Experiment related
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--tb_dir", default="logs", type=str, help="Project name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("-d", "--out_dir", default="experiments", type=str, help="Output directory")
    parser.add_argument(
        "-f", "--validation_frequency", default=1, type=int, help="Validation frequency"
    )
    parser.add_argument("--save_epoch_freq", default=10, type=int, help="Saving frequency by epoch")
    parser.add_argument(
        "--checkpoint_steps",
        type=str,
        default="epoch",
        help="Save after every n steps, epoch",
    )

    # Dataset related
    parser.add_argument("--dataset", default="sprite", type=str, help="Dataset to use for training")
    parser.add_argument("--height", default=224, type=int, help="Image height")
    parser.add_argument("-w", "--width", default=224, type=int, help="Image width")
    parser.add_argument("--flip", default=0.5, type=float, help="Image flip horizontal probability")
    parser.add_argument("--degree", default=1.5, type=float, help="Image rotation degrees")
    parser.add_argument(
        "--data_dir", default="../mnt/hm_images", type=str, help="Image data directory"
    )

    # Training related
    parser.add_argument("-e", "--num_epochs", default=30, type=int, help="Number of epochs")
    parser.add_argument("-t", "--timesteps", default=500, type=int, help="Number of timesteps")
    parser.add_argument("-b", "--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("-l", "--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--decay", default=0.0, type=float, help="Decay rate")
    parser.add_argument(
        "--grad_accum_steps", default=1, type=int, help="Gradient accumulating steps"
    )
    parser.add_argument("-n", "--num_workers", default=2, type=int, help="Num workers")
    parser.add_argument(
        "--pin_memory", default=torch.cuda.is_available(), type=bool, help="Pin memory"
    )

    # Model related
    parser.add_argument("--init_channels", default=3, type=int, help="Number of image channels")
    parser.add_argument("--num_classes", default=5, type=int, help="Number of classes")

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = args_parser()
    params = Params(vars(args))
    print(params)

    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    model = Unet(
        in_channels=params.init_channels, context_dim=params.num_classes, channels=(64, 128)
    )
    optimizer = Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.decay)
    dataloader = get_dataloader(params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.learning_rate,
        epochs=params.num_epochs,
        steps_per_epoch=len(dataloader),
    )

    trainer = CustomTrainer(params)
    trainer.fit(model, optimizer, scheduler, dataloader)


if __name__ == "__main__":
    main()
