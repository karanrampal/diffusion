"""Dataloader for Sprite datatset"""

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader, Dataset

from utils.utils import Params


def get_transform() -> tvt.Compose:
    """Get dataset transform"""
    transform = tvt.Compose(
        [
            tvt.ToTensor(),  # from [0,255] to range [0.0,1.0]
            tvt.Normalize((0.5,), (0.5,)),  # range [-1,1]
        ]
    )
    return transform


class SpriteDataset(Dataset):
    """Sprite pytorch dataset
    Args:
        sfilename: Name of sprite numpy file
        lfilename: Name of sprite labels numpy file
        transfor: Pytorch data augmentations
        null_context: If context/labels will be used during diffusion model training
    """

    def __init__(
        self,
        sfilename: str,
        lfilename: str,
        transform: tvt.Compose,
        null_context: bool = False,
    ) -> None:
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.transform = transform
        self.null_context = null_context

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.sprites)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image and label at a given index, return the image and label as a tuple"""
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return image, label


def get_dataloader(
    params: Params,
) -> DataLoader:
    """Get DataLoader objects.
    Args:
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataset = SpriteDataset(
        sfilename="data/sprites_1788_16x16.npy",
        lfilename="data/sprite_labels_nc_1788_16x16.npy",
        transform=get_transform(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        shuffle=True,
    )

    return dataloader
