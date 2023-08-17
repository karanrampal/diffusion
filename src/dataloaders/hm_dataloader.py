"""Define custom dataset class extending the Pytorch Dataset class"""

import os
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.utils import Params


class AttributesDataset(Dataset):
    """Custom class for Attribute prediction dataset
    Args:
        root: Directory containing the dataset
        file_path: Path of the train/val/test file relative to the root
        transforms: Data augmentation to be done
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        transforms: tvt.Compose,
    ) -> None:
        self.root = root
        self.data = pd.read_csv(os.path.join(root, file_path))
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset given the index idx"""
        row = self.data.iloc[idx]

        im_name = row["path"]
        im_path = os.path.join(self.root, "images", im_name)
        img_pil = Image.open(im_path).convert("RGB")
        img = self.transforms(img_pil)

        labels = torch.as_tensor(row[1:], dtype=torch.float32)

        return img, labels

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.data)


def get_transform(is_train: bool, params: Params) -> tvt.Compose:
    """Data augmentation
    Args:
        is_train: If the dataset is training
        params: Hyperparameters
    Returns:
        Composition of all the data transforms
    """
    trans = [
        tvt.Resize((params.height, params.width)),
        tvt.ToTensor(),
        tvt.Normalize(mean=(0.5,), std=(0.5,)),
    ]
    if is_train:
        trans += [
            tvt.RandomHorizontalFlip(params.flip),
            tvt.RandomRotation(params.degree),
        ]
    return tvt.Compose(trans)


def get_dataloader(
    params: Params,
) -> DataLoader:
    """Get DataLoader objects.
    Args:
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataset = AttributesDataset(
        root=params.data_dir,
        file_path="annotations/all.csv",
        transforms=get_transform(True, params),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        shuffle=True,
    )

    return dataloader
