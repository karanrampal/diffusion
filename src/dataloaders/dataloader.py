"""Dataloader for pytorch"""

from torch.utils.data import DataLoader

import dataloaders.hm_dataloader as hmd
import dataloaders.sprite_dataloader as spd
from utils.utils import Params


def get_dataloader(
    params: Params,
) -> DataLoader:
    """Get DataLoader objects.
    Args:
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    if params.dataset == "hm":
        dataloader = hmd.get_dataloader(params)
    else:
        dataloader = spd.get_dataloader(params)

    return dataloader
