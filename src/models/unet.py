"""Conditional Unet"""

from typing import Optional, Tuple

import torch
import torch.nn as tnn

from layers import layers
from models.decoder import ContextDecoder
from models.encoder import ContextEncoder


class MidBlock(tnn.Module):
    """Mid level block
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
        time_dim: Number of time embedding dimensions
        context_emb: Number of context embedding dimensions
        num_heads: Number of heads
        head_dim: Dimensions of each head
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        time_dim: Optional[int] = None,
        context_emb: Optional[int] = None,
        num_groups: int = 8,
        num_heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.mid_block1 = layers.ResnetBlock(
            in_channel, out_channel, time_dim, context_emb, num_groups
        )
        self.mid_attn = layers.Residual(
            layers.PreNorm(out_channel, layers.Attention(out_channel, num_heads, head_dim))
        )
        self.mid_block2 = layers.ResnetBlock(
            out_channel, out_channel, time_dim, context_emb, num_groups
        )

    def forward(
        self,
        x_inp: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation
        Args:
            x_inp: Input batch of tensor, (batch, channels, height, width)
            time: Time embeddings, optional, (batch, emb_dimension)
            context: Context embeddings, optional, (Batch, emb_dimension)
        Returns:
            Output tensor
        """
        x_inp = self.mid_block1(x_inp, time, context)
        x_inp = self.mid_attn(x_inp)
        x_inp = self.mid_block2(x_inp, time, context)
        return x_inp


class FinalBlock(tnn.Module):
    """Final level block
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
        time_dim: Number of time embedding dimensions
        context_dim: Number of context embedding dimensions
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        time_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.final_res_block = layers.ResnetBlock(
            in_channel * 2, in_channel, time_dim, context_dim, num_groups
        )
        self.final_conv = tnn.Conv2d(in_channel, out_channel, 1)

    def forward(
        self,
        x_inp: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation
        Args:
            x_inp: Input batch of tensor, (batch, channels, height, width)
            time: Time embeddings, optional, (batch, emb_dimension)
            context: Context embeddings, optional, (Batch, emb_dimension)
        Returns:
            Output tensor
        """
        out = self.final_res_block(x_inp, time, context)
        return self.final_conv(out)


class Unet(tnn.Module):
    """Conditional Unet"""

    def __init__(
        self,
        in_channels: int,
        context_dim: Optional[int] = None,
        channels: Tuple[int, ...] = (16, 32, 64, 128),
        num_heads: int = 4,
        head_dim: int = 32,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        init_channels = channels[0]
        time_dim = channels[0]
        channels = (init_channels,) + channels
        self.init_conv = tnn.Conv2d(in_channels, init_channels, 1)

        self.time_mlp = tnn.Sequential(
            layers.SinusoidalPositionEmbeddings(32),
            tnn.Linear(32, time_dim),
            tnn.GELU(),
            tnn.Linear(time_dim, time_dim),
        )

        self.enc = ContextEncoder(channels, num_groups, num_heads, head_dim, time_dim, context_dim)
        self.dec = ContextDecoder(channels, num_groups, num_heads, head_dim, time_dim, context_dim)

        mid_channels = channels[-1]
        self.mid_block = MidBlock(
            mid_channels, mid_channels, time_dim, context_dim, num_heads, head_dim
        )

        self.final_conv = FinalBlock(init_channels, in_channels, time_dim, context_dim, num_groups)

    def forward(
        self,
        x_inp: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation
        Args:
            x_inp: Input batch of tensor, (batch, channels, height, width)
            time: Time embeddings, optional, (batch,)
            context: Context embeddings, optional, (Batch, emb_dimension)
        Returns:
            Output tensor
        """
        x_inp = self.init_conv(x_inp)
        time = self.time_mlp(time)

        tmp = self.enc(x_inp, time, context)
        out = self.mid_block(tmp.pop(), time, context)
        out = self.dec(out, tmp, time, context)

        out = torch.cat((out, x_inp), dim=1)
        return self.final_conv(out, time, context)
