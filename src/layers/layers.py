"""Define custom layers"""

import math
from typing import Optional

import torch
import torch.nn as tnn
from einops import rearrange
from einops.layers.torch import Rearrange


class Residual(tnn.Module):
    """Create residual connection layer"""

    def __init__(self, func: tnn.Module) -> None:
        super().__init__()
        self.func = func

    def forward(self, x_inp: torch.Tensor, *args: int, **kwargs: int) -> torch.Tensor:
        """Forward propagation"""
        return self.func(x_inp, *args, **kwargs) + x_inp


class Upsample(tnn.Module):
    """Upsample layer, no more Strided/transposed Convolutions
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
    """

    def __init__(self, in_channel: int, out_channel: Optional[int] = None) -> None:
        super().__init__()
        out_channel = out_channel if out_channel else in_channel
        self.layer = tnn.Sequential(
            tnn.Upsample(scale_factor=2, mode="nearest"),
            tnn.Conv2d(in_channel, out_channel, 3, padding=1),
        )

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        return self.layer(x_inp)


class Downsample(tnn.Module):
    """Downsample layer, no more Strided Convolutions or Pooling
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
    """

    def __init__(self, in_channel: int, out_channel: Optional[int] = None) -> None:
        super().__init__()
        out_channel = out_channel if out_channel else in_channel
        self.layer = tnn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            tnn.Conv2d(in_channel * 4, out_channel, 1),
        )

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        return self.layer(x_inp)


class SinusoidalPositionEmbeddings(tnn.Module):
    """Position Embeddings"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        device = time.device
        half_dim = self.dim // 2
        norm_emb = math.log(10000) / (half_dim - 1)
        pos_emb = torch.exp(torch.arange(half_dim, device=device) * -norm_emb)
        pos_emb = time[:, None] * pos_emb[None, :]
        pos_emb = torch.cat((pos_emb.sin(), pos_emb.cos()), dim=-1)
        return pos_emb


class Block(tnn.Module):
    """Block of layers
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
        groups: Number of groups in group normalization
    """

    def __init__(self, in_channel: int, out_channel: Optional[int] = None, groups: int = 8) -> None:
        super().__init__()
        out_channel = out_channel if out_channel else in_channel
        self.proj = tnn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.norm = tnn.GroupNorm(groups, out_channel)
        self.act = tnn.GELU()

    def forward(
        self,
        x_inp: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation
        Args:
            x_inp: Input batch of tensor, (batch, channels, height, width)
            scale: Scale using context embeddings, optional, (Batch, emb_dimension, 1, 1)
            shift: Add time embeddings, optional, (batch, emb_dimension, 1, 1)
        """
        if scale is not None:
            x_inp = x_inp * scale
        if shift is not None:
            x_inp = x_inp + shift

        x_inp = self.proj(x_inp)
        x_inp = self.norm(x_inp)
        x_inp = self.act(x_inp)
        return x_inp


class ResnetBlock(tnn.Module):
    """Resnet block
    Args:
        in_channel: Number of input channels
        out_channel: Number of output channels, optional
        time_emb: Number of time embedding dimensions
        context_emb: Number of context embedding dimensions
        groups: Number of groups in group normalization
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: Optional[int],
        time_emb: Optional[int] = None,
        context_emb: Optional[int] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        out_channel = out_channel if out_channel else in_channel
        self.time_mlp = (
            tnn.Sequential(
                tnn.Linear(time_emb, in_channel), tnn.GELU(), tnn.Linear(in_channel, in_channel)
            )
            if time_emb is not None
            else None
        )
        self.context_mlp = (
            tnn.Sequential(
                tnn.Linear(context_emb, in_channel),
                tnn.GELU(),
                tnn.Linear(in_channel, in_channel),
            )
            if context_emb is not None
            else None
        )

        self.block1 = Block(in_channel, out_channel, groups=groups)
        self.block2 = Block(out_channel, out_channel, groups=groups)
        self.res_conv = (
            tnn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else tnn.Identity()
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
        scale, shift = None, None
        if self.time_mlp is not None and time is not None:
            time_ = self.time_mlp(time)
            shift = rearrange(time_, "b c -> b c 1 1")
        if self.context_mlp is not None and context is not None:
            context_ = self.context_mlp(context)
            scale = rearrange(context_, "b c -> b c 1 1")

        hid = self.block1(x_inp, scale=scale, shift=shift)
        hid = self.block2(hid)
        return hid + self.res_conv(x_inp)


class Attention(tnn.Module):
    """Self attention layer
    Args:
        dim: Dimension of embedding layer
        heads: Number of heads
        dim_head: Dimensions of each head
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = tnn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = tnn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        _, _, height, width = x_inp.shape
        qkv = self.to_qkv(x_inp).chunk(3, dim=1)
        query, key, value = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        query = query * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", query, key)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, value)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=height, y=width)
        return self.to_out(out)


class LinearAttention(tnn.Module):
    """Linear self attention
    Args:
        dim: Dimension of embedding layer
        heads: Number of heads
        dim_head: Dimensions of each head
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = tnn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = tnn.Sequential(tnn.Conv2d(hidden_dim, dim, 1), tnn.GroupNorm(1, dim))

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        _, _, height, width = x_inp.shape
        qkv = self.to_qkv(x_inp).chunk(3, dim=1)
        query, key, value = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        query = query.softmax(dim=-2)
        key = key.softmax(dim=-1)

        query = query * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", key, value)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, query)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=height, y=width)
        return self.to_out(out)


class PreNorm(tnn.Module):
    """Group Normalization before torch module
    Args:
        dim: Dimension of the slice to be normalized
        fun: Torch module to aply after normalization
    """

    def __init__(self, dim: int, fun: tnn.Module) -> None:
        super().__init__()
        self.fun = fun
        self.norm = tnn.GroupNorm(1, dim)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        x_inp = self.norm(x_inp)
        return self.fun(x_inp)
