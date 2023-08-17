"""Implementation of a contextual encoder"""

from typing import Optional, Tuple

import torch

from layers import layers


class ContextEncoder(torch.nn.Module):
    """Contextual encoder neural network architecture
    Args:
        channels: Input tuple of channels of the layers
        time_dim: Number of time embedding dimensions
        context_dim: Number of context embedding dimensions
        num_groups: Number of groups in group normalization
        num_heads: Number of heads
        head_dim: Dimensions of each head
    """

    def __init__(
        self,
        channels: Tuple[int, ...],
        num_groups: int,
        num_heads: int,
        head_dim: int,
        time_dim: Optional[int] = None,
        context_emb: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.enc_layers = torch.nn.ModuleList([])
        for i in range(len(channels) - 1):
            self.enc_layers.append(
                torch.nn.ModuleList(
                    [
                        layers.ResnetBlock(
                            channels[i], channels[i + 1], time_dim, context_emb, num_groups
                        ),
                        layers.ResnetBlock(
                            channels[i + 1], channels[i + 1], time_dim, context_emb, num_groups
                        ),
                        layers.Residual(
                            layers.PreNorm(
                                channels[i + 1],
                                layers.LinearAttention(channels[i + 1], num_heads, head_dim),
                            )
                        ),
                        layers.Downsample(channels[i + 1], channels[i + 1]),
                    ]
                )
            )

    def forward(
        self,
        xinp: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Forward propagation of the encoder
        Args:
            xinp: Batch of noisy image tensor, (batch_size, channels, height, width)
            time: Batch of time step tensor, (batch_size, emb_dimension), optional
            context: Batch of context vectors, (batch_size, vector_dimension), optional
        Returns:
            List of encoded tensors at various downsampled sizes
        """
        out = []
        for block1, block2, attention, down_layer in self.enc_layers:
            xinp = block1(xinp, time, context)
            out.append(xinp)
            xinp = block2(xinp, time, context)
            xinp = attention(xinp)
            out.append(xinp)
            xinp = down_layer(xinp)
        out.append(xinp)
        return out
