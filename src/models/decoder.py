"""Implementation of a contextual decoder"""

from typing import Optional, Tuple

import torch

from layers import layers


class ContextDecoder(torch.nn.Module):
    """Contextual decoder neural network architecture
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
        self.dec_layers = torch.nn.ModuleList([])
        for i in reversed(range(len(channels) - 1)):
            self.dec_layers.append(
                torch.nn.ModuleList(
                    [
                        layers.Upsample(channels[i + 1], channels[i]),
                        layers.ResnetBlock(
                            channels[i] + channels[i + 1],
                            channels[i],
                            time_dim,
                            context_emb,
                            num_groups,
                        ),
                        layers.ResnetBlock(
                            channels[i] + channels[i + 1],
                            channels[i],
                            time_dim,
                            context_emb,
                            num_groups,
                        ),
                        layers.Residual(
                            layers.PreNorm(
                                channels[i],
                                layers.LinearAttention(channels[i], num_heads, head_dim),
                            )
                        ),
                    ]
                )
            )

    def forward(
        self,
        xinp: torch.Tensor,
        out: list[torch.Tensor],
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation of the decoder
        Args:
            xinp: Batch of noisy encoder output at last scale, (batch_size, channels, height, width)
            out: List of intermediate encoder outputs, (batch_size, channels, height, width)
            time: Batch of time step tensor, (batch_size, emb_dimension), optional
            context: Batch of context vectors, (batch_size, vector_dimension), optional
        Returns:
            Decoded tensor same size as input of the encoder
        """
        for up_layer, block1, block2, attention in self.dec_layers:
            xinp = up_layer(xinp)
            xinp = torch.cat((xinp, out.pop()), dim=1)
            xinp = block1(xinp, time, context)
            xinp = torch.cat((xinp, out.pop()), dim=1)
            xinp = block2(xinp, time, context)
            xinp = attention(xinp)
        return xinp
