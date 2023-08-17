"""Diffusion schedule"""

from typing import Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm


class Diffuser:
    """Diffusion scheduler and sample generation
    Args:
        timesteps: Number of timesteps to run the forward diffusion process
        schedule: Type of schedule to use i.e. linear, quadratic, sigmoid
    """

    def __init__(self, timesteps: int, device: str, schedule: str = "linear") -> None:
        self.device = device
        self.timesteps = timesteps

        self.betas = self._select_schedule(schedule)

        alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumsum(alphas.log(), dim=0).exp()
        self.alphas_cumprod[0] = 1.0

    def _select_schedule(self, schedule: str) -> torch.Tensor:
        """Select a schedule for betas"""
        if schedule == "linear":
            return self.linear_beta_schedule()
        if schedule == "quadratic":
            return self.quadratic_beta_schedule()
        if schedule == "sigmoid":
            return self.sigmoid_beta_schedule()
        raise ValueError(f"Schedule can be linear, quadratic or sigmoid. {schedule} given!")

    def linear_beta_schedule(
        self, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> torch.Tensor:
        """Linear schedule"""
        return torch.linspace(beta_start, beta_end, self.timesteps + 1, device=self.device)

    def quadratic_beta_schedule(
        self, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> torch.Tensor:
        """Quadratic schedule"""
        schedule = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, self.timesteps + 1, device=self.device
            )
            ** 2
        )
        return schedule

    def sigmoid_beta_schedule(
        self, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> torch.Tensor:
        """Sigoid schedule"""
        betas = torch.linspace(-6, 6, self.timesteps + 1, device=self.device)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def q_sample(
        self, x_start: torch.Tensor, time: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion (using the nice property)"""
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[time, None, None, None].sqrt()
        one_minus_alphas_cumprod_t = 1.0 - self.alphas_cumprod[time, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + one_minus_alphas_cumprod_t * noise

    def _p_sample(
        self, x_inp: torch.Tensor, time: torch.Tensor, pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """Reverse sampling process"""
        noise = torch.randn_like(x_inp, device=self.device) if time > 1 else 0.0
        scaled_noise = self.betas[time].sqrt() * noise

        scaled_pred_noise = pred_noise * (self.betas[time] / (1 - self.alphas_cumprod[time]).sqrt())
        mean = (x_inp - scaled_pred_noise) / (1.0 - self.betas[time]).sqrt()

        return mean + scaled_noise

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: torch.nn.Module,
        num_samples: int,
        size: Tuple[int, int],
        context: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Algorithm 2 (including returning all images)"""
        # start from pure noise (for each example in the batch)
        imgs = []
        img = torch.randn((num_samples, 3, *size), device=self.device)

        for i in tqdm((range(self.timesteps, 0, -1)), desc="Sampling loop", total=self.timesteps):
            time = torch.tensor([i], device=self.device)
            pred_noise = model(img, time / self.timesteps, context)
            img = self._p_sample(img, time, pred_noise)
            imgs.append(img.cpu().numpy())
        return np.stack(imgs)
