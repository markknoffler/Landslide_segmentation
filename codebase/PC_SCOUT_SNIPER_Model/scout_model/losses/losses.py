from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        return self.l1(pred_rgb, real_rgb)


def gradient_penalty(critic: nn.Module, dem: torch.Tensor, real_rgb: torch.Tensor, fake_rgb: torch.Tensor) -> torch.Tensor:
    b = real_rgb.size(0)
    epsilon = torch.rand(b, 1, 1, 1, device=real_rgb.device)
    interpolated = epsilon * real_rgb + (1.0 - epsilon) * fake_rgb.detach()
    interpolated.requires_grad_(True)

    d_interpolated = critic(dem, interpolated)

    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(b, -1)
    grad_norm = grad.norm(2, dim=1)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty


def critic_loss(critic: nn.Module, dem: torch.Tensor, real_rgb: torch.Tensor, fake_rgb: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
    d_real = critic(dem, real_rgb)
    d_fake = critic(dem, fake_rgb.detach())
    gp = gradient_penalty(critic, dem, real_rgb, fake_rgb)
    return d_fake.mean() - d_real.mean() + lambda_gp * gp


def generator_loss(critic: nn.Module, dem: torch.Tensor, fake_rgb: torch.Tensor) -> torch.Tensor:
    d_fake = critic(dem, fake_rgb)
    return -d_fake.mean()
