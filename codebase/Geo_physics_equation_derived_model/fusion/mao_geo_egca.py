import torch
import torch.nn as nn
import torch.nn.functional as F


class MAOGeoEGCA(nn.Module):
    """Manifold-Aligned Orthogonal Geo-Equilibrium Gated Cross-Attention."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.physics_blend = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_project = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, t_fm: torch.Tensor, x_rgb: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        b, c, h, w = t_fm.shape
        n = h * w

        x_physics = self.physics_blend(torch.cat([x_rgb, x_dem], dim=1))
        gate = self.gate_net(t_fm * x_physics)

        q = self.q_proj(x_physics).view(b, c, -1).transpose(1, 2)
        k = self.k_proj(t_fm).view(b, c, -1).transpose(1, 2)
        v = self.v_proj(t_fm).view(b, c, -1).transpose(1, 2)

        q_norm = F.normalize(q, p=2, dim=-1)
        k_proj = k * q_norm

        q_h = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k_proj.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_h)
        context = context.transpose(1, 2).contiguous().view(b, c, h, w)

        return self.out_project(context * gate) + x_physics
