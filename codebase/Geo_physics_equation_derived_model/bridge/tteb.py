import torch
import torch.nn as nn
import torch.nn.functional as F

class TriTemporalTriStreamBridge(nn.Module):
    """Combine prev/present/next features across rgb, dem, and fm pyramids."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.anchor = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.context_mix = nn.Sequential(
            nn.Conv2d(channels * 8, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.stream_phase = nn.Parameter(torch.zeros(3, channels))
        self.temporal_router = nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.stability_r = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.stability_d = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.psi = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        # Chunked attention avoids materializing full (H*W)^2 scores at 256x256 (~64 GiB per sample).
        self.attn_chunk_size = 4096

    def _spatial_attention(
        self,
        q_h: torch.Tensor,
        k_h: torch.Tensor,
        v_h: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.head_dim**-0.5
        _, _, n, _ = q_h.shape
        chunk = self.attn_chunk_size
        if n <= chunk:
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v_h)

        out_chunks = []
        for start in range(0, n, chunk):
            q_chunk = q_h[:, :, start : start + chunk, :]
            scores = torch.matmul(q_chunk, k_h.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            out_chunks.append(torch.matmul(attn, v_h))
        return torch.cat(out_chunks, dim=2)

    @staticmethod
    def _gather_level(pyramid: list[torch.Tensor], level: int) -> torch.Tensor:
        level = max(0, min(level, len(pyramid) - 1))
        return pyramid[level]

    @staticmethod
    def _align(feat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if feat.shape[-2:] == size:
            return feat
        return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        p_rgb: list[torch.Tensor],
        p_dem: list[torch.Tensor],
        t_fm: list[torch.Tensor],
        level: int,
    ) -> torch.Tensor:
        streams = [p_rgb, p_dem, t_fm]
        present = [self._gather_level(s, level) for s in streams]
        size = present[0].shape[-2:]
        prev = [self._align(self._gather_level(s, level - 1), size) for s in streams]
        nxt = [self._align(self._gather_level(s, level + 1), size) for s in streams]
        present = [self._align(p, size) for p in present]

        anchor = self.anchor(torch.cat(present, dim=1))

        # Eight off-diagonal nodes: prev/next for each stream (6) + two cross-stream present refs
        off_diag = prev + nxt + [present[0], present[2]]
        context = self.context_mix(torch.cat(off_diag, dim=1))

        r = F.softplus(self.stability_r(anchor))
        d = F.softplus(self.stability_d(anchor))
        delta = torch.abs(self.psi - r / (d + 1e-6))

        b, c, h, w = anchor.shape
        n = h * w
        q = self.q_proj(anchor).view(b, c, -1).transpose(1, 2)
        k = self.k_proj(context).view(b, c, -1).transpose(1, 2)
        v = self.v_proj(context).view(b, c, -1).transpose(1, 2)

        phase = self.stream_phase.mean(dim=0).view(1, 1, c)
        k = k + phase

        q_h = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        out = self._spatial_attention(q_h, k_h, v_h).transpose(1, 2).contiguous().view(b, c, h, w)

        tau_w = torch.softmax(self.temporal_router(delta), dim=1).mean(dim=(2, 3), keepdim=True)
        out = out * (1.0 + tau_w[:, 1:2])

        return anchor + self.out(out * torch.sigmoid(delta))
