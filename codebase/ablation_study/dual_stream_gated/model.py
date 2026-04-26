import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )


class GateFuse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        alpha = torch.sigmoid(self.gate(torch.cat([a, b], dim=1)))
        fused = alpha * a + (1.0 - alpha) * b
        reg = torch.mean(alpha * (1.0 - alpha))
        return fused, reg


class SubPixelUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * (scale**2), kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels * (scale**2))
        self.act = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = self.norm(x)
        x = self.act(x)
        return self.shuffle(x)


class TransUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.up = SubPixelUp(in_channels, out_channels)
        self.q_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.refine = DoubleConv(out_channels, out_channels)

    def forward(self, d: torch.Tensor, s: torch.Tensor):
        d_up = self.up(d)
        q = self.q_proj(d_up)
        k = self.k_proj(s)
        v = self.v_proj(s)

        b, c, h, w = q.shape
        q_tokens = q.flatten(2).transpose(1, 2)
        k_tokens = k.flatten(2).transpose(1, 2)
        v_tokens = v.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(q_tokens, k_tokens, v_tokens)
        attn_out = attn_out + self.ffn(attn_out)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return self.refine(attn_out)


class UpFlex(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = SubPixelUp(in_channels, out_channels)
        self.gate_g = nn.Conv2d(out_channels, skip_channels, kernel_size=1)
        self.gate_x = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        self.out = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, d: torch.Tensor, s: torch.Tensor):
        d_up = self.up(d)
        alpha = torch.sigmoid(self.gate_g(d_up) + self.gate_x(s))
        s_gated = alpha * s
        return self.out(torch.cat([s_gated, d_up], dim=1))


class SharedTinyEncoder(nn.Module):
    """
    Lightweight fallback encoder with 5 pyramid scales.
    Uses shared weights for both streams in siamese setup.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stage1 = DoubleConv(in_channels, c)          # 1x
        self.stage2 = DoubleConv(c, c * 2)                # 1/2x
        self.stage3 = DoubleConv(c * 2, c * 4)            # 1/4x
        self.stage4 = DoubleConv(c * 4, c * 8)            # 1/8x
        self.stage5 = DoubleConv(c * 8, c * 16)           # 1/16x
        self.pool = nn.MaxPool2d(2, 2)
        self.channels = [c, c * 2, c * 4, c * 8, c * 16]

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool(f1))
        f3 = self.stage3(self.pool(f2))
        f4 = self.stage4(self.pool(f3))
        f5 = self.stage5(self.pool(f4))
        return [f1, f2, f3, f4, f5]


class DualStreamGateNet(nn.Module):
    def __init__(self, in_channels_a: int = 3, in_channels_b: int = 3, base_channels: int = 32):
        super().__init__()
        self.adapter_a = nn.Conv2d(in_channels_a, 3, kernel_size=1)
        self.adapter_b = nn.Conv2d(in_channels_b, 3, kernel_size=1)

        self.shared_encoder = SharedTinyEncoder(in_channels=3, base_channels=base_channels)
        c1, c2, c3, c4, c5 = self.shared_encoder.channels

        self.early_fuse3 = GateFuse(c3)
        self.early_fuse4 = GateFuse(c4)

        self.decA4 = TransUp(c5, c4, c4)
        self.decA3 = UpFlex(c4, c3, c3)
        self.decA2 = UpFlex(c3, c2, c2)
        self.decA1 = UpFlex(c2, c1, c1)

        self.decB4 = TransUp(c5, c4, c4)
        self.decB3 = UpFlex(c4, c3, c3)
        self.decB2 = UpFlex(c3, c2, c2)
        self.decB1 = UpFlex(c2, c1, c1)

        self.late_fuse4 = GateFuse(c4)
        self.late_fuse3 = GateFuse(c3)

        self.merge = DoubleConv(c1 + c3 + c4, c1)
        self.main_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux2_head = nn.Conv2d(c3, 1, kernel_size=1)
        self.aux3_head = nn.Conv2d(c4, 1, kernel_size=1)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        xa = self.adapter_a(xa)
        xb = self.adapter_b(xb)

        fa = self.shared_encoder(xa)
        fb = self.shared_encoder(xb)
        a1, a2, a3, a4, a5 = fa
        b1, b2, b3, b4, b5 = fb

        f3, reg3 = self.early_fuse3(a3, b3)
        f4, reg4 = self.early_fuse4(a4, b4)

        x4a = self.decA4(a5, f4)
        x3a = self.decA3(x4a, f3)
        x2a = self.decA2(x3a, a2)
        x1a = self.decA1(x2a, a1)

        x4b = self.decB4(b5, b4)
        x3b = self.decB3(x4b, b3)
        x2b = self.decB2(x3b, b2)
        x1b = self.decB1(x2b, b1)

        lf4, reg_l4 = self.late_fuse4(x4a, x4b)
        lf3, reg_l3 = self.late_fuse3(x3a, x3b)

        lf4_up = F.interpolate(lf4, size=x1a.shape[-2:], mode="bilinear", align_corners=False)
        lf3_up = F.interpolate(lf3, size=x1a.shape[-2:], mode="bilinear", align_corners=False)
        x = self.merge(torch.cat([x1a, lf3_up, lf4_up], dim=1))

        main = self.main_head(x)
        aux2 = self.aux2_head(lf3)
        aux3 = self.aux3_head(lf4)

        return {
            "main": main,
            "aux2": aux2,
            "aux3": aux3,
            "gate_regs": [reg3, reg4, reg_l4, reg_l3],
        }
