"""CardioOncoNet: 12-lead 1D-ResNet + clinical metadata, fused by a bilinear (tensor) product.

Architecture
------------
ECG branch   x in R^{12 x T}  ->  1D residual CNN  ->  v_e in R^{d_e}
Meta branch  m = [age_z, sex, age_missing]  ->  MLP  ->  v_m in R^{d_m}
Fusion       z = vec( [v_e; 1] (x) [v_m; 1] )  in R^{(d_e+1)(d_m+1)}
Head         logits = W z + b,   p = sigmoid(logits)   (multi-label)

Appending a constant 1 to each vector before the outer product (Zadeh et al.,
"Tensor Fusion Network", EMNLP 2017) makes z contain three blocks at once:
    v_e (x) v_m   -- bimodal interactions   (e.g. "flat T-wave AND age 70")
    v_e * 1       -- ECG-only terms
    1 * v_m       -- metadata-only terms
    1             -- bias
so the fused model can never be worse than either branch alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 7, p_drop: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Identity() if (c_in == c_out and stride == 1) else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ECGEncoder(nn.Module):
    def __init__(self, n_leads: int = 12, width: int = 32, d_out: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, w, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResBlock1d(w, w),
            ResBlock1d(w, 2 * w, stride=2),
            ResBlock1d(2 * w, 2 * w),
            ResBlock1d(2 * w, 4 * w, stride=2),
            ResBlock1d(4 * w, 4 * w, stride=2),
        )
        self.proj = nn.Linear(8 * w, d_out)   # concat(avg-pool, max-pool) -> d_out

    def forward(self, x):
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return torch.relu(self.proj(h))


class MetaEncoder(nn.Module):
    def __init__(self, d_in: int = 3, d_out: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(inplace=True),
                                 nn.Linear(32, d_out), nn.ReLU(inplace=True))

    def forward(self, m):
        return self.net(m)


class BilinearFusion(nn.Module):
    def __init__(self, d_e: int, d_m: int, n_out: int, p_drop: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear((d_e + 1) * (d_m + 1), n_out)

    def forward(self, v_e, v_m):
        one = v_e.new_ones(v_e.size(0), 1)
        e = torch.cat([v_e, one], dim=1)                 # (B, d_e+1)
        m = torch.cat([v_m, one], dim=1)                 # (B, d_m+1)
        z = torch.bmm(e.unsqueeze(2), m.unsqueeze(1))    # (B, d_e+1, d_m+1) outer product
        return self.head(self.drop(z.flatten(1)))


class CardioOncoNet(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5, d_e: int = 64, d_m: int = 16,
                 width: int = 32, d_meta_in: int = 3):
        super().__init__()
        self.ecg_encoder = ECGEncoder(n_leads, width, d_e)
        self.meta_encoder = MetaEncoder(d_meta_in, d_m)
        self.fusion = BilinearFusion(d_e, d_m, n_classes)

    def forward(self, ecg, meta):
        """ecg: (B, 12, T) standardised mV;  meta: (B, 3).  Returns logits (B, n_classes)."""
        return self.fusion(self.ecg_encoder(ecg), self.meta_encoder(meta))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
