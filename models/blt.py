
import torch
from torch import nn
from torch.nn import functional as F


class LocalSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        attention_dim = max(1, channels // 4)
        self.q = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.k = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, valid_mask=None):
        # x [B,C,K,S]; attention independently within each spatial group S.
        q = self.q(x).permute(0, 3, 2, 1)
        k = self.k(x).permute(0, 3, 1, 2)
        v = self.v(x).permute(0, 3, 2, 1)
        scores = torch.matmul(q, k) / (q.shape[-1] ** 0.5)
        if valid_mask is not None:
            keys = valid_mask[:, 0].permute(0, 2, 1)
            scores = scores.masked_fill(~keys.unsqueeze(-2), -torch.inf)
        out = torch.matmul(F.softmax(scores, dim=-1), v).permute(0, 3, 2, 1)
        return out if valid_mask is None else out * valid_mask


class BLTModule(nn.Module):
    """Two branch local attention, scalar BGU and output residual H+X."""
    def __init__(self, in_geo_channels=14, in_sem_channels=128,
                 out_channels=128, use_bgu=True):
        super().__init__()
        self.use_bgu = use_bgu
        def mlp(channels):
            return nn.Sequential(nn.Conv2d(channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))
        self.geo_mlp = mlp(in_geo_channels)
        self.sem_mlp = mlp(in_sem_channels)
        self.geo_attn = LocalSelfAttention(out_channels)
        self.sem_attn = LocalSelfAttention(out_channels)
        if use_bgu:
            self.gate = nn.Sequential(nn.Conv2d(2 * out_channels, 1, 1), nn.Sigmoid())

    def forward(self, grouped_geometry, grouped_feat, valid_mask=None):
        geo = self.geo_mlp(grouped_geometry)
        x = self.sem_mlp(grouped_feat)
        hg = self.geo_attn(geo, valid_mask)
        hs = self.sem_attn(x, valid_mask)
        if self.use_bgu:
            beta = self.gate(torch.cat([hg, hs], dim=1))
            h = beta * hg + (1 - beta) * hs
        else:
            # Explicit ablation convention, not an alternative full model.
            h = 0.5 * (hg + hs)
        return h + x


class BoundaryDualStreamPooling(nn.Module):
    """(1-theta)*boundary_mean + theta*nonboundary_mean."""
    def __init__(self):
        super().__init__()
        self.theta_logit = nn.Parameter(torch.zeros(()))

    @property
    def theta(self):
        return self.theta_logit.sigmoid()

    def forward(self, features, curvature, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(curvature, dtype=torch.bool)
        mask = valid_mask.to(features.dtype)
        cmin = curvature.masked_fill(~valid_mask, torch.inf).amin(dim=2, keepdim=True)
        cmax = curvature.masked_fill(~valid_mask, -torch.inf).amax(dim=2, keepdim=True)
        span = cmax - cmin
        eps = torch.finfo(features.dtype).eps
        w = torch.where(span > eps, (curvature - cmin) / span.clamp_min(eps),
                        torch.full_like(curvature, 0.5))
        mean = (features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        def weighted_mean(weight):
            weight = weight * mask
            denom = weight.sum(dim=2)
            value = (features * weight).sum(dim=2) / denom.clamp_min(eps)
            return torch.where(denom > eps, value, mean)
        boundary = weighted_mean(w)
        interior = weighted_mean(1 - w)
        return (1 - self.theta) * boundary + self.theta * interior
