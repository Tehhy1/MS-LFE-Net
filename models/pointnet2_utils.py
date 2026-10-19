import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from models.blt import BLTModule

def timeit(tag, t):
    print('{}: {}s'.format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
        Calculate Euclid distance between each two points.

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
       Input:
           points: input points data, [B, N, C]
           idx: sample index data, [B, S]
       Return:
           new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """ Greedy FPS, with no repeated row indices even for coincident points."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 10000000000.0
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """ Nearest points within a metric radius; padding is marked invalid."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
        Input:
            npoint:
            radius:
            nsample:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, nsample, 3]
            new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return (new_xyz, new_points, grouped_xyz, fps_idx)
    else:
        return (new_xyz, new_points)


def sample_and_group_all(xyz, points):
    """
       Input:
           xyz: input points position data, [B, N, 3]
           points: input points data, [B, N, D]
       Return:
           new_xyz: sampled points position data, [B, 1, 3]
           new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points)


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return (new_xyz, new_points)


# Per-point geometry from 16 nearest neighbors.
@torch.no_grad()
def point_geometry(xyz, k=16):
    # Fixed-size KNN neighborhoods.
    distances = square_distance(xyz, xyz)
    distances.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    neighbors = index_points(xyz, distances.topk(k, largest=False).indices)
    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / k
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # The smallest-eigenvalue eigenvector is the normal.
    normals = eigenvectors[..., 0]
    direction = normals.abs().argmax(dim=-1, keepdim=True)
    normals = normals * normals.gather(-1, direction).sign()
    # Curvature and height range.
    curvature = eigenvalues[..., :1] / (eigenvalues.sum(-1, keepdim=True) + 1e-8)
    height = neighbors[..., 2].amax(-1, keepdim=True) - neighbors[..., 2].amin(-1, keepdim=True)
    return torch.cat([normals, curvature, height], dim=-1)


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, use_blt=False):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.use_blt = use_blt
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.blt_blocks = nn.ModuleList()
        if use_blt:
            # Sigmoid(0)=0.5：Per-scale dual-stream mixing starts at 0.5.
            self.theta_logits = nn.Parameter(torch.zeros(len(radius_list)))
        for mlp in mlp_list:
            if use_blt:
                # Dual-branch attention and BGU at each scale.
                self.blt_blocks.append(BLTModule(14, in_channel + 3, mlp[-1]))
            else:
                convs, bns = nn.ModuleList(), nn.ModuleList()
                last_channel = in_channel + 3
                for out_channel in mlp:
                    convs.append(nn.Conv2d(last_channel, out_channel, 1))
                    bns.append(nn.BatchNorm2d(out_channel))
                    last_channel = out_channel
                self.conv_blocks.append(convs)
                self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        geometry = point_geometry(xyz) if self.use_blt else None
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            group_idx = query_ball_point(radius, self.nsample_list[i], xyz, new_xyz)
            neighbors = index_points(xyz, group_idx)
            centers = new_xyz.unsqueeze(2).expand_as(neighbors)
            relative_xyz = neighbors - centers
            grouped_points = index_points(points, group_idx) if points is not None else neighbors[..., :0]
            semantic = torch.cat([grouped_points, relative_xyz], dim=-1).permute(0, 3, 2, 1)
            if self.use_blt:
                attributes = index_points(geometry, group_idx)
                # 14-D: center/neighbor/relative XYZ, normal, curvature, height.
                geometric = torch.cat([centers, neighbors, relative_xyz, attributes], dim=-1).permute(0, 3, 2, 1)
                fused = self.blt_blocks[i](geometric, semantic)
                # Curvature-based dual-stream pooling, Eqs. (8)-(9).
                curvature = attributes[..., 3:4].permute(0, 3, 2, 1)
                cmin = curvature.amin(dim=2, keepdim=True)
                cmax = curvature.amax(dim=2, keepdim=True)
                w = (curvature - cmin) / (cmax - cmin + 1e-8)
                boundary = (fused * w).sum(2) / (w.sum(2) + 1e-8)
                interior = (fused * (1 - w)).sum(2) / ((1 - w).sum(2) + 1e-8)
                theta = self.theta_logits[i].sigmoid()
                new_points = (1 - theta) * boundary + theta * interior
            else:
                for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                    semantic = F.relu(bn(conv(semantic)))
                new_points = torch.max(semantic, 2)[0]
            new_points_list.append(new_points)
        return new_xyz.permute(0, 2, 1), torch.cat(new_points_list, dim=1)


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
            Input:
                xyz1: input points position data, [B, C, N]
                xyz2: sampled input points position data, [B, C, S]
                points1: input points data, [B, D, N]
                points2: input points data, [B, D, S]
            Return:
                new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = (dists[:, :, :3], idx[:, :, :3])
            dist_recip = 1.0 / (dists + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
