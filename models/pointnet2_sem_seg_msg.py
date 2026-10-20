"""PointNet++ MSG + BLT/BGU/BDP; input XYZ is in metres"""
from torch import nn
from torch.nn import functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation


class get_model(nn.Module):
    """PointNet++ SemSeg (MSG + BLT)"""
    def __init__(self, num_classes):
        super().__init__()
        # Multi-scale radii and channels from Table 1.
        self.sa1 = PointNetSetAbstractionMsg(1024, [2., 6.], [16, 32], 3,
                                            [[16, 16, 32], [32, 32, 64]], use_blt=True)
        self.sa2 = PointNetSetAbstractionMsg(256, [6., 10.], [16, 32], 96,
                                            [[64, 64, 128], [64, 96, 128]], use_blt=True)
        self.sa3 = PointNetSetAbstractionMsg(64, [10., 20.], [16, 32], 256,
                                            [[128, 196, 256], [128, 196, 256]], use_blt=True)
        self.sa4 = PointNetSetAbstractionMsg(16, [20., 40.], [16, 32], 512,
                                            [[256, 256, 512], [256, 384, 512]], use_blt=True)
        # Retain the four PointNet++ feature-propagation layers.
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # Geometry-only input; SA computes normals and other descriptors.
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.conv2(self.drop1(F.relu(self.bn1(self.conv1(l0_points)))))
        return F.log_softmax(x, dim=1).permute(0, 2, 1), l4_points, None


class get_loss(nn.Module):
    def forward(self, pred, target, trans_feat=None, weight=None):
        # Weighted cross entropy, Eq. (10).
        # log_softmax + NLL is equivalent to cross entropy.
        return F.nll_loss(pred, target, weight=weight)
