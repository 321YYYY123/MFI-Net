from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction


class TransformerBlock(nn.Module):

    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(),
                                      nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        # self.attn_dropout = nn.Dropout(0.1)
        # self.proj_dropout = nn.Dropout(0.1)

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(
            xyz, knn_idx)  # b x n x 3,   # b x n x k -》[B, n, [K], 3]

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x),
                                             knn_idx), index_points(
                                                 self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre

        return res, attn


class TransitionDown(nn.Module):

    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k,
                                         0,
                                         nneighbor,
                                         channels[0],
                                         channels[1:],
                                         group_all=False,
                                         knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):

    def __init__(self, dim1, dim2,
                 dim_out):  #设置点1 和 点2 的输入channel 和输出的 channel

        class SwapAxes(nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None,
                         feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


###Tnet相关#################
class Conv1dBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes):
        super(FCBNReLU,
              self).__init__(nn.Linear(in_planes, out_planes, bias=False),
                             nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))


class TNet(nn.Module):

    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = nn.Sequential(Conv1dBNReLU(3, 64),
                                     Conv1dBNReLU(64, 128),
                                     Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(FCBNReLU(256, 128), FCBNReLU(128, 64),
                                     nn.Linear(64, 6))

    @staticmethod
    def f2R(f):
        r1 = F.normalize(f[:, :3])  #单位化
        proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
        r2 = F.normalize(f[:, 3:] - proj * r1)
        r3 = r1.cross(r2)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, pts):
        pts = pts.transpose(1, 2)
        f = self.encoder(pts)  # #（bs，3，1024）
        f, _ = f.max(dim=2)
        f = self.decoder(f)
        R = self.f2R(f)
        res = R @ pts  #（bs，3，1024）
        res = res.transpose(1, 2)  #（bs,1024,3）
        return res
