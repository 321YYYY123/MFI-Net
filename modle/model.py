import torch
import torch.nn as nn

from .transformer import *
import torch.nn.functional as F


def getC(i):
    return 32 * 2**(i)


class Backbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),  #3-》32
            nn.ReLU(),
            nn.Linear(32, 32))
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim,
                                             nneighbor)  #32，512，16
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):  #4
            channel = 32 * 2**(i + 1)  # 32 64 128 256
            self.transition_downs.append(
                TransitionDown(npoints // 4**(i + 1), nneighbor,
                               [channel // 2 + 3, channel, channel]))
            self.transformers.append(
                TransformerBlock(channel, cfg.model.transformer_dim,
                                 nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]  #（N，32）

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats  #points是backBone最后的输出特征 xyz_and_feats是存了每一层输出的 xyz 和 points 特征


class ClsHeadBlock(nn.Module):

    def __init__(self, channel, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        # 特征提取网络
        self.fc = nn.Sequential(nn.Linear(channel, 256), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(64, n_c))

    def forward(self, x):
        res = self.fc(x)
        return res


class TupAndAtBlock(nn.Module):

    def __init__(self, channelin, channel, cfg):
        super().__init__()
        self.Tup = TransitionUp(channelin, channel, channel)
        self.At = TransformerBlock(channel, cfg.model.transformer_dim,
                                   cfg.model.nneighbor)

    def forward(self, xp1, xp2):
        points = self.Tup(xp1[0], xp1[1], xp2[0], xp2[1])
        points = self.At(xp2[0], points)[0]
        return xp2[0], points


class TdownAndAtBlock(nn.Module):

    def __init__(self, npoints, channel, cfg):
        super().__init__()
        # self.Tnet = TNet()
        self.Tdown = TransitionDown(npoints, cfg.model.nneighbor,
                                    [channel // 2 + 3, channel, channel])
        self.At = TransformerBlock(channel, cfg.model.transformer_dim,
                                   cfg.model.nneighbor)

    def forward(self, xp):

        # xyz = self.Tnet(xp[0])
        xyz, points = self.Tdown(xp[0], xp[1])
        points = self.At(xyz, points)[0]
        return xyz, points


class PointTransformerCls(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # self.Tnet = TNet()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.nblocks = nblocks

        self.MLS = nn.ModuleList([
            TupAndAtBlock(512, 256, cfg),
            TupAndAtBlock(256, 128, cfg),
            TupAndAtBlock(128, 64, cfg),
            TdownAndAtBlock(npoints // 16, 128, cfg),
            TupAndAtBlock(128, 128, cfg),
            TdownAndAtBlock(npoints // 64, 256, cfg),
            TupAndAtBlock(256, 256, cfg),
            TdownAndAtBlock(npoints // 256, 512, cfg),
            TupAndAtBlock(512, 512, cfg)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 2**nblocks, 256),
            nn.ReLU(),
            #  nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(256, 32 * 2**nblocks))
        self.transformer1 = TransformerBlock(32 * 2**nblocks,
                                             cfg.model.transformer_dim,
                                             nneighbor)

        self.Heads = nn.ModuleList([
            ClsHeadBlock(128, cfg),
            ClsHeadBlock(256, cfg),
            ClsHeadBlock(512, cfg)
        ])
        
        self.visFeature = []

    def forward(self, x):
        
        points, xps = self.backbone(x)  #xps xyz and feats

        xyz = xps[-1][0]

        # xyz = self.Tnet(xyz)

        points = self.transformer1(xyz, self.fc1(points))[0]

        xp0 = self.MLS[0]([xyz, points], xps[-2])

        xp1 = self.MLS[1](xp0, xps[-3])

        xp2 = self.MLS[2](xp1, xps[-4])

        xp3 = self.MLS[3](xp2)  #down

        xp4 = self.MLS[4](xp3, xp1)  #cat

        h0 = self.Heads[0](xp4[1].mean(1))  #head
        
       

        xp5 = self.MLS[5](xp4)  #down

        xp6 = self.MLS[6](xp5, xp0)  #cat

        h1 = self.Heads[1](xp6[1].mean(1))  #head
     

        xp7 = self.MLS[7](xp6)  #down

        xp8 = self.MLS[8](xp7, xps[-1])  #cat

        h2 = self.Heads[2](xp8[1].mean(1))  #head

        # print(xp4[1].shape)
        # print(xp6[1].shape)
        # print(xp8[1].shape)
        self.visFeature=[xps[-1][1],xp4[1],xp6[1],xp8[1]]

        # self.visFeature1=[h0.unsqueeze(1),h1.unsqueeze(1),h2.unsqueeze(1)]

        res = torch.cat([h0.unsqueeze(1),#（bs，classnum）->（bs，1，classnum）
                         h1.unsqueeze(1),
                         h2.unsqueeze(1)],
                        dim=1).mean(1)

        return res

