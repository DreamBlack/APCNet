import torch.nn as nn
import torch.nn.functional as F
import torch
from PointNetPlus_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNetPlusEncoder(nn.Module):
    def __init__(self):
        super(PointNetPlusEncoder, self).__init__()
        #xyz: input points position data, [B, C, N]
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 512], group_all=True)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        norm = None
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 512)
        l2_points=self.pool(l2_points) # B*256*128->B*256*1
        l2_points = l2_points.permute(0, 2, 1) #B*256*1->B*1*256
        l3_points = x.unsqueeze(1)  # B*1*512
        cat_feature = torch.cat((l2_points, l3_points), 2)  # b*1*(768)
        x = cat_feature.view(B, 768)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x) #B*512
        l3_points = x.unsqueeze(1)  # B*1*512
        return l3_points


if __name__ == '__main__':
    b, n = 64, 1024
    pc = torch.ones((b, n, 3))

    model = PointNetPlusEncoder()
    print("input_size")
    print(pc.shape)
    l3_points = model(pc)
    print(l3_points.shape)
