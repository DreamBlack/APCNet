import torch
import torch.nn as nn
from pointnet import PointNetVanilla
class pointcloud_encoder(nn.Module):
    def __init__(self,mlp_dims,fc_dims,mlp_dolastReLU=False):
        assert(mlp_dims[-1]==fc_dims[0])
        super(pointcloud_encoder,self).__init__()
        self.PointNet=PointNetVanilla(mlp_dims,fc_dims,mlp_dolastReLU)

    def forward(self,x):
        # B*N*3
        f=self.PointNet.forward(x)  # B*K
        f=f.unsqueeze(1)  # B*1*512
        return f


if __name__ == '__main__':
    b,n=64,1024
    x = torch.ones((b,n,3))
    MLP_dims = (3, 64, 64, 64, 128, 1024)
    FC_dims = (1024, 512, 512)
    model=pointcloud_encoder(MLP_dims,FC_dims)
    out=model(x)
    print(out.shape)