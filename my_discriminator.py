
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pointcloud_encoder import pointcloud_encoder
class myDiscriminator(nn.Module):
    def __init__(self,pc_num,mlp_dims, fc_dims):
        super(myDiscriminator,self).__init__()
        self.pc_num=pc_num
        self.pc_encoder=pointcloud_encoder(mlp_dims, fc_dims)
        self.fc_resize = nn.Sequential(
            nn.Linear(512 , 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self,x): # X:b*N*3
        x = self.pc_encoder.forward(x)  # B*1*512
        x = x.squeeze(1)
        x = self.fc_resize(x)  # N*1
        return x

if __name__ == '__main__':
    b,n=64,256
    x = torch.ones((b,n,3))
    MLP_dims = (3, 64, 64, 64, 128, 1024)
    FC_dims = (1024, 512, 512)
    model=myDiscriminator(n,MLP_dims,FC_dims)
    out=model(x)
