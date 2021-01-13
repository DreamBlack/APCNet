import torch
import torch.nn as nn
from pointcloud_encoder import pointcloud_encoder
from decoder import decoder

class pcnFoldingNet(nn.Module):
    def __init__(self,mlp_dims, fc_dims,grid_dims, Folding1_dims,
                 Folding2_dims, Weight1_dims, Weight3_dims):
        assert (mlp_dims[-1] == fc_dims[0])
        super(pcnFoldingNet, self).__init__()
        self.pointcloud_encoder=pointcloud_encoder(mlp_dims, fc_dims)

        self.decoder=decoder(grid_dims, Folding1_dims,
                 Folding2_dims, Weight1_dims, Weight3_dims)


    def forward(self, pc):
        # B*N*3,B*H*W
        pc_feature=self.pointcloud_encoder.forward(pc) # B*1*c
        f, _ = self.decoder.forward(pc_feature)
        return f


if __name__ == '__main__':
    b, n = 64, 1024

    pc = torch.ones((b, n, 3))
    MLP_dims = (3, 64, 64, 64, 128, 1024)
    FC_dims = (1024, 512, 512)
    grid_dims = (45, 45)  # defaul 45
    Folding1_dims = (514, 512, 512, 3)  # for foldingnet
    Folding2_dims = (515, 512, 512, 3)  # for foldingnet
    Weight1_dims = (45 * 45 + 512, 512, 512, 512)  # for weight matrix estimation 45x45+512 = 2537
    Weight3_dims = (512 + 512, 1024, 1024, 2025)
    knn = 48
    sigma = 0.008
    model = pcnFoldingNet(MLP_dims, FC_dims,grid_dims,Folding1_dims,Folding2_dims,Weight1_dims,Weight3_dims)
    print("input_size")
    print(pc.shape)
    out = model(pc)
    print(out.shape)
