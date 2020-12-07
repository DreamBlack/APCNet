import torch
import torch.nn as nn
from image_encoder import image_encoder
from pointcloud_encoder import pointcloud_encoder
from decoder import decoder
class myNet(nn.Module):
    def __init__(self,c,h,w, mlp_dims, fc_dims,grid_dims, Folding1_dims,
                 Folding2_dims, Weight1_dims, Weight3_dims,dropout=0,folding=1,attention=1):
        assert (mlp_dims[-1] == fc_dims[0])
        super(myNet, self).__init__()
        self.folding = folding
        self.attention=attention
        self.dropout=dropout
        self.image_encoder = image_encoder(c,h,w,self.attention)
        if attention==1:
            print("Use attention in image encoder")
        else:
            print("Do not use attention in image encoder")
        if folding==1:
            print("Use folding as decoder")
        else:
            print("Use fc as decoder")
        self.pointcloud_encoder=pointcloud_encoder(mlp_dims, fc_dims)
        self.get_sig_weight=nn.Sequential(
            nn.Linear(512 * 2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(True),
            nn.Sigmoid()
        )
        self.feature_fusion=nn.Sequential(
            nn.Linear(1024 , 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.decoder=decoder(grid_dims, Folding1_dims,
                 Folding2_dims, Weight1_dims, Weight3_dims)
        self.fc_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(dropout),
            #nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            #nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 256 * 3)
        )

    def forward(self, pc,image):
        # B*N*3,B*H*W
        image_feature=self.image_encoder.forward(image) # B*1*C
        pc_feature=self.pointcloud_encoder.forward(pc) # B*1*c

        cat_feature = torch.cat((image_feature, pc_feature), 1)  # b*2*(c)
        fc_cat_feature = cat_feature.view(cat_feature.size(0), 1024)
        sig_weight=self.get_sig_weight(fc_cat_feature)

        sig_weight=sig_weight.unsqueeze(2)
        cat_feature = torch.mul(sig_weight, cat_feature)
        fusion_feature = cat_feature.view(cat_feature.size(0), 1024)
        fusion_feature=self.feature_fusion(fusion_feature)
        if self.folding==1:
            fusion_feature = fusion_feature.unsqueeze(1)
            f, _ = self.decoder.forward(fusion_feature)
        else:
            f = self.fc_decoder(fusion_feature)
            f = f.view(-1, 256, 3)
        return f


if __name__ == '__main__':
    b, n = 64, 1024
    pc = torch.ones((b, n, 3))
    image=torch.ones((b,3,128,128))
    MLP_dims = (3, 64, 64, 64, 128, 1024)
    FC_dims = (1024, 512, 512)
    grid_dims = (16, 16)  # defaul 45
    Folding1_dims = (514, 512, 512, 3)  # for foldingnet
    Folding2_dims = (515, 512, 512, 3)  # for foldingnet
    Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
    Weight3_dims = (512 + 128, 1024, 1024, 256)
    knn = 48
    sigma = 0.008
    model = myNet(3,128,128,MLP_dims, FC_dims,grid_dims,Folding1_dims,Folding2_dims,Weight1_dims,Weight3_dims,folding=0,attention=0)
    print("input_size")
    print(pc.shape,image.shape)
    out = model(pc,image)
    print(out.shape)