import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_bn_relu(nn.Module):  # con->bn->relu
    def __init__(self,ch_in,ch_out,kernel_size=3,stride=1,padding=1): # padding=1可以使得输入和输出大小一样，ｐａｄｄｉｎｇ＝２大小减半
        super(conv_bn_relu,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding), # inputchannel,out_channel,padding=1 才能让输出和输入大小不变
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x=self.conv(x)
        return x

class attention_block(nn.Module): # attention_block
    def __init__(self,ch_in,r=16,i=2):
        super(attention_block,self).__init__()
        self.channel_attention=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(ch_in,ch_in//r,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // r, ch_in, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.spatial_attention=nn.Sequential(
            nn.Conv2d(ch_in,ch_in*i,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in*i, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.last_conv=nn.Sequential(
            nn.Conv2d(ch_in*2,ch_in,kernel_size=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.last_relu=nn.ReLU(inplace=True)

    def forward(self,x):
        channel_feature=self.channel_attention(x)
        channel_feature=torch.mul(channel_feature,x)

        spatial_feature=self.spatial_attention(x)
        spatial_feature = torch.mul(spatial_feature, x)

        out=torch.cat([channel_feature,spatial_feature],dim=1)
        out=self.last_conv(out)

        out=torch.add(x,out)
        out=self.last_relu(out)
        return out

class image_encoder(nn.Module):
    def __init__(self,c,h,w):
        super(image_encoder,self).__init__()
        self.conv_1=nn.Sequential(
            conv_bn_relu(c,32),
            conv_bn_relu(32,32),
            conv_bn_relu(32,64,stride=2)
            )
        self.attention_1 = attention_block(64)
        self.conv_2 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 128, stride=2)
        )
        self.attention_2 = attention_block(128)
        self.conv_3 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 256, stride=2)
        )
        self.attention_3 = attention_block(256)
        self.conv_4 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 512, stride=2)
        )
        self.attention_4 = attention_block(512)
        self.conv_5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512,kernel_size=5, padding=2,stride=2)
        )
        self.attention_5 = attention_block(512)
        self.fc_resize=nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )


    def forward(self, x):

        # 3*128*128
        n=x.shape[0]
        x=self.conv_1(x)
        x=self.attention_1(x)

        x = self.conv_2(x)
        x = self.attention_2(x)

        x = self.conv_3(x)
        x = self.attention_3(x)

        x = self.conv_4(x)
        x = self.attention_4(x)

        x = self.conv_5(x)
        x = self.attention_5(x)

        x=x.view(x.size(0),512*4*4)

        x=self.fc_resize(x) # N*512
        x=x.view(n,1,-1)  # N*1*512
        return x

if __name__ == '__main__':
    n,c,h,w=64,32,128,128
    x = torch.ones((n,c,h,w))
    model=image_encoder(32,128,128)
    out=model(x)
    print(out.shape)