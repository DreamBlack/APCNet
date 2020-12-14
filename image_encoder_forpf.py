import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from image_encoder import attention_block,conv_bn_relu
class image_encoder(nn.Module):
    def __init__(self,c=3,h=128,w=128,attention=1):
        super(image_encoder,self).__init__()
        self.attention=attention
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
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1920),
            nn.BatchNorm1d(1920),
            nn.ReLU()
        )


    def forward(self, x):

        # 3*128*128
        n=x.shape[0]
        if self.attention==1:
            x = self.conv_1(x)
            x = self.attention_1(x)

            x = self.conv_2(x)
            x = self.attention_2(x)

            x = self.conv_3(x)
            x = self.attention_3(x)

            x = self.conv_4(x)
            x = self.attention_4(x)

            x = self.conv_5(x)
            x = self.attention_5(x)
        else:
            x = self.conv_1(x)

            x = self.conv_2(x)

            x = self.conv_3(x)

            x = self.conv_4(x)

            x = self.conv_5(x)

        x=x.view(x.size(0),512*4*4)

        x=self.fc_resize(x) # N*512
        x = x.view(n, 1, -1)  # N*1*512
        return x

if __name__ == '__main__':
    n,c,h,w=64,32,128,128
    x = torch.ones((n,c,h,w))
    model=image_encoder(32,128,128,attention=0)
    out=model(x)
    print(out.shape)