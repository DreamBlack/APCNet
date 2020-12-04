import os
import sys
import argparse
import random
import torch
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from MyDataset import MyDataset
from completion_net import myNet
from myutils import PointLoss

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')

parser.add_argument('--folding_decoder', type = bool, default = True, help='enables cuda')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/withDCkPath/Trained_Model/point_netG50.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='weight for loss 0 means do not use else use with this weight')
opt = parser.parse_args()  # 这个函数会使用上述add进来的参数，返回添加的那些参数
print(opt)

continueLast=False
resume_epoch=0
weight_decay=0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # 据说可以增加程序运行效率
resume_epoch=0
USE_CUDA=opt.cuda

# 权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

MLP_dims = (3, 64, 64, 64, 128, 1024)
FC_dims = (1024, 1024, 512)
grid_dims = (16, 16)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 128, 1024, 1024, 256)
knn = 48
sigma = 0.008
mynet = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims, Weight3_dims,folding=opt.folding_decoder)


print("Let's use", torch.cuda.device_count(), "GPUs!")
mynet = torch.nn.DataParallel(mynet)
mynet.to(device)
mynet.apply(weights_init_normal)



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


# 加载测试
test_dset = MyDataset( classification=True, class_choice=opt.class, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=False,num_workers = int(opt.workers))


mynet.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
mynet.eval()

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)



if __name__=='__main__':
    errG_min = 100
    n = 0
    test_num=len(test_dataloader)
    print_time=0
    for i, data in enumerate(test_dataloader, 0):
        incomplete, gt, image ,complet= data

        batch_size = incomplete.size()[0]

        incomplete = incomplete.to(device)

        gt = gt.to(device)
        image = image.to(device)
        incomplete = Variable(incomplete, requires_grad=True)
        image = Variable(image.float(), requires_grad=True)  # 这里图片是不是先要归一化－１，１
        image = torch.squeeze(image, 1)

        fake = mynet(incomplete, image)
        fake = fake.cuda()
        gt = gt.cuda()
        errG = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(gt, 1))  #
        errG = errG.cpu()
        if errG.detach().numpy() > errG_min:
            #    if a!=b:
            pass

        else:
            if print_time%10==0:
                errG_min = errG.detach().numpy()
                print("nownownow____%d____"%print_time)
                print(errG_min)
                all=torch.cat([fake,incomplete],dim=1)
                all=all.cpu()
                np_all=all.data[0].detach().numpy()

                fake = fake.cpu()
                np_fake = fake[0].detach().numpy()  # 256

                gt = gt.cpu()
                np_gt = gt.data[0].detach().numpy()  # 256

                incomplete = incomplete.cpu()
                np_inco = incomplete[0].detach().numpy()  # 1024

                print(np_all.shape)
                print(np_fake.shape)
                print(np_gt.shape)
               # np.savetxt('test_example_gan/all_txt' + str(print_time) + '.xyz', np_all)
            #np.savetxt('test_example_gan/fake_txt' + str(print_time) + '.xyz', np_fake)
              #  np.savetxt('test_example_gan/gt_txt' + str(print_time) + '.xyz', np_gt)

        print_time=print_time+1