#!/usr/bin/env python3
import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import os
import torch.nn.parallel
import utils
from model_PFNet import _netG
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from MyDataset_former import MyDataset
from completion_net import myNet
from myutils import PointLoss_test
import numpy as np

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Lamp', help="which class choose to train")
parser.add_argument('--attention_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--folding_decoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--pointnetplus_encoder', type = int, default = 0, help='enables cuda')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', help="path to netG (to load as model)")
parser.add_argument('--result_path', help="path to netG (to load as model)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--four_data', type = int, default = 0, help='enables cuda')
opt = parser.parse_args()
print(opt)

print('命令行参数写入文件')
f_all=open(os.path.join(opt.result_path,'all_test_result.txt'), 'a')
f_all.write("\n"+"workers"+"  "+str(opt.workers))
f_all.write("\n"+"batchSize"+"  "+str(opt.batchSize))
f_all.write("\n"+"attention_encoder"+"  "+str(opt.attention_encoder))
f_all.write("\n"+"folding_decoder"+"  "+str(opt.folding_decoder))
f_all.write("\n"+"pointnetplus_encoder"+"  "+str(opt.pointnetplus_encoder))
f_all.write("\n"+"class_choice"+"  "+str(opt.class_choice))
f_all.write("\n"+"netG"+"  "+str(opt.netG))
f_all.write("\n"+"four_data"+"  "+str(opt.four_data))
f_all.close()

continueLast = False
resume_epoch = 0
weight_decay = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # promote
USE_CUDA = opt.cuda

def distance_squre1(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2

test_dset = MyDataset(classification=True,three=opt.folding_decoder,
                      class_choice=opt.class_choice, split='test',four_data=opt.four_data)
assert test_dset
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=opt.workers,drop_last=True)

length = len(test_dataloader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


mynet = _netG(3, 1, [1024, 512, 256], 256)
mynet.to(device)
mynet.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'],strict=False)
mynet.eval()

criterion_PointLoss = PointLoss_test().to(device)

errG_min = 100
n = 0
CD = 0
CD_ALL = 0
Gt_Pre_ALL = 0
Pre_Gt_ALL = 0
Gt_Pre = 0
Pre_Gt = 0

for i, data in enumerate(test_dataloader, 0):
    incomplete, gt, image,filename = data
    batch_size = incomplete.size()[0]
    incomplete = incomplete.to(device)

    gt = gt.to(device)
    image = image.to(device)
    complete_gt = torch.cat([gt, incomplete], dim=1)
    complete_gt=complete_gt.to(device)
    incomplete = incomplete.to(device)
    image = image.to(device)

    incomplete = Variable(incomplete, requires_grad=False)
    image = Variable(image.float(), requires_grad=False)

    ############################
    # (1) data prepare
    ###########################
    real_center = Variable(gt, requires_grad=False)
    real_center = torch.squeeze(real_center, 1)
    real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
    real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
    real_center_key1 = Variable(real_center_key1, requires_grad=False)

    real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
    real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
    real_center_key2 = Variable(real_center_key2, requires_grad=False)

    input_cropped1 = torch.squeeze(incomplete, 1)
    input_cropped2_idx = utils.farthest_point_sample(input_cropped1, 512, RAN=True)
    input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
    input_cropped3_idx = utils.farthest_point_sample(input_cropped1, 256, RAN=False)
    input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
    input_cropped1 = Variable(input_cropped1, requires_grad=False)
    input_cropped2 = Variable(input_cropped2, requires_grad=False)
    input_cropped3 = Variable(input_cropped3, requires_grad=False)
    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)
    input_cropped = [input_cropped1, input_cropped2, input_cropped3]

    fake_center1, fake_center2, fake = mynet(input_cropped,image)

    dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake, 1).to(device), torch.squeeze(gt, 1).to(device))

    dist_all = dist_all.cpu().detach().numpy()
    dist1 = dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()
    CD = CD + dist_all / length
    Gt_Pre = Gt_Pre + dist1 / length
    Pre_Gt = Pre_Gt + dist2 / length
    #print("part")
    #print(CD*1000, Gt_Pre*1000, Pre_Gt*1000) # missing

    complete_pc = torch.cat([fake, incomplete], dim=1).to(device)
    dist_all_all, dist1_all, dist2_all = criterion_PointLoss(torch.squeeze(complete_pc, 1).to(device),
                                                             torch.squeeze(complete_gt, 1).to(device))
    dist_all_all = dist_all_all.cpu().detach().numpy()
    dist1_all = dist1_all.cpu().detach().numpy()
    dist2_all = dist2_all.cpu().detach().numpy()
    CD_ALL = CD_ALL + dist_all_all / length
    Gt_Pre_ALL = Gt_Pre_ALL + dist1_all / length
    Pre_Gt_ALL = Pre_Gt_ALL + dist2_all / length
    #print("all")
    #print(CD_ALL*1000, Gt_Pre_ALL*1000, Pre_Gt_ALL*1000) # overall



print(CD, Gt_Pre, Pre_Gt)
print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD*1000), float(Gt_Pre*1000), float(Pre_Gt*1000)))
print(CD_ALL, Gt_Pre_ALL, Pre_Gt_ALL)
print("CD_ALL:{} , Gt_Pre_ALL:{} , Pre_Gt_ALL:{}".format(float(CD_ALL*1000), float(Gt_Pre_ALL*1000), float(Pre_Gt_ALL*1000)))
print(length)

#assert len(bests_missing_pregt)==bests_num

# 写入文件
print('将总值结果写入文件')
f_all=open(os.path.join(opt.result_path,'all_test_result.txt'), 'a')
f_all.write('\n'+'CD: %.4f Gt_Pre: %.4f Pre_Gt: %.4f '
                  % (float(CD*1000), float(Gt_Pre*1000), float(Pre_Gt*1000)))
f_all.write('\n'+'CD_ALL: %.4f Gt_Pre_ALL: %.4f Pre_Gt_ALL: %.4f '
                  % (float(CD_ALL*1000), float(Gt_Pre_ALL*1000), float(Pre_Gt_ALL*1000)))
f_all.close()


