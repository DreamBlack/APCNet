import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from MyDataset import MyDataset
from completion_net import myNet
from myutils import PointLoss,RepulsionLoss#,PointEMDLoss
from tensorboardX import SummaryWriter
from my_discriminator import myDiscriminator
from DatasetGeneration import farthest_point_sample,index_points,distance_squre
import time

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Lamp', help="which class choose to train")
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--fc_decoder', type = bool, default = True, help='use full connect as decoder')
parser.add_argument('--emd', type = bool, default = False, help='use emd loss')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--Rep_choose',type=int, default=0, help='0 not use Rep Loss,1 use Rep Loss')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--withDCkPath', default='withDCkPath', help="which class choose to train")
parser.add_argument('--noDCkPath', default='noDCkPath', help="which class choose to train")
parser.add_argument('--tensorboardDir', default='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/fc_gan_cd/tensorboard', help="which path to store tensorboard")
parser.add_argument('--checkpointDir',default='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/fc_gan_cd/checkpoint',  help="which path to store checkpoint")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--alpha',type=float,default=1.0, help='rep loss weight')
parser.add_argument('--radius',type=float,default=0.07, help='radius of knn')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='weight for loss 0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

continueLast=False
resume_epoch=0
weight_decay=0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True # promote
USE_CUDA=opt.cuda

# weight init
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

MLP_dimsG = (3, 64, 64, 64, 128, 1024)
FC_dimsG = (1024, 1024, 512)
MLP_dimsD = (3, 64, 64, 64, 128, 1024)
FC_dimsD = (1024, 1024, 512)
grid_dims = (16, 16)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 128, 1024, 1024, 256)
knn = 48
sigma = 0.008
myNet = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims, Weight3_dims,folding=opt.fc_decoder)
myNetD=myDiscriminator(256,MLP_dimsD,FC_dimsD)

print("Let's use", torch.cuda.device_count(), "GPUs!")
myNet = torch.nn.DataParallel(myNet)
myNet.to(device)
myNet.apply(weights_init_normal)
myNetD = torch.nn.DataParallel(myNetD)
myNetD.to(device)
myNetD.apply(weights_init_normal)

if opt.netG != '' :
    myNet.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    myNetD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# load data
dset = MyDataset(classification=True,three=opt.fc_decoder,class_choice=opt.class_choice, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers)

test_dset = MyDataset(classification=True,class_choice=opt.class_choice, split='test')
assert test_dset
test_dataloader = torch.utils.data.DataLoader(test_dset,batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)
#if opt.emd:
    #criterion_PointLoss=PointEMDLoss().to(device)
criterion_RepLoss=RepulsionLoss(alpha=opt.alpha,radius=opt.radius).to(device)

# setup optimizer
optimizerG = torch.optim.Adam(myNet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=60, gamma=0.2)
optimizerD = torch.optim.Adam(myNetD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=60, gamma=0.2)

real_label = 1
fake_label = 0
missingNum=256
incompleteNum=1024
num_batch = len(dset) / opt.batchSize
label = torch.FloatTensor(opt.batchSize)
missingNum = 256
incompleteNum = 1024
if __name__=='__main__':
    ###########################
    #  G-NET and T-NET
    ##########################
    if opt.D_choose == 1:
        for epoch in range(resume_epoch, opt.niter):
            time_start=time.time()
            for i, data in enumerate(dataloader, 0):
                now_time = time.time()
                begin=now_time
                print("get a batch:" + str(now_time - time_start))
                time_start=now_time
                real_point, image, view = data

                batch_size = real_point.size()[0]
                num_sum = missingNum + incompleteNum  # 256,1024

                # real_point采样为1280个点
                real_point = Variable(real_point, requires_grad=False).cuda()  # 只有variable才能在gpu上计算
                real_point_key1_idx = farthest_point_sample(real_point, num_sum, RAN=False)
                real_point = index_points(real_point, real_point_key1_idx)
                complete_gt = real_point
                now_time = time.time()
                print("prepare_data1,farthest_point_sample:" + str(now_time - time_start))
                time_start = now_time
                # 占位符
                missingPC = torch.FloatTensor(batch_size, missingNum, 3).cuda()
                incompletePC = torch.FloatTensor(batch_size, num_sum, 3).cuda()  # 这里还没有构成精确的incomplete大小
                incompletePC = incompletePC.data.copy_(real_point)  # post-fixed with _ 表示是原地操作
                center = view
                center=center.to(device)
                # real_point=real_point.to(device)
                # missingPC=missingPC.to(device)
                # incompletePC = incompletePC.to(device)
                for m in range(batch_size):
                    distance_list = []
                    for n in range(num_sum):
                        distance_list.append(distance_squre(real_point[m, n], center[m]))
                    distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                    # sorted返回的是重新排序的列表以及下标,nums[0]=data's index in distance_list,nums[1]=data
                    for sp in range(missingNum):
                        # distance_order[sp][0]为这个值在list里面原来的下标，也就是在realpoint中的下标
                        missingPC.data[m, sp] = real_point[m, distance_order[sp][0]]  # 离中心点进度在missing中

                    for sp in range(missingNum, num_sum):
                        incompletePC.data[m, sp] = real_point[m, distance_order[sp][0]]  # 离中心点进度在missing中

                now_time = time.time()
                print("prepare_data_split:" + str(now_time - time_start))
                time_start = now_time
                incompletePC = incompletePC.data.cpu()[:, missingNum:]

                #missingPC = missingPC.to(device)  # input complete，不过有2048个点
                #incompletePC = incompletePC.to(device)  # input incomplete,不过点的个数太多了需要点采样一下

                gt = missingPC.to(device)
                image = image.to(device)
                complete_gt = complete_gt.to(device)

                incomplete = Variable(incompletePC, requires_grad=True).cuda()
                image = Variable(image.float(), requires_grad=True)
                image = torch.squeeze(image, 1)
                label.resize_([batch_size, 1]).fill_(real_label)
                label = label.to(device)
                now_time = time.time()
                print("prepare_data:" + str(now_time - time_start))
                time_start=now_time
                myNet = myNet.train()
                myNetD=myNetD.train()
                #########################
                # update d
                ############################
                myNetD.zero_grad()
                output=myNetD(torch.squeeze(gt, 1))  # input B*N*3
                errD_real = criterion(output, label)
                errD_real.backward()

                fake = myNet(incomplete, image)
                fake = torch.squeeze(fake, 1)
                label.data.fill_(fake_label)
                output = myNetD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()
                #########################
                # update g
                ############################
                myNet.zero_grad()
                label.data.fill_(real_label)
                output = myNetD(torch.squeeze(fake, 1))
                errG_D = criterion(output, label)
                errG_l2=0
                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(gt, 1))
                RepLoss=0
                errG_l2 = CD_LOSS
                if opt.Rep_choose==1:
                    RepLoss = criterion_RepLoss(torch.squeeze(fake, 1))
                    errG_l2+=RepLoss

                errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
                errG.backward()
                optimizerG.step()
                now_time = time.time()
                print("time_train:" + str(now_time - time_start))
                time_start = now_time
                writer = SummaryWriter(log_dir=opt.tensorboardDir)
                writer.add_scalar('cd_missing', CD_LOSS, num_batch * epoch + i)
                writer.add_scalar('repulsion', RepLoss, num_batch * epoch + i)
                writer.add_scalar('D_Loss', errD.data, num_batch * epoch + i)
                writer.add_scalar('GD_Loss', errG_D.data, num_batch * epoch + i)
                writer.add_scalar('GwithD_sum_loss', errG, num_batch * epoch + i)
                writer.add_scalar('GwithD_l2', errG_l2, num_batch * epoch + i)
                complete_pc = torch.cat([fake, incomplete], dim=1)
                CD_LOSS_ALL = criterion_PointLoss(torch.squeeze(complete_pc, 1), torch.squeeze(complete_gt, 1))
                writer.add_scalar('cd_all', CD_LOSS_ALL, num_batch * epoch + i)
                writer.add_scalar('lr', schedulerG.get_lr()[0], num_batch * epoch + i)
                writer.close()
                print('[%d/%d][%d/%d] [missing_cd/all_cd/rep/d_loss/GD_loss/GwithD_l2/GwithD_sum_loss]: %.4f / %.4f / %.4f / %.4f / %.4f / %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         CD_LOSS, CD_LOSS_ALL, RepLoss, errD.data,errG_D.data,errG_l2,errG))

                now_time = time.time()
                print("one of dataloader:" + str(now_time - begin))
            schedulerD.step()
            schedulerG.step()
            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': myNet.state_dict()},
                           opt.checkpointDir+'/Trained_Model/point_netG' + str(epoch) + '.pth')
                torch.save({'epoch': epoch + 1,
                            'state_dict': myNetD.state_dict()},
                           opt.checkpointDir+'/Trained_Model/point_netD' + str(epoch) + '.pth')

    else:

        # only g-net
        for epoch in range(resume_epoch, opt.niter):
            for i, data in enumerate(dataloader, 0):
                real_point, image, view = data

                batch_size = real_point.size()[0]
                num_sum = missingNum + incompleteNum  # 256,1024

                # real_point采样为1280个点
                real_point = Variable(real_point, requires_grad=False)  # 只有variable才能在gpu上计算
                real_point_key1_idx = farthest_point_sample(real_point, num_sum, RAN=False)
                real_point = index_points(real_point, real_point_key1_idx)
                complete_gt = real_point

                # 占位符
                missingPC = torch.FloatTensor(batch_size, missingNum, 3)
                incompletePC = torch.FloatTensor(batch_size, num_sum, 3)  # 这里还没有构成精确的incomplete大小
                incompletePC = incompletePC.data.copy_(real_point)  # post-fixed with _ 表示是原地操作

                center = view
                for m in range(batch_size):
                    distance_list = []
                    p_center = center

                    for n in range(num_sum):
                        distance_list.append(distance_squre(real_point[m, n], center[m]))
                    distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
                    # sorted返回的是重新排序的列表以及下标,nums[0]=data's index in distance_list,nums[1]=data
                    for sp in range(missingNum):
                        # distance_order[sp][0]为这个值在list里面原来的下标，也就是在realpoint中的下标
                        missingPC.data[m, sp] = real_point[m, distance_order[sp][0]]  # 离中心点进度在missing中

                    for sp in range(missingNum, num_sum):
                        incompletePC.data[m, sp] = real_point[m, distance_order[sp][0]]  # 离中心点进度在missing中

                incompletePC = incompletePC.data[:, missingNum:]

                missingPC = missingPC.to(device)  # input complete，不过有2048个点
                incompletePC = incompletePC.to(device)  # input incomplete,不过点的个数太多了需要点采样一下

                gt = missingPC.to(device)
                image = image.to(device)
                complete_gt = complete_gt.to(device)

                incomplete = Variable(incompletePC, requires_grad=True)
                image = Variable(image.float(), requires_grad=True)
                image = torch.squeeze(image, 1)

                myNet = myNet.train()
                myNet.zero_grad()
                fake = myNet(incomplete, image)
                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################

                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(gt, 1))
                RepLoss = 0
                errG = CD_LOSS
                if opt.Rep_choose == 1:
                    RepLoss = criterion_RepLoss(torch.squeeze(fake, 1))
                    errG += RepLoss
                errG.backward()

                optimizerG.step()

                writer = SummaryWriter(log_dir=opt.tensorboardDir)
                writer.add_scalar('cd_missing', CD_LOSS, num_batch * epoch + i)
                writer.add_scalar('repulsion', RepLoss, num_batch * epoch + i)
                writer.add_scalar('GnoD_SumLoss', errG, num_batch * epoch + i)
                complete_pc = torch.cat([fake, incomplete], dim=1)
                CD_LOSS_ALL = criterion_PointLoss(torch.squeeze(complete_pc, 1), torch.squeeze(complete_gt, 1))
                writer.add_scalar('cd_all', CD_LOSS_ALL, num_batch * epoch + i)
                writer.add_scalar('lr', schedulerG.get_lr()[0], num_batch * epoch + i)
                writer.close()
                print('[%d/%d][%d/%d] [missing_cd/all_cd/rep/all_loss]: %.4f / %.4f / %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         CD_LOSS, CD_LOSS_ALL,RepLoss, errG))

            schedulerG.step()

            if epoch % 10 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': myNet.state_dict()},
                           opt.checkpointDir+'/point_netG' + str(epoch) + '.pth')