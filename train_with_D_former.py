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
from MyDataset_former import MyDataset
from completion_net import myNet
from myutils import PointLoss,EMDLoss,RepulsionLoss
from tensorboardX import SummaryWriter
from my_discriminator import myDiscriminator
import time
import os

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Lamp', help="which class choose to train")
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--attention_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--folding_decoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--pointnetplus_encoder', type = int, default = 0, help='enables cuda')
parser.add_argument('--four_data', type = int, default = 0, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=0, help='0 not use D-net,1 use D-net')
parser.add_argument('--Rep_choose',type=int, default=0, help='0 not use Rep Loss,1 use Rep Loss')
parser.add_argument('--loss_emd',type=int, default=0, help='0 use cd Loss,1 use emd Loss')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--checkpointDir', default='', help="path to netG (to continue training)")
parser.add_argument('--tensorboardDir', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0, help='use when charpter 4 in fc decoder')
parser.add_argument('--dropout_feature',type=float,default=0, help='use when charpter 4 folding decoder')
parser.add_argument('--alpha',type=float,default=0.5, help='rep loss weight')
parser.add_argument('--radius',type=float,default=0.07, help='radius of rep loss')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--step_size',type=int,default=40,help='LR step size')
parser.add_argument('--gamma',type=float,default=0.2,help='LR gamma')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='weight for loss 0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)
if opt.pointnetplus_encoder==1:
    torch.backends.cudnn.enabled = False

root_path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/best_three/02958343'

print('命令行参数写入文件')
f_all=open(os.path.join(opt.checkpointDir,'train_param.txt'), 'w')
f_all.write("\n"+"workers"+"  "+str(opt.workers))
f_all.write("\n"+"batchSize"+"  "+str(opt.batchSize))
f_all.write("\n"+"attention_encoder"+"  "+str(opt.attention_encoder))
f_all.write("\n"+"folding_decoder"+"  "+str(opt.folding_decoder))
f_all.write("\n"+"pointnetplus_encoder"+"  "+str(opt.pointnetplus_encoder))
f_all.write("\n"+"class_choice"+"  "+str(opt.class_choice))
f_all.write("\n"+"D_choose"+"  "+str(opt.D_choose))
f_all.write("\n"+"Rep_choose"+"  "+str(opt.Rep_choose))
f_all.write("\n"+"alpha"+"  "+str(opt.alpha))
f_all.write("\n"+"step_size"+"  "+str(opt.step_size))
f_all.write("\n"+"gamma"+"  "+str(opt.gamma))
f_all.write("\n"+"drop"+"  "+str(opt.drop))
f_all.write("\n"+"dropout_feature"+"  "+str(opt.dropout_feature))
f_all.write("\n"+"learning_rate"+"  "+str(opt.learning_rate))
f_all.write("\n"+"four_data"+"  "+str(opt.four_data))
f_all.write("\n"+"loss_emd"+"  "+str(opt.loss_emd))


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
if  opt.class_choice=='Car' : # 不论是第三还是第四章，在car上的实验point都用小的
    MLP_dimsG = (3, 64, 64, 64, 128, 512)
    FC_dimsG = (512, 512, 512)
if opt.folding_decoder==0 and opt.class_choice=='Lamp':# 第四章，在lamp上的fc实验point都用小的
    MLP_dimsG = (3, 64, 64, 64, 128, 512)
    FC_dimsG = (512, 512, 512)
if opt.folding_decoder==1 and opt.four_data==1:# 第四章，在所有数据集上的folding实验point都用小的
    MLP_dimsG = (3, 64, 64, 64, 128, 512)
    FC_dimsG = (512, 512, 512)

f_all.write("\n"+"MLP_dimsG"+"  "+str(MLP_dimsG))
f_all.write("\n"+"FC_dimsG"+"  "+str(FC_dimsG))

f_all.close()
MLP_dimsD = (3, 64, 64, 64, 128, 1024)
FC_dimsD = (1024, 1024, 512)
grid_dims = (16, 16)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 128, 1024, 1024, 256)
knn = 48
sigma = 0.008
myNet = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims, Weight3_dims,dropout=opt.drop,folding=opt.folding_decoder,dropout_feature=opt.dropout_feature,attention=opt.attention_encoder,pointnetplus=opt.pointnetplus_encoder)
myNetD=myDiscriminator(256,MLP_dimsD,FC_dimsD)


print("Let's use", torch.cuda.device_count(), "GPUs!")
myNet = torch.nn.DataParallel(myNet)
myNet.to(device)
myNet.apply(weights_init_normal)
myNetD = torch.nn.DataParallel(myNetD)
myNetD.to(device)
myNetD.apply(weights_init_normal)

if opt.netG != '' :
    myNet.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'],strict=False)
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    myNetD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'],strict=False)
    resume_epoch = torch.load(opt.netD)['epoch']

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# load data
dset = MyDataset(classification=True,three=opt.folding_decoder,class_choice=opt.class_choice, split='train',four_data=opt.four_data)
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers)

test_dset = MyDataset(classification=True,class_choice=opt.class_choice, split='test',four_data=opt.four_data)
assert test_dset
test_dataloader = torch.utils.data.DataLoader(test_dset,batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)
if opt.loss_emd==1:
    criterion_PointLoss = EMDLoss().to(device)
    print("Emd loss is used.")
criterion_RepLoss=RepulsionLoss(alpha=opt.alpha,radius=opt.radius).to(device)

# setup optimizer
optimizerG = torch.optim.Adam(myNet.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=opt.step_size, gamma=opt.gamma)
optimizerD = torch.optim.Adam(myNetD.parameters(), lr=opt.learning_rate*0.5,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=opt.step_size, gamma=opt.gamma)

real_label = 1
fake_label = 0

num_batch = len(dset) / opt.batchSize
label = torch.FloatTensor(opt.batchSize).cuda()
if __name__=='__main__':
    ###########################
    #  G-NET and T-NET
    ##########################
    if opt.D_choose == 1:

        for epoch in range(resume_epoch, opt.niter):
            for i, data in enumerate(dataloader, 0):

                incomplete, gt, image,filename = data

                batch_size = incomplete.size()[0]

                incomplete = incomplete.to(device)

                gt = gt.to(device)
                image = image.to(device)

                incomplete = Variable(incomplete, requires_grad=True).cuda()
                image = Variable(image.float(), requires_grad=True).cuda()
                image = torch.squeeze(image, 1)
                label.resize_([batch_size, 1]).fill_(real_label)
                label = label.to(device)

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
                fake = torch.squeeze(fake, 1).cuda()
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
                output = myNetD(torch.squeeze(fake, 1).cuda())
                errG_D = criterion(output, label.cuda())
                errG_l2=0
                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1).cuda(), torch.squeeze(gt, 1).cuda())
                RepLoss=0
                errG_l2 = CD_LOSS
                if opt.Rep_choose==1:
                    RepLoss = criterion_RepLoss(torch.squeeze(fake, 1).cuda())
                    errG_l2+=RepLoss

                errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
                errG.backward()
                optimizerG.step()
                writer = SummaryWriter(log_dir=opt.tensorboardDir)
                writer.add_scalar('cd_missing', CD_LOSS, num_batch * epoch + i)
                writer.add_scalar('repulsion', RepLoss, num_batch * epoch + i)
                writer.add_scalar('D_Loss', errD.data, num_batch * epoch + i)
                writer.add_scalar('GD_Loss', errG_D.data, num_batch * epoch + i)
                writer.add_scalar('GwithD_sum_loss', errG, num_batch * epoch + i)
                writer.add_scalar('GwithD_l2', errG_l2, num_batch * epoch + i)
                complete_pc = torch.cat([fake, incomplete], dim=1)
                complete_gt = torch.cat([gt, incomplete], dim=1)
                CD_LOSS_ALL = criterion_PointLoss(torch.squeeze(complete_pc, 1).cuda(), torch.squeeze(complete_gt, 1).cuda())
                CD_LOSS_ALL=CD_LOSS_ALL.data.cpu()
                writer.add_scalar('cd_all', CD_LOSS_ALL, num_batch * epoch + i)
                writer.add_scalar('lr', schedulerG.get_lr()[0], num_batch * epoch + i)
                writer.close()
                print('[%d/%d][%d/%d] [missing_cd/all_cd/rep/d_loss/GD_loss/GwithD_l2/GwithD_sum_loss]: %.4f / %.4f / %.4f / %.4f / %.4f / %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         CD_LOSS, CD_LOSS_ALL, RepLoss, errD.data,errG_D.data,errG_l2,errG))

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
                incomplete, gt, image, filename = data
                batch_size = incomplete.size()[0]
                incomplete = incomplete.to(device)
                gt = gt.to(device)
                image = image.to(device)
                incomplete = Variable(incomplete, requires_grad=True).cuda()
                image = Variable(image.float(), requires_grad=True).cuda()
                image = torch.squeeze(image, 1)

                myNet = myNet.train()
                myNet.zero_grad()
                fake = myNet(incomplete, image)
                ############################
                # (3) Update G network: maximize log(D(G(z)))
                ###########################

                CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1).cuda(), torch.squeeze(gt, 1).cuda())
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
                complete_gt = torch.cat([gt, incomplete], dim=1)
                CD_LOSS_ALL = criterion_PointLoss(torch.squeeze(complete_pc, 1), torch.squeeze(complete_gt, 1))
                writer.add_scalar('cd_all', CD_LOSS_ALL, num_batch * epoch + i)
                writer.add_scalar('lr', schedulerG.get_lr()[0], num_batch * epoch + i)
                writer.close()
                print('[%d/%d][%d/%d] [missing_cd/all_cd/rep/all_loss]: %.4f / %.4f / %.4f / %.4f '
                      % (epoch, opt.niter, i, len(dataloader),
                         CD_LOSS, CD_LOSS_ALL,RepLoss, errG))

            schedulerG.step()

            if epoch % 5 == 0:
                torch.save({'epoch': epoch + 1,
                            'state_dict': myNet.state_dict()},
                           opt.checkpointDir+'/point_netG' + str(epoch) + '.pth')