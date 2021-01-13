import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from MyDataset_former import MyDataset
from pcn_foldingnet import pcnFoldingNet
from myutils import PointLoss,EMDLoss,RepulsionLoss
from tensorboardX import SummaryWriter
import os

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Lamp', help="which class choose to train")
parser.add_argument('--niter', type=int, default=151, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--four_data', type = int, default = 0, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--checkpointDir', default='', help="path to netG (to continue training)")
parser.add_argument('--tensorboardDir', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--step_size',type=int,default=40,help='LR step size')
parser.add_argument('--gamma',type=float,default=0.2,help='LR gamma')
parser.add_argument('--wtl2',type=float,default=0.95,help='weight for loss 0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

root_path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/best_three/02958343'

print('命令行参数写入文件')
f_all=open(os.path.join(opt.checkpointDir,'train_param.txt'), 'w')
f_all.write("\n"+"workers"+"  "+str(opt.workers))
f_all.write("\n"+"batchSize"+"  "+str(opt.batchSize))
f_all.write("\n"+"class_choice"+"  "+str(opt.class_choice))
f_all.write("\n"+"step_size"+"  "+str(opt.step_size))
f_all.write("\n"+"learning_rate"+"  "+str(opt.learning_rate))
f_all.write("\n"+"four_data"+"  "+str(opt.four_data))


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

f_all.close()

MLP_dims = (3, 64, 64, 64, 128, 1024)
FC_dims = (1024, 512, 512)
grid_dims = (45, 45)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (45 * 45 + 512, 512, 512, 512)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 512, 1024, 1024, 2025)
knn = 48
sigma = 0.008
myNet = pcnFoldingNet(MLP_dims, FC_dims,grid_dims,Folding1_dims,Folding2_dims,Weight1_dims,Weight3_dims)

print("Let's use", torch.cuda.device_count(), "GPUs!")
myNet = torch.nn.DataParallel(myNet)
myNet.to(device)
myNet.apply(weights_init_normal)

if opt.netG != '' :
    myNet.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'],strict=False)
    resume_epoch = torch.load(opt.netG)['epoch']


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# load data
dset = MyDataset(classification=True,class_choice=opt.class_choice, split='train',four_data=opt.four_data)
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers,drop_last=True)

test_dset = MyDataset(classification=True,class_choice=opt.class_choice, split='test',four_data=opt.four_data)
assert test_dset
test_dataloader = torch.utils.data.DataLoader(test_dset,batch_size=opt.batchSize,shuffle=True,num_workers = opt.workers,drop_last=True)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
optimizerG = torch.optim.Adam(myNet.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=opt.step_size, gamma=opt.gamma)


num_batch = len(dset) / opt.batchSize
if __name__=='__main__':
    ###########################
    #  G-NET and T-NET
    ##########################
    # only g-net
    for epoch in range(resume_epoch, opt.niter):
        for i, data in enumerate(dataloader, 0):
            incomplete, gt, image, filename,cmpgt = data
            batch_size = incomplete.size()[0]
            incomplete = incomplete.to(device)
            gt = cmpgt.to(device)
            incomplete = Variable(incomplete, requires_grad=True).cuda()

            myNet = myNet.train()
            myNet.zero_grad()
            fake = myNet(incomplete)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################

            CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1).cuda(), torch.squeeze(gt, 1).cuda())
            errG = CD_LOSS
            errG.backward()

            optimizerG.step()

            writer = SummaryWriter(log_dir=opt.tensorboardDir)
            writer.add_scalar('all_cd', CD_LOSS, num_batch * epoch + i)
            writer.add_scalar('lr', schedulerG.get_lr()[0], num_batch * epoch + i)
            writer.close()
            print('[%d/%d][%d/%d] [all_cd]: %.4f '
                  % (epoch, opt.niter, i, len(dataloader),
                     CD_LOSS))

        schedulerG.step()

        if epoch % 10 == 0:
            torch.save({'epoch': epoch + 1,
                        'state_dict': myNet.state_dict()},
                       opt.checkpointDir + '/point_netG' + str(epoch) + '.pth')