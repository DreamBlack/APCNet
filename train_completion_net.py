
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
from myutils import PointLoss
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Lamp', help="which class choose to train")
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
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

MLP_dims = (3, 64, 64, 64, 128, 1024)
FC_dims = (1024, 1024, 512)
grid_dims = (16, 16)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 128, 1024, 1024, 256)
knn = 48
sigma = 0.008
mynet = myNet(3,128,128,MLP_dims, FC_dims,grid_dims,Folding1_dims,Folding2_dims,Weight1_dims,Weight3_dims)


print("Let's use", torch.cuda.device_count(), "GPUs!")
mynet = torch.nn.DataParallel(mynet)
mynet.to(device)
mynet.apply(weights_init_normal)

if opt.netG != '' :
    mynet.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


# load data
dset = MyDataset(classification=True,
                 class_choice=opt.class_choice, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = opt.workers)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
optimizerG = torch.optim.Adam(mynet.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=weight_decay)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

num_batch = len(dset) / opt.batchSize

if __name__=='__main__':
    # only g-net
    writer = SummaryWriter(log_dir='tensorboard/')
    for epoch in range(resume_epoch, opt.niter):
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        for i, data in enumerate(dataloader, 0):
            incomplete, gt, image,complete_pc = data

            batch_size = incomplete.size()[0]

            incomplete = incomplete.to(device)

            gt = gt.to(device)
            image = image.to(device)
            incomplete = Variable(incomplete, requires_grad=True)
            image = Variable(image.float(), requires_grad=True)
            image = torch.squeeze(image, 1)

            mynet = mynet.train()
            mynet.zero_grad()
            fake = mynet(incomplete, image)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################

            CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(gt, 1))

            CD_LOSS.backward()
            optimizerG.step()
            writer.add_scalar('loss/cd', CD_LOSS, num_batch * epoch + i)
            print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader),
                     CD_LOSS, CD_LOSS))
            f = open('loss_PFNet.txt', 'a')
            f.write('\n' + '[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                    % (epoch, opt.niter, i, len(dataloader),
                       CD_LOSS, CD_LOSS))
            f.close()
        schedulerG.step()

        if epoch % 10 == 0:
            torch.save({'epoch': epoch + 1,
                        'state_dict': mynet.state_dict()},
                       'Checkpoint/point_netG' + str(epoch) + '.pth')

        writer.close()