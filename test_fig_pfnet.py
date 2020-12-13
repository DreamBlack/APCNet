import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from MyDataset_former import MyDataset
from model_PFNet import _netG
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--manualSeed', type=int,default=1, help='manual seed')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--class_choice', default='Car', help="which class choose to train")
parser.add_argument('--attention_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--four_data', type = int, default = 0, help='enables cuda')
parser.add_argument('--folding_decoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--pointnetplus_encoder', type = int, default = 0, help='enables cuda')
parser.add_argument('--netG', help="path to netG (to load as model)")
parser.add_argument('--index', type=int,default=400, help='which obj to show')
opt = parser.parse_args()
print(opt)

# path_Nets=['/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/Checkpoint (复件)/point_netG150.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/withDCkPath/Trained_Model/point_netG50.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/noDCkPath/Checkpoint/point_netG120.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/withDCkPath/WITHDAR/Trained_Model/point_netG130.pth']
#path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/fc_decoder_withG_exp/checkpoint/Trained_Model/point_netG100.pth'
path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/folding_rep_attention/02958343/checkpoint/point_netG150.pth'
path_Nets=[opt.netG]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True # promote
MLP_dimsG = (3, 64, 64, 64, 128, 1024)
FC_dimsG = (1024, 1024, 512)

if opt.class_choice=='Lamp' or  opt.class_choice=='Car' :
    if opt.pointnetplus_encoder==0 and opt.folding_decoder==0:# 第四章，在所有car lamp数据集上的folding实验point都用小的
        MLP_dimsG = (3, 64, 64, 64, 128, 512)
        FC_dimsG = (512, 512, 512)

grid_dims = (16, 16)  # defaul 45
Folding1_dims = (514, 512, 512, 3)  # for foldingnet
Folding2_dims = (515, 512, 512, 3)  # for foldingnet
Weight1_dims = (16 * 16 + 512, 512, 512, 128)  # for weight matrix estimation 45x45+512 = 2537
Weight3_dims = (512 + 128, 1024, 1024, 256)
knn = 48
sigma = 0.008

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
incompleteColor='lightgray'
recColor='royalblue'
pointSize=50
def draw_point_cloud(incomplete,rec_missing, elev=0,azim=0,output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure(figsize=(24, 12), facecolor='w')  #
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)


    ax0 = fig.add_subplot(111, projection='3d')
    ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax0.scatter(rec_missing[:, 0], rec_missing[:, 1], rec_missing[:, 2], color=recColor, s=pointSize)
    ax0.set_axis_off()
    plt.title("GT")


    plt.show()

#画出3d点云,image+gt+result
def pyplot_draw_point_cloud(image,incomplete,rec_missing, elev=0,azim=0,output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure(figsize=(24, 12), facecolor='w')  #
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(131)
    ax.imshow(image)
    plt.title("input image")

    ax0 = fig.add_subplot(132, projection='3d')
    ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax0.scatter(missings[0][:, 0], missings[0][:, 1], missings[0][:, 2], color=recColor, s=pointSize)
    ax0.set_axis_off()
    plt.title("GT")

    ax1 = fig.add_subplot(133, projection='3d')
    ax1.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax1.scatter(missings[1][:, 0], missings[1][:, 1], missings[1][:, 2], color=recColor, s=pointSize)
    ax1.set_axis_off()
    plt.title("Vanilla")

    plt.show()
    # for angle in range(0, 360):
    #     ax0.view_init(30, angle)
    #     ax1.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
    # savefig(output_filename)
def pyplot_for_one_object(image,incomplete,missings, elev=0,azim=0,output_filename=None):
    '''
    对于每个物体，再一行中画出6个关于该物体的结果，从左到右分别是image，gt(in+mi),vanilla,gan,rep,full
    :param incomplete:gt
    :param missings: 5个missing，分别为gt,vanilla,gan,rep,full
    :param elev:
    :param azim:
    :param output_filename:
    :return:
    '''
    """ incompletes is a 5*Nx3 numpy array """
    fig=plt.figure(figsize=(24,12),facecolor='w')#
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax= fig.add_subplot(161)
    ax.imshow(image)
    plt.title("input image")

    ax0 = fig.add_subplot(162,projection='3d')
    ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2],color=incompleteColor, s=pointSize)
    ax0.scatter(missings[0][:, 0], missings[0][:, 1], missings[0][:, 2], color=recColor, s=pointSize)
    ax0.set_axis_off()
    plt.title("GT")

    ax1 = fig.add_subplot(163,projection='3d')
    ax1.scatter(incomplete[ :, 0], incomplete[ :, 1], incomplete[ :, 2], color=incompleteColor, s=pointSize)
    ax1.scatter(missings[1][ :, 0], missings[1][ :, 1], missings[1][ :, 2], color=recColor, s=pointSize)
    ax1.set_axis_off()
    plt.title("Vanilla")

    ax2 = fig.add_subplot(164, projection='3d')
    ax2.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax2.scatter(missings[2][:, 0], missings[2][:, 1], missings[2][:, 2], color=recColor, s=pointSize)
    ax2.set_axis_off()
    plt.title("With D")

    ax3 = fig.add_subplot(165, projection='3d')
    ax3.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax3.scatter(missings[3][:, 0], missings[3][:, 1], missings[3][:, 2], color=recColor, s=pointSize)
    ax3.set_axis_off()
    plt.title("With Rep")

    ax4 = fig.add_subplot(166, projection='3d')
    ax4.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax4.scatter(missings[4][:, 0], missings[4][:, 1], missings[4][:, 2], color=recColor, s=pointSize)
    ax4.set_axis_off()
    plt.title("With D&Rep")
    plt.show()

    # for angle in range(0, 360):
    #     ax0.view_init(30, angle)
    #     ax1.view_init(30, angle)
    #     ax2.view_init(30, angle)
    #     ax3.view_init(30, angle)
    #     ax4.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
    # savefig(output_filename)
test_dset = MyDataset(classification=True,three=opt.folding_decoder,
                 class_choice=opt.class_choice, split='test',four_data=opt.four_data)
assert test_dset
# good:10749,2269,5320
# bad:9958,9960,4305
if __name__ == '__main__':

    no=True
    if no:
        # if opt.class_choice=='Car':
        #     MLP_dimsG = (3, 64, 64, 64, 128, 512)
        #     FC_dimsG = (512, 512, 512)
        missings = []
        incomplete, gt, image, filename = test_dset.__getitem__(opt.index)
        missings.append(gt)
        my_image = np.transpose(image, (1, 2, 0))

        incomplete = torch.unsqueeze(incomplete, 0)
        image = torch.unsqueeze(image, 0)
        gt = torch.unsqueeze(gt, 0)
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

        for i in range(1):
            net=_netG(3,1,[1024,512,256],256)
            net = torch.nn.DataParallel(net)
            net.to(device)
            net.load_state_dict(torch.load(path_Nets[i], map_location=lambda storage, location: storage)['state_dict'],strict=False)
            net.eval()
            fake_center1, fake_center2, fake = net(input_cropped)
            missings.append(fake.cuda().data.cpu().squeeze(0).numpy())

        print(len(missings))
        print(incomplete.cuda().data.squeeze(0).shape)

        # 显示单个结果
        # pyplot_draw_point_cloud(my_image,incomplete.cuda().data.squeeze(0).cpu().numpy(),missings)
        pyplot_draw_point_cloud(my_image, incomplete.cuda().data.squeeze(0).cpu().numpy(), missings)

        # 构建incomplete和missings数组
        # pyplot_for_one_object(my_image,incomplete.cuda().data.squeeze(0).cpu().numpy(),missings)
    else:
        # dir_path = '/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerateForFour/03636649'
        dir_path='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerateForFour/04379243'
        filename1 = '/gt4ceba450382724f7861fea89ab9e083a.pts'
        filename2 = '/incomplete4ceba450382724f7861fea89ab9e083a.pts'
        # complete = np.loadtxt("/home/dream/study/codes/PCCompletion/新建文件夹chair/test/4incomplete1b81441b7e597235d61420a53a0cb96d.pts").astype(np.float32)
        # incomplete = np.loadtxt("/home/dream/study/codes/PCCompletion/新建文件夹chair/test/4gt1b81441b7e597235d61420a53a0cb96d.pts").astype(np.float32)
        # incompletechair=np.loadtxt("/home/dream/study/0incomplete1a00aa6b75362cc5b324368d54a7416f.pts")
        complete = '/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/04379243/points/1a00aa6b75362cc5b324368d54a7416f.pts'
        gt = np.loadtxt(dir_path + filename1)
        incomplete = np.loadtxt(dir_path + filename2)
        draw_point_cloud(gt, incomplete)
