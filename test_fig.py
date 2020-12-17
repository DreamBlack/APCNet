import sys

import cv2

sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from MyDataset_former import MyDataset
from completion_net import myNet
import numpy as np
import matplotlib.pyplot as plt
from pf_net_three import  _netG
import former_comnet
import utils
import os
parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--pccenter', type=int,default=1, help='number of data loading workers')
parser.add_argument('--imageview',  default="00", help='input batch size')
parser.add_argument('--obj_id',  default="1a56d596c77ad5936fa87a658faf1d26", help='input batch size')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Car', help="which class choose to train")
parser.add_argument('--attention_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--folding_decoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--pointnetplus_encoder', type = int, default = 0, help='enables cuda')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--pathA', default="/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/folding_rep_attention/02958343/checkpoint/point_netG130.pth")
parser.add_argument('--pathB', default="/home/dream/study/codes/PCCompletion/best_three_exp/pfnet/car/checkpoint/point_netG130.pth")
parser.add_argument('--result_path', help="path to netG (to load as model)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--four_data', type = int, default = 0, help='enables cuda')
parser.add_argument('--index', type = int, default = 1, help='enables cuda')
opt = parser.parse_args()
print(opt)


# path_Nets=['/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/Checkpoint (复件)/point_netG150.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/withDCkPath/Trained_Model/point_netG50.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/noDCkPath/Checkpoint/point_netG120.pth',
#            '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/withDCkPath/WITHDAR/Trained_Model/point_netG130.pth']
#path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/fc_decoder_withG_exp/checkpoint/Trained_Model/point_netG100.pth'
path='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/folding_rep_attention/02958343/checkpoint/point_netG150.pth'
path_Nets=[opt.pathA]
netpaths=['/home/dream/study/codes/PCCompletion/best_three_exp/lamp/checkpoint/point_netG130.pth',
'/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/noDCkPath/Checkpoint/point_netG120.pth']
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

#inin my net
# net = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims,
#             Weight3_dims, folding=opt.folding_decoder, attention=opt.attention_encoder,
#             pointnetplus=opt.pointnetplus_encoder)
# net = torch.nn.DataParallel(net)
# net.to(device)
# net.load_state_dict(torch.load(opt.pathA, map_location=lambda storage, location: storage)['state_dict'],
#                     strict=False)
# net.eval()


image_home='/home/dream/study/codes/densepcr/datasets/rendering'
pc_home='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate'
catname_id={'Car':"02958343",'Lamp':"03636649"}
def get_mode_by_name(cat,filename,imageview='00',pccenter=0):
    image_path=os.path.join(image_home,catname_id[cat],filename,"rendering",imageview+".png")
    input_crop_path=os.path.join(pc_home,catname_id[cat],str(pccenter)+"incomplete"+filename+".pts")
    input_gt_path=os.path.join(pc_home,catname_id[cat],str(pccenter)+"gt"+filename+".pts")
    print("image path %s"%image_path)
    print("input_crop_path %s"%input_crop_path)

    print("input_gt_path %s"%input_gt_path)
    if os.path.exists(image_path)==False or os.path.exists(image_path)==False or os.path.exists(image_path)==False:
        print("path is no correct. Exit!")
        exit(0)

    incomplete = np.loadtxt(input_crop_path).astype(np.float32)

    gt = np.loadtxt(input_gt_path).astype(np.float32)
    image = cv2.imread(image_path)[4:-5, 4:-5, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    incomplete = torch.from_numpy(incomplete)
    gt = torch.from_numpy(gt)
    image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return incomplete,gt,image

def get_result_for_pfNet(incomplete):
    point_netG = _netG(3, 1, [1024, 512, 256], 256)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    point_netG.load_state_dict(torch.load(opt.pathB, map_location=lambda storage, location: storage)['state_dict'],
                        strict=False)
    point_netG.eval()

    ############################
    # (1) data prepare
    ###########################
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
    fake_center1, fake_center2, fake = point_netG(input_cropped)
    return fake.cuda().data.cpu().squeeze(0).numpy()

def get_result_for_myNet(incomplete,image,netpath):
    net = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims,
                Weight3_dims, folding=opt.folding_decoder, attention=opt.attention_encoder,
                pointnetplus=opt.pointnetplus_encoder)
    net = torch.nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
                        strict=False)
    net.eval()

    fake = net(incomplete, image)
    return fake.cuda().data.cpu().squeeze(0).numpy()
def get_result_for_myNet2(incomplete,image,netpath):
    net2 = former_comnet.myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims,
                Weight3_dims, folding=opt.folding_decoder, attention=opt.attention_encoder,
                pointnetplus=opt.pointnetplus_encoder)
    net2 = torch.nn.DataParallel(net2)
    net2.to(device)
    net2.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
                        strict=False)
    net2.eval()

    fake = net2(incomplete, image)
    return fake.cuda().data.cpu().squeeze(0).numpy()
def compare_my_and_pf(cat,obj_id,imageview,pccenter,mynmet_path):

    incomplete,gt,image=get_mode_by_name(cat,obj_id,imageview,pccenter)

    my_image = np.transpose(image, (1, 2, 0))
    incomplete = torch.unsqueeze(incomplete, 0)
    image = torch.unsqueeze(image, 0)

    incomplete = incomplete.to(device)
    image = image.to(device)

    incomplete = Variable(incomplete, requires_grad=False)
    image = Variable(image.float(), requires_grad=False)


    pf_fake=get_result_for_pfNet(incomplete)
    my_fakes=[]
    my_fakes.append(get_result_for_myNet2(incomplete,image,opt.pathA))#mynmet_path[0]
    #my_fakes.append(get_result_for_myNet2(incomplete, image, mynmet_path[1]))

    incomplete=incomplete.cuda().data.squeeze(0).cpu().numpy()
    fig = plt.figure(figsize=(24, 12), facecolor='w')  #
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(141)
    fig.set_facecolor('none')
    ax.imshow(my_image)
    plt.title("input image")
    plt.axis("off")

    ax0 = fig.add_subplot(142, projection='3d')
    ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax0.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color=recColor, s=pointSize)
    ax0.set_axis_off()
    plt.title("GT")

    ax1 = fig.add_subplot(143, projection='3d')
    ax1.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax1.scatter(pf_fake[:, 0], pf_fake[:, 1], pf_fake[:, 2], color=recColor, s=pointSize)
    ax1.set_axis_off()
    plt.title("PF-Net")

    ax2 = fig.add_subplot(144, projection='3d')
    ax2.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax2.scatter(my_fakes[0][:, 0], my_fakes[0][:, 1], my_fakes[0][:, 2], color=recColor, s=pointSize)
    ax2.set_axis_off()
    plt.title("MyGood")

    # ax3 = fig.add_subplot(155, projection='3d')
    # ax3.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    # ax3.scatter(my_fakes[1][:, 0], my_fakes[1][:, 1], my_fakes[1][:, 2], color=recColor, s=pointSize)
    # ax3.set_axis_off()
    # plt.title("My2")

    plt.show()


incompleteColor='lightgray'
recColor='royalblue'
pointSize=55
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
def pyplot_draw_point_cloud(image,incomplete,rec_missing,modelnum=2, elev=0,azim=0,output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure(figsize=(24, 12), facecolor='w')  #
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    ax = fig.add_subplot(141)
    #image = np.transpose(image, (1,0,2))
    fig.set_facecolor('none')
    #image=np.fliplr(image.reshape(-1,3)).reshape(image.shape)
    ax.imshow(image)
    plt.title("input image")
    plt.axis("off")

    ax0 = fig.add_subplot(142, projection='3d')
    ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax0.scatter(missings[0][:, 0], missings[0][:, 1], missings[0][:, 2], color=recColor, s=pointSize)
    ax0.set_axis_off()
    plt.title("GT")

    ax1 = fig.add_subplot(143, projection='3d')
    ax1.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
    ax1.scatter(missings[1][:, 0], missings[1][:, 1], missings[1][:, 2], color=recColor, s=pointSize)
    ax1.set_axis_off()
    plt.title("Vanilla")

    ax1 = fig.add_subplot(144, projection='3d')
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
        missings = []
        #incomplete, gt, image, filename = test_dset.__getitem__(opt.index)
        gt=torch.rand((4,6))
        missings.append(gt)


        compare_my_and_pf(opt.class_choice,opt.obj_id,opt.imageview,opt.pccenter,netpaths) # (cat,obj_id,imageview,pccenter):

        # for i in range(1):
        #
        #     fake = net(incomplete, image)
        #     missings.append(fake.cuda().data.cpu().squeeze(0).numpy())
        #
        # print(len(missings))
        # print(incomplete.cuda().data.squeeze(0).shape)
        # 显示单个结果
        # pyplot_draw_point_cloud(my_image,incomplete.cuda().data.squeeze(0).cpu().numpy(),missings)


        #pyplot_draw_point_cloud(my_image, incomplete.cuda().data.squeeze(0).cpu().numpy(), missings)

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
