
from pix3d.pix3d_dataset import Pix3DMultiDataset
import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
from PIL import Image
from completion_net import myNet
from test_fig import get_img_by_modelname
from model_PFNet import _netG
from pf_net_three import _netG as vanilia_netG
from former_comnet import myNet as formernet
import time
parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--class_choice', default='Table', help="which class choose to train")
parser.add_argument('--attention_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--folding_decoder', type = int, default = 0, help='enables cuda')
parser.add_argument('--pointnetplus_encoder', type = int, default = 1, help='enables cuda')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--four_data', type = int, default = 1, help='enables cuda')
parser.add_argument('--index', type = int, default = 300, help='enables cuda')
parser.add_argument('--usedataset', type = bool, default = True, help='enables cuda')
parser.add_argument('--test', type = bool, default = True, help='enables cuda')
parser.add_argument('--needSave', type = bool, default = False, help='enables cuda')

opt = parser.parse_args()
print(opt)

#good lamp
# 244 -134 -138
# 234 -99 99
# 184 -127 148
# 174 -87 84
# 144 -155 132

elev = 37  # a=elev
azim = 27 # b=azim
# good airplane
catname_lower={'Car':"car",'Lamp':"lamp",'Chair':"chair","Table":'table',"Airplane":'airplane'}
my_net_path_home='/home/dream/study/codes/PCCompletion/best_four_exp/pix3d'
my_net_path=os.path.join(my_net_path_home,catname_lower[opt.class_choice]+"0.3","checkpoint","point_netG150.pth")
my_net_path2=os.path.join(my_net_path_home,catname_lower[opt.class_choice],"checkpoint","with_rep_point_netG150.pth")
if not os.path.exists(my_net_path):
    my_net_path = os.path.join(my_net_path_home,catname_lower[opt.class_choice], "checkpoint", "point_netG130.pth")
netpaths=[my_net_path,my_net_path2]

pf_net_path_vanilia=os.path.join("/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/","vanilia",catname_lower[opt.class_choice],"checkpoint","point_netG130.pth")
pf_net_path_image=os.path.join("/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/","with_image",catname_lower[opt.class_choice],"checkpoint","point_netG130.pth")
pfnet_path=[pf_net_path_vanilia,pf_net_path_image]

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


image_home='/home/dream/study/codes/densepcr/datasets/rendering'
pc_home='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate'
catname_id={'Car':"02958343",'Lamp':"03636649",'Chair':"03001627","Table":'04379243',"Airplane":'02691156'}

def get_result_for_image_pfNet(incomplete,image,netpath):
    point_netG = _netG(3, 1, [1024, 512, 256], 256)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    point_netG.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
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
    fake_center1, fake_center2, fake = point_netG(input_cropped,image)
    return fake.cuda().data.cpu().squeeze(0).numpy()
def compute_mytime(netpath):
    net = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims,
                Weight3_dims, folding=opt.folding_decoder, attention=opt.attention_encoder,
                pointnetplus=opt.pointnetplus_encoder)
    net = torch.nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
                        strict=False)
    total_time=0
    for i in range(len(test_dset)):

        incomplete, gt, image, filename, view_id, center_id = test_dset.__getitem__(i)

        incomplete = torch.unsqueeze(incomplete, 0)
        image = torch.unsqueeze(image, 0)

        incomplete = incomplete.to(device)
        image = image.to(device)

        incomplete = Variable(incomplete, requires_grad=False)
        image = Variable(image.float(), requires_grad=False)
        net.eval()
        start=time.time()
        fake = net(incomplete, image)
        end=time.time()
        total_time=total_time+end-start
    print(("mynet ---- average time %.4f for cat: %s"%(float(total_time/len(test_dset)),opt.class_choice)))


def compute_pftime(netpath):
    point_netG = _netG(3, 1, [1024, 512, 256], 256)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    point_netG.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
                               strict=False)
    point_netG.eval()


    total_time = 0
    for i in range(len(test_dset)):
        incomplete, gt, image, filename, view_id, center_id = test_dset.__getitem__(i)

        incomplete = torch.unsqueeze(incomplete, 0)
        image = torch.unsqueeze(image, 0)

        incomplete = incomplete.to(device)
        image = image.to(device)

        incomplete = Variable(incomplete, requires_grad=False)
        image = Variable(image.float(), requires_grad=False)
        start = time.time()
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
        fake_center1, fake_center2, fake = point_netG(input_cropped, image)

        end = time.time()
        total_time = total_time + end - start
    print(("pfnet---average time %.4f for cat: %s" % (float(total_time / len(test_dset)), opt.class_choice)))
def get_result_for_myNet(incomplete,image,netpath):
    print(netpath)
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
def get_result_for_myNet_before(incomplete,image,netpath):
    net = formernet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims,
                Weight3_dims, folding=opt.folding_decoder, attention=opt.attention_encoder,
                pointnetplus=opt.pointnetplus_encoder)
    net = torch.nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
                        strict=False)
    net.eval()
    fake = net(incomplete, image)
    return fake.cuda().data.cpu().squeeze(0).numpy()
def get_result_for_pfNet(incomplete,netpath=""):
    point_netG = vanilia_netG(3, 1, [1024, 512, 256], 256)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    point_netG.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'],
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

def compare_my_and_pf(cat):
    incomplete, gt, image,pcl_2025,img_path= test_dset.__getitem__(opt.index)
    print("filename:" + str(img_path))

    imgpng = get_img_by_modelname(img_path)

    incomplete = torch.unsqueeze(incomplete, 0)
    image = torch.unsqueeze(image, 0)

    incomplete = incomplete.to(device)
    image = image.to(device)

    incomplete = Variable(incomplete, requires_grad=False)
    image = Variable(image.float(), requires_grad=False)

    pf_fakes = []

    pf_fakes.append(get_result_for_pfNet(incomplete, pfnet_path[0]))

    pf_fakes.append(get_result_for_image_pfNet(incomplete,image,pfnet_path[1]))

    my_fake=get_result_for_myNet(incomplete, image,netpaths[0]) #_before
    #my_fake = get_result_for_myNet(incomplete, image, netpaths[1])  # _before


    incomplete=incomplete.cuda().data.squeeze(0).cpu().numpy()
    fig = plt.figure(figsize=(26, 6), facecolor='w')  #
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    if opt.test:
        ax = fig.add_subplot(151)
        fig.set_facecolor('none')
        ax.imshow(imgpng)
        #plt.title("input image")
        plt.axis("off")


        #plt.title("GT")

        ax1 = fig.add_subplot(152, projection='3d')
        ax1.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax1.scatter(pf_fakes[0][:, 0], pf_fakes[0][:, 1], pf_fakes[0][:, 2], color=recColor, s=pointSize)
        ax1.set_axis_off()
        #plt.title("PF-Net-vanilia")

        ax2 = fig.add_subplot(153, projection='3d')
        ax2.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax2.scatter(pf_fakes[1][:, 0], pf_fakes[1][:, 1], pf_fakes[1][:, 2], color=recColor, s=pointSize)
        #  ax2.scatter(my_fake2[:, 0], my_fake2[:, 1], my_fake2[:, 2], color=recColor, s=pointSize)
        ax2.set_axis_off()
        #plt.title("PF-Net-image")

        ax3 = fig.add_subplot(154, projection='3d')
        ax3.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax3.scatter(my_fake[:, 0], my_fake[:, 1], my_fake[:, 2], color=recColor, s=pointSize)
        ax3.set_axis_off()
        #plt.title("My")


        ax0 = fig.add_subplot(155, projection='3d')
        ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax0.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color=recColor, s=pointSize)
        ax0.set_axis_off()




        ax0.view_init(elev, azim)
        ax1.view_init(elev, azim)
        ax2.view_init(elev, azim)
        ax3.view_init(elev, azim)
        #fig.set_tight_layout(True)
        if opt.needSave:
            output_filename = os.path.join("/home/dream/study/four_result_images",
                                           cat +"a_"+str(azim)+"e_"+str(elev)+str(opt.index)  + ".png")
            plt.savefig(output_filename)
            plt.show()
            im = Image.open(output_filename)
            x = 100
            y = 50
            h = 500
            w = 2600
            region = im.crop((x, y, x + w, y + h))
            region.save(output_filename)
        else:
            plt.show()
    else:
        ax = fig.add_subplot(151)
        fig.set_facecolor('none')
        ax.imshow(imgpng)
        plt.axis("off")

        ax0 = fig.add_subplot(152, projection='3d')
        ax0.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax0.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color=recColor, s=pointSize)
        ax0.set_axis_off()

        ax1 = fig.add_subplot(153, projection='3d')
        ax1.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax1.scatter(pf_fakes[0][:, 0], pf_fakes[0][:, 1], pf_fakes[0][:, 2], color=recColor, s=pointSize)
        ax1.set_axis_off()

        ax2 = fig.add_subplot(154, projection='3d')
        ax2.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax2.scatter(pf_fakes[1][:, 0], pf_fakes[1][:, 1], pf_fakes[1][:, 2], color=recColor, s=pointSize)
        ax2.set_axis_off()

        ax3 = fig.add_subplot(155, projection='3d')
        ax3.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2], color=incompleteColor, s=pointSize)
        ax3.scatter(my_fake[:, 0], my_fake[:, 1], my_fake[:, 2], color=recColor, s=pointSize)
        ax3.set_axis_off()

        ax0.view_init(elev, azim)
        ax1.view_init(elev, azim)
        ax2.view_init(elev, azim)

        fig.set_tight_layout(True)
        if opt.needSave:
            output_filename = os.path.join("/home/dream/study/four_result_images",
                                           cat +  ".png")
            plt.savefig(output_filename)
            plt.show()
            im = Image.open(output_filename)
            x = 100
            y = 50
            h = 500
            w = 2300
            region = im.crop((x, y, x + w, y + h))
            region.save(output_filename)
        else:
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

test_dset = Pix3DMultiDataset(
                 class_choice=opt.class_choice)
assert test_dset
# good:10749,2269,5320
# bad:9958,9960,4305
if __name__ == '__main__':
    # compute_mytime(netpaths[0])
    # #compute_pftime(pfnet_path[1])
    # exit(0)
    no=True
    if no:
        missings = []
        #incomplete, gt, image, filename = test_dset.__getitem__(opt.index)
        gt=torch.rand((4,6))
        missings.append(gt)
        compare_my_and_pf(opt.class_choice) # (cat,obj_id,imageview,pccenter):

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
