import sys
sys.path.append('/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network')
import argparse
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from MyDataset_former import MyDataset
from completion_net import myNet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from myutils import PointLoss_test
import heapq
import os

parser = argparse.ArgumentParser()  # create an argumentparser object

parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--class_choice', default='Car', help="which class choose to train")
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--folding_decoder', type = bool, default = True, help='enables cuda')
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

netpath='/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/best_three/02958343/checkpoint/Trained_Model/point_netG200.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True # promote
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
mynet = myNet(3, 128, 128, MLP_dimsG, FC_dimsG, grid_dims, Folding1_dims, Folding2_dims, Weight1_dims, Weight3_dims,folding=opt.folding_decoder)
mynet = torch.nn.DataParallel(mynet)
mynet.to(device)
mynet.load_state_dict(torch.load(netpath, map_location=lambda storage, location: storage)['state_dict'])
mynet.eval()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

test_dset = MyDataset( classification=True, three=opt.folding_decoder,class_choice=opt.class_choice, split='test')
criterion_PointLoss = PointLoss_test().to(device)
#画出3d点云
def pyplot_draw_point_cloud(image,incomplete,rec_missing, elev,azim,output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.view_init(elev, azim)
    ax.scatter(incomplete[:, 0], incomplete[:, 1], incomplete[:, 2],color='r')
    ax.scatter(rec_missing[:, 0], rec_missing[:, 1], rec_missing[:, 2], color='g')

    ax.set_axis_off()
    #plt.imshow(image)
    plt.show()
    # savefig(output_filename)

if __name__ == '__main__':
    listG,listB,listG=[],[],[]
    fGood = open('now_index_of_good_test.txt', 'a')
    fBad = open('now_index_of_bad_test.txt', 'a')

    bests_overall_pregt = []  # 这个元素是元组，(cd,index)
    bests_overall_gtpre = []  # 这个元素是元组，(cd,index)
    bests_missing_pregt = []
    bests_missing_gtpre = []
    bests_missing_cd = []
    bests_all_cd = []
    bests_num = 30
    have_heaped = False

    max_dist=0
    min_dist=10000
    max_index=0
    min_index=0
    for i in range(len(test_dset)):
        incomplete, gt, image,filename = test_dset.__getitem__(i)
        incomplete = torch.unsqueeze(incomplete, 0)
        gt = torch.unsqueeze(gt, 0)
        image = torch.unsqueeze(image, 0)

        complete_gt = torch.cat([gt, incomplete], dim=1)
        complete_gt = complete_gt.to(device)

        incomplete = incomplete.to(device)
        image = image.to(device)
        gt = gt.to(device)

        incomplete = Variable(incomplete, requires_grad=False)
        image = Variable(image.float(), requires_grad=False)

        fake = mynet(incomplete, image)

        dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(gt, 1))

        dist_all = dist_all.cpu().detach().numpy()*1000
        dist1 = dist1.cpu().detach().numpy()*1000
        dist2 = dist2.cpu().detach().numpy()*1000

        complete_pc = torch.cat([fake, incomplete], dim=1).to(device)
        dist_all_all, dist1_all, dist2_all = criterion_PointLoss(torch.squeeze(complete_pc, 1).to(device),
                                                                 torch.squeeze(complete_gt, 1).to(device))
        dist_all_all = dist_all_all.cpu().detach().numpy()*1000
        dist1_all = dist1_all.cpu().detach().numpy()*1000
        dist2_all = dist2_all.cpu().detach().numpy()*1000

        if len(bests_missing_pregt) < bests_num:
            # 先加满
            bests_missing_pregt.append((dist2, filename,i))
            bests_missing_gtpre.append((dist1, filename,i))
            bests_overall_pregt.append((dist2_all, filename,i))
            bests_overall_gtpre.append((dist1_all, filename,i))
            bests_missing_cd.append((dist_all, filename,i))
            bests_all_cd.append((dist_all_all, filename,i))
        elif have_heaped == False and len(bests_missing_pregt) == bests_num:
            heapq.heapify(bests_missing_pregt)
            heapq.heapify(bests_missing_gtpre)
            heapq.heapify(bests_overall_pregt)
            heapq.heapify(bests_overall_gtpre)
            heapq.heapify(bests_missing_cd)
            heapq.heapify(bests_all_cd)
            have_heaped = True
            print("once is ok")
        else:
            # 堆中元素的个数达到要求后，进行push
            heapq.heappushpop(bests_missing_pregt, (dist2, filename,i))
            heapq.heappushpop(bests_missing_gtpre, (dist1, filename,i))
            heapq.heappushpop(bests_overall_pregt, (dist2_all, filename,i))
            heapq.heappushpop(bests_overall_gtpre, (dist1_all, filename,i))
            heapq.heappushpop(bests_missing_cd, (dist_all, filename,i))
            heapq.heappushpop(bests_all_cd, (dist_all_all, filename,i))

        # if max_dist<dist1:
        #     max_dist=dist1
        #     max_index=i
        # if min_dist>dist1:
        #     min_dist=dist1
        #     min_index = i
        # if dist1<0.3:
        #     listG.append(i)
        #     fGood.write('\n' + str(i)+","+'ALL:  %.4f'%(dist_all )+","+'GT_PRE:  %.4f'%(dist1 ) +","+'PRE_GT:  %.4f'%(dist2 ) )
        # if dist1>10:
        #     listB.append(i)
        #     fBad.write('\n' + str(i)+","+'ALL:  %.4f'%(dist_all)+","+'GT_PRE:  %.4f'%(dist1 ) +","+'PRE_GT:  %.4f'%(dist2) )
        # if i%50==0:
        #     print(str(i)+"samples has been processed!\n"+
        #           'Max dist:  %.4f'%(max_dist )+" "+'Min_dist:  %.4f'%(min_dist ))

    # print(len(listG),len(listB))
    # print('Max dist:  %.4f'%(max_dist)+" "+'Max index:  %.4f'%(max_index)+"\n"+
    #       'MIn dist:  %.4f'%( min_dist)+" "+'Min index:  %.4f'%(min_index))


    # plt.figure("Image")
    # plt.imshow(np.transpose(listI[0], (1, 2, 0)))
    #
    # pyplot_draw_point_cloud(listI[0], listR[0], listG[0], 0, 0)
    # plt.show()
    # 写入文件
    root_path = '/home/dream/study/codes/PCCompletion/PFNet/PF-Net-Point-Fractal-Network/exp/best_three/02958343'
    f0 = open(os.path.join(root_path, opt.class_choice, '_bests_overall_pregt_test_result.txt'), 'a')
    f1 = open(os.path.join(root_path, opt.class_choice, '_bests_overall_gtpre_test_result.txt'), 'a')
    f2 = open(os.path.join(root_path, opt.class_choice, '_bests_missing_pregt_test_result.txt'), 'a')
    f3 = open(os.path.join(root_path, opt.class_choice, '_bests_missing_gtpre_test_result.txt'), 'a')
    f4 = open(os.path.join(root_path, opt.class_choice, '_bests_missing_cd_test_result.txt'), 'a')
    f5 = open(os.path.join(root_path, opt.class_choice, '_bests_all_cd_test_result.txt'), 'a')
    print("开始将最好的结果写入文件")
    for i in range(bests_num):
        f0.write('\n' + 'bests_overall_pregt: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_overall_pregt[i][0]), bests_overall_pregt[i][1],i))
        f1.write('\n' + '_bests_overall_gtpre: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_overall_gtpre[i][0]), bests_overall_gtpre[i][1],i))
        f2.write('\n' + '_bests_missing_pregt: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_missing_pregt[i][0]), bests_missing_pregt[i][1],i))
        f3.write('\n' + '_bests_missing_gtpre: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_missing_gtpre[i][0]), bests_missing_gtpre[i][1],i))
        f4.write('\n' + '_bests_missing_cd: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_missing_cd[i][0]), bests_missing_cd[i][1],i))
        f5.write('\n' + '_bests_all_cd: %.4f Obj_id: %s, Obj_index: %d'
                 % (float(bests_all_cd[i][0]), bests_all_cd[i][1],i))
    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    print("文件写入完毕")