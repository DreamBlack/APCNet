# coding=utf-8
import torch
import random
from torch.autograd import Variable
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

def generatCharpterFourePCForCat(cat_id):
    pc_father_path="/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/"
    cat_path=os.path.join(pc_father_path,cat_id,"points")
    out_father_path=os.path.join("/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerateForFour",cat_id)

    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)

    cnt=0
    num_each_group = [50, 50, 50, 50, 56]

    for home,dirs,files in os.walk(cat_path):
        for pc_filename in files:
            left_pc_num = 1280
            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                      torch.Tensor([-1, 0, 0]),
                      torch.Tensor([-1, 1, 0])]
            gt_filename = os.path.join(out_father_path, "gt" + pc_filename)
            incomplete_filename = os.path.join(out_father_path, "incomplete" + pc_filename)
            point_set = np.loadtxt(os.path.join(home, pc_filename)).astype(np.float32)
            point_set = torch.from_numpy(point_set)
            incompletePC=point_set
            gt=torch.FloatTensor(256, 3)
            start=0
            for i in range(len(choice)):
                left_pc_num=left_pc_num-num_each_group[i]
                part_gt, incompletePC = dividePointCloud(incompletePC, choice[i],num_each_group[i],left_pc_num)
                for index in range(start,start+num_each_group[i]):
                    gt.data[index]=part_gt[index-start]
                start=start+num_each_group[i]
            np.savetxt(gt_filename, gt)
            np.savetxt(incomplete_filename, incompletePC)

            if cnt%50==0:
                print("Charpter4. Succeed generate gt and incomplete for %s [%d/%d]"%(cat_id,cnt,len(files)))
            cnt=cnt+1


def dividePointCloud(pc,center,missingNum=256,incompleteNum=1024):
    '''
    Input:
        pc: complete point cloud [N,C]
        center: in which view to cut 1/4 of the pc
        missingNum:missingPC number
        incompleteNum: number of incomplete PC
    Return:
        missingPC: [missingNum,C]
        incompletePC: [incompleteNum,C]
    '''

    num_sum=missingNum+incompleteNum

    # real_point采样为1280个点
    pc = torch.unsqueeze(pc, 0)  # [N,3]->[1,N,3]
    pc_key_idx = farthest_point_sample(pc, num_sum, RAN=False)
    pc = index_points(pc, pc_key_idx)
    pc = torch.squeeze(pc, 0)

    missingPC = torch.FloatTensor(missingNum, 3)
    incompletePC = torch.FloatTensor(num_sum, 3)  # 这里还没有构成精确的incomplete大小
    incompletePC = incompletePC.data.copy_(pc)  # post-fixed with _ 表示是原地操作

    distance_list = []

    for n in range(num_sum):
        distance_list.append(distance_squre(pc[n], center))
    distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

    # sorted返回的是重新排序的列表以及下标,nums[0]=data's index in distance_list,nums[1]=data
    for sp in range(missingNum):

        # distance_order[sp][0]为这个值在list里面原来的下标，也就是在realpoint中的下标
        missingPC.data[sp] = pc[distance_order[sp][0]]  # 离中心点进度在missing中

    for sp in range(missingNum,num_sum):
        incompletePC.data[sp] = pc[distance_order[sp][0]]  # 离中心点进度在missing中

    return missingPC, incompletePC.data[missingNum:]

def generatPCForCat(cat_id):
    pc_father_path="/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/"
    cat_path=os.path.join(pc_father_path,cat_id,"points")
    out_father_path=os.path.join("/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate",cat_id)

    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)

    cnt=0
    for home,dirs,files in os.walk(cat_path):
        for pc_filename in files:
            choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]

            for i in range(len(choice)):
                gt_filename=os.path.join(out_father_path,str(i)+"gt"+pc_filename)
                incomplete_filename = os.path.join(out_father_path, str(i) + "incomplete"+ pc_filename)

                point_set = np.loadtxt(os.path.join(home,pc_filename)).astype(np.float32)
                point_set = torch.from_numpy(point_set)
                gt, incompletePC = dividePointCloud(point_set, choice[i])

                np.savetxt(gt_filename, gt)
                np.savetxt(incomplete_filename, incompletePC)

            cnt=cnt+1
            if cnt%100==0:
                print("Succeed generate gt and incomplete for %s [%d/%d]"%(cat_id,cnt,len(files)))

def generateMissingPCAndIncompletePC(real_point,center,missingNum=256,incompleteNum=1024):
    '''
    Input:
        real_point: pointcloud data, [B,N,C]
        center: center of view generate missing pc 3*1
        missingNum:missingPC number
        incompleteNum: number of incomplete PC
    Return:
        missingPC: [B, missingNum,C]
        incompletePC: [B,incompleteNum,C]
    '''
    batch_size=real_point.shape[0]

    # real_point采样为1280个点
    real_point_key_idx = farthest_point_sample(real_point, 1280, RAN=False)
    real_point = index_points(real_point, real_point_key_idx)

    num_size = real_point.shape[1]
    missingPC = torch.FloatTensor(batch_size, 1, missingNum, 3)
    incompletePC = torch.FloatTensor(batch_size, real_point.shape[1], 3)  # 这里还没有构成精确的incomplete大小

    incompletePC = incompletePC.data.copy_(real_point) # post-fixed with _ 表示是原地操作
    real_point = torch.unsqueeze(real_point, 1) # [B,N,3]->[B,1,N,3]
    incompletePC = torch.unsqueeze(incompletePC, 1) # [B,N,3]->[B,1,N,3]

    p_origin = [0, 0, 0]

    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
    for m in range(batch_size):
        index = random.sample(choice, 1)
        distance_list = []

        # p_center = index[0]
        p_center=center

        for n in range(num_size):
            distance_list.append(distance_squre(real_point[m, 0, n], p_center))
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
        # sorted返回的是重新排序的列表以及下标,nums[0]=data's index in distance_list,nums[1]=data

        for sp in range(missingNum):
            # distance_order[sp][0]为这个值在list里面原来的下标，也就是在realpoint中的下标
            incompletePC.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])  # 原理incomplete里面点都改为0
            missingPC.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]] # 离中心点进度在missing中


    missingPC=torch.squeeze(missingPC,1)
    missingPC = missingPC.to(device)
    incompletePC = incompletePC.to(device)

    incompletePC = torch.squeeze(incompletePC, 1)

    # incompletePC_key1_idx = farthest_point_sample(incompletePC, 1024, RAN=False)
    # incompletePC = index_points(incompletePC, incompletePC_key1_idx)
    return missingPC, incompletePC


def distance_squre(p1,p2):
    device = p1.device
    tensor=p1-p1
    val=tensor.mul(tensor)
    val = val.sum()
    return val

def pc_normalize(pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)


    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def funtest():
    # try to open .npy
    pc_path='/home/dream/study/codes/densepcr/datasets/pointcloud/2048/00.xyz'
    point_set = np.loadtxt(pc_path).astype(np.float32)
    # point_set=pc_normalize(point_set)
    print('----------111111-------\n')
    print(point_set.shape)

    point_set = torch.from_numpy(point_set)
    print('----------111111-------\n')
    print(point_set.shape)

    point_set = torch.unsqueeze(point_set, 0)  # 1*4096*3
    print('----------111111-------\n')
    print(point_set.shape)

    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([-1, 0, 0]), torch.Tensor([0, 0, -1]),
              torch.Tensor([0, -1, 0]), torch.Tensor([0, 1, 0]), torch.Tensor([1, 0, 1]),
              torch.Tensor([-1, 1, 0])]

    for i in range(len(choice)):
        gt, incompletePC = generateMissingPCAndIncompletePC(point_set, choice[i])
        # 1a32f10b20170883663e90eaf6b4ca52
        np.savetxt(os.path.join("/home/dream/study/codes/PCCompletion/datasets/1a04e3eab45ca15dd86060f189eb133/",
                                str(i) + "_gt.xyz"), gt.cpu()[0])
        np.savetxt(os.path.join("/home/dream/study/codes/PCCompletion/datasets/1a04e3eab45ca15dd86060f189eb133/",
                                str(i) + "_incompletePC.xyz"), incompletePC.cpu()[0])

    print('----------end-------\n')
    print(gt.shape)
    print(incompletePC.shape)

if __name__=='__main__':
    generatCharpterFourePCForCat("03636649")
    #generatPCForCat("02958343")
    #generatPCForCat("03001627")


