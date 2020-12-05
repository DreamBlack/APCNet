# coding=utf-8
import torch
import random
import numpy as np
import os
import argparse
import json
import sys
import threading

parser = argparse.ArgumentParser()  # create an argumentparser object
parser.add_argument('--class_choice', default='04379243', help="which class choose to train")
parser.add_argument('--four_dataset', type = bool, default = False, help='generate chapter 4 dataset')
parser.add_argument('--pf_net_pointcloud_father_path',default='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate', help="pf_net_pointcloud_father_path")
parser.add_argument('--generate_pc_father_path',default='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/', help="generate_pc_father_path(no need for three or four)")
parser.add_argument('--train_split_path', default='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',help="train_split_path")
opt = parser.parse_args()
print(opt)
#"/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/"
#"/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate"
pfnet_pc_father_path=opt.pf_net_pointcloud_father_path
outpc_father_path = opt.generate_pc_father_path
train_split_path=opt.train_split_path
def getFilenamesForTable(split,cat_path,train_split_path):
    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # set() 函数创建一个无序不重复元素集,para is  可迭代对象对象；
    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])


    fns = sorted(os.listdir(cat_path))  # listdir返回目录下包含的文件和文件夹名字列表

    fnsTrain = [fn for fn in fns if fn[0:-4] in train_ids]
    fnsVal = [fn for fn in fns if fn[0:-4] in val_ids]
    fnsTest = [fn for fn in fns if fn[0:-4] in test_ids]

    # table 数据集太大了，选其中几个就好了
    fnsTrain = fns[:2658]
    fnsVal = fns[:396]
    fnsTest = fns[:704]



    fns=fnsTrain+fnsTest+fnsVal
    if split == 'trainval':
        return fns+fnsVal
    elif split == 'train':
        print("Train_size %d" % (len(fnsTrain)))  # 1118*5*8=44720
        return fnsTrain
    elif split == 'val':
        print("Val_size %d" % (len(fnsVal)))  # 1118*5*8=44720
        return fnsVal
    elif split == 'test':
        print("Test_size %d" % (len(fnsTest)))  # 1118*5*8=44720
        return fnsTest
    else:
        print('Unknown split: %s. Exiting..' % (split))
        sys.exit(-1)
    return fns

def generatCharpterFourForTable(split):
    cat_path = os.path.join(pfnet_pc_father_path, "04379243", "points")
    out_father_path = os.path.join(outpc_father_path, "Four", "04379243")
    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)
    cnt = 0
    num_each_group = [50, 50, 50, 50, 56]

    fns=getFilenamesForTable(split,cat_path,train_split_path)

    for pc_filename in fns:
        left_pc_num = 1280
        choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                  torch.Tensor([-1, 0, 0]),
                  torch.Tensor([-1, 1, 0])]
        gt_filename = os.path.join(out_father_path, "gt" + pc_filename)
        incomplete_filename = os.path.join(out_father_path, "incomplete" + pc_filename)
        point_set = np.loadtxt(os.path.join(cat_path, pc_filename)).astype(np.float32)
        point_set = torch.from_numpy(point_set)
        incompletePC = point_set
        gt = torch.FloatTensor(256, 3)
        start = 0
        for i in range(len(choice)):
            left_pc_num = left_pc_num - num_each_group[i]
            part_gt, incompletePC = dividePointCloud(incompletePC, choice[i], num_each_group[i], left_pc_num)
            for index in range(start, start + num_each_group[i]):
                gt.data[index] = part_gt[index - start]
            start = start + num_each_group[i]
        np.savetxt(gt_filename, gt)
        np.savetxt(incomplete_filename, incompletePC)

        if cnt % 50 == 0:
            print("Charpter4. Succeed generate gt and incomplete for %s [%d/%d]" % ("04379243", cnt, len(fns)))
        cnt = cnt + 1

def generatCharpterFourePCForCat(cat_id):
    cat_path = os.path.join(pfnet_pc_father_path, cat_id, "points")
    out_father_path=os.path.join(outpc_father_path,"Four", cat_id)
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

def generatPCForTableCat(split):
    cat_path = os.path.join(pfnet_pc_father_path, "04379243", "points")
    out_father_path = os.path.join(outpc_father_path,"Three", "04379243")
    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)

    fns = getFilenamesForTable(split,cat_path,train_split_path)

    cnt = 0
    for pc_filename in fns:
        choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
                  torch.Tensor([-1, 1, 0])]
        for i in range(len(choice)):
            gt_filename = os.path.join(out_father_path, str(i) + "gt" + pc_filename)
            incomplete_filename = os.path.join(out_father_path, str(i) + "incomplete" + pc_filename)

            point_set = np.loadtxt(os.path.join(cat_path, pc_filename)).astype(np.float32)
            point_set = torch.from_numpy(point_set)
            gt, incompletePC = dividePointCloud(point_set, choice[i])

            np.savetxt(gt_filename, gt)
            np.savetxt(incomplete_filename, incompletePC)

        cnt = cnt + 1
        if cnt % 100 == 0:
            print("Succeed generate gt and incomplete for %s [%d/%d]" % ("04379243", cnt, len(fns)))

def generatPCForCat(cat_id):
    cat_path = os.path.join(pfnet_pc_father_path, cat_id, "points")
    out_father_path = os.path.join(outpc_father_path,"Three", cat_id)
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

# def generateMissingPCAndIncompletePC(real_point,center,missingNum=256,incompleteNum=1024):
#     '''
#     Input:
#         real_point: pointcloud data, [B,N,C]
#         center: center of view generate missing pc 3*1
#         missingNum:missingPC number
#         incompleteNum: number of incomplete PC
#     Return:
#         missingPC: [B, missingNum,C]
#         incompletePC: [B,incompleteNum,C]
#     '''
#     batch_size=real_point.shape[0]
#
#     # real_point采样为1280个点
#     real_point_key_idx = farthest_point_sample(real_point, 1280, RAN=False)
#     real_point = index_points(real_point, real_point_key_idx)
#
#     num_size = real_point.shape[1]
#     missingPC = torch.FloatTensor(batch_size, 1, missingNum, 3)
#     incompletePC = torch.FloatTensor(batch_size, real_point.shape[1], 3)  # 这里还没有构成精确的incomplete大小
#
#     incompletePC = incompletePC.data.copy_(real_point) # post-fixed with _ 表示是原地操作
#     real_point = torch.unsqueeze(real_point, 1) # [B,N,3]->[B,1,N,3]
#     incompletePC = torch.unsqueeze(incompletePC, 1) # [B,N,3]->[B,1,N,3]
#
#     p_origin = [0, 0, 0]
#
#     choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
#     for m in range(batch_size):
#         index = random.sample(choice, 1)
#         distance_list = []
#
#         # p_center = index[0]
#         p_center=center
#
#         for n in range(num_size):
#             distance_list.append(distance_squre(real_point[m, 0, n], p_center))
#         distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
#         # sorted返回的是重新排序的列表以及下标,nums[0]=data's index in distance_list,nums[1]=data
#
#         for sp in range(missingNum):
#             # distance_order[sp][0]为这个值在list里面原来的下标，也就是在realpoint中的下标
#             incompletePC.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])  # 原理incomplete里面点都改为0
#             missingPC.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]] # 离中心点进度在missing中
#
#
#     missingPC=torch.squeeze(missingPC,1)
#     missingPC = missingPC.to(device)
#     incompletePC = incompletePC.to(device)
#
#     incompletePC = torch.squeeze(incompletePC, 1)
#
#     # incompletePC_key1_idx = farthest_point_sample(incompletePC, 1024, RAN=False)
#     # incompletePC = index_points(incompletePC, incompletePC_key1_idx)
#     return missingPC, incompletePC


def distance_squre(p1,p2):
    device = p1.device
    tensor=p1-p1
    tensor=torch.FloatTensor(p1)-torch.FloatTensor(p2)
    val=tensor.mul(tensor)
    val = val.sum()
    return val

# def pc_normalize(pc):
#         """ pc: NxC, return NxC """
#         l = pc.shape[0]
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#         pc = pc / m
#         return pc

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

# def funtest():
#     # try to open .npy
#     pc_path='/home/dream/study/codes/densepcr/datasets/pointcloud/2048/00.xyz'
#     point_set = np.loadtxt(pc_path).astype(np.float32)
#     # point_set=pc_normalize(point_set)
#     print('----------111111-------\n')
#     print(point_set.shape)
#
#     point_set = torch.from_numpy(point_set)
#     print('----------111111-------\n')
#     print(point_set.shape)
#
#     point_set = torch.unsqueeze(point_set, 0)  # 1*4096*3
#     print('----------111111-------\n')
#     print(point_set.shape)
#
#     choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([-1, 0, 0]), torch.Tensor([0, 0, -1]),
#               torch.Tensor([0, -1, 0]), torch.Tensor([0, 1, 0]), torch.Tensor([1, 0, 1]),
#               torch.Tensor([-1, 1, 0])]
#
#     for i in range(len(choice)):
#         gt, incompletePC = generateMissingPCAndIncompletePC(point_set, choice[i])
#         # 1a32f10b20170883663e90eaf6b4ca52
#         np.savetxt(os.path.join("/home/dream/study/codes/PCCompletion/datasets/1a04e3eab45ca15dd86060f189eb133/",
#                                 str(i) + "_gt.xyz"), gt.cpu()[0])
#         np.savetxt(os.path.join("/home/dream/study/codes/PCCompletion/datasets/1a04e3eab45ca15dd86060f189eb133/",
#                                 str(i) + "_incompletePC.xyz"), incompletePC.cpu()[0])
#
#     print('----------end-------\n')
#     print(gt.shape)
#     print(incompletePC.shape)

if __name__=='__main__':
    if opt.class_choice=="04379243":
        if opt.four_dataset:
            print("start generate charpter four point clouds for table")
            generatCharpterFourForTable()
        else:
            print("start generate charpter three point clouds for table")
            threads=[threading.Thread(target=generatPCForTableCat,args=('train',)),
                     threading.Thread(target=generatPCForTableCat,args=('val',)),
                     threading.Thread(target=generatPCForTableCat,args=('test',))]

            for t in threads:
                print(t.name+"starts!")
                t.start()
    else:
        if opt.four_dataset:
            print("start generate charpter four point clouds for %s" % (opt.class_choice))
            generatCharpterFourePCForCat(opt.class_choice)
        else:
            print("start generate charpter three point clouds for %s"%(opt.class_choice))
            generatPCForCat(opt.class_choice)
    #generatPCForCat("02958343")
    #generatPCForCat("03001627")


