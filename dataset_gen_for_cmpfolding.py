# coding=utf-8
import torch
import random
import numpy as np
import os
import argparse
import json

pfnet_pc_father_path = '/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0'  # opt.pf_net_pointcloud_father_path
outpc_father_path = '/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/completepc'  # opt.generate_pc_father_path
train_split_path = '/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0'  # opt.train_split_path
help_path = train_split_path

def generatCharpterFourePCForCat(cat_id):
    cat_path = os.path.join(pfnet_pc_father_path, cat_id, "points")
    out_father_path = os.path.join(outpc_father_path, cat_id)
    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)
    cnt = 0
    num_each_group = [50, 50, 50, 50, 56]

    fns = sorted(os.listdir(cat_path))  # listdir返回目录下包含的文件和文件夹名字列表

    with open(os.path.join(help_path, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

    fns = [fn for fn in fns if fn[0:-4] in test_ids]
    print("start generate charpter four point clouds for %s total: %d " % (cat_id, len(fns)))

    for home, dirs, files in os.walk(cat_path):
        for pc_filename in files:
            if pc_filename in fns:
                left_pc_num = 1280
                choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                          torch.Tensor([-1, 0, 0]),
                          torch.Tensor([-1, 1, 0])]
                gt_filename = os.path.join(out_father_path, "gt" + pc_filename)
                incomplete_filename = os.path.join(out_father_path, "incomplete" + pc_filename)
                point_set = np.loadtxt(os.path.join(home, pc_filename)).astype(np.float32)
                point_set = torch.from_numpy(point_set)
                incompletePC = point_set
                if os.path.exists(gt_filename) and os.path.exists(incomplete_filename):
                    continue

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
                    print("Charpter4. Succeed generate gt and incomplete for %s [%d/%d]" % (cat_id, cnt, len(fns)))
                cnt = cnt + 1


def dividePointCloud(pc, numSum=2025):
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


    # real_point采样为1280个点
    pc = torch.unsqueeze(pc, 0)  # [N,3]->[1,N,3]
    pc_key_idx = farthest_point_sample(pc, numSum, RAN=False)
    pc = index_points(pc, pc_key_idx)
    pc = torch.squeeze(pc, 0)

    gtpc = torch.FloatTensor(numSum, 3)  # 这里还没有构成精确的incomplete大小
    gtpc = gtpc.data.copy_(pc)  # post-fixed with _ 表示是原地操作

    return gtpc



def generatPCForCat(cat_id):
    cat_path = os.path.join(pfnet_pc_father_path, cat_id, "points")
    out_father_path = os.path.join(outpc_father_path, cat_id)
    if not os.path.exists(out_father_path):
        os.makedirs(out_father_path)
    print("cat_path %s" % (cat_path))
    print("out_path %s" % (out_father_path))

    fns = sorted(os.listdir(cat_path))  # listdir返回目录下包含的文件和文件夹名字列表


    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # set() 函数创建一个无序不重复元素集,para is  可迭代对象对象；
    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join( train_split_path, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

    fns = [fn for fn in fns if fn[0:-4] in train_ids]

    cnt = 0

    for home, dirs, files in os.walk(cat_path):
        for pc_filename in files:
            if pc_filename in fns:

                gt_filename = os.path.join(out_father_path, "cmpgt" + pc_filename)

                point_set = np.loadtxt(os.path.join(home, pc_filename)).astype(np.float32)
                point_set = torch.from_numpy(point_set)
                gt = dividePointCloud(point_set)
                np.savetxt(gt_filename, gt)

                cnt = cnt + 1
                if cnt % 100 == 0:
                    print("Succeed generate gt and incomplete for %s [%d/%d]" % (cat_id, cnt, len(fns)))



def distance_squre(p1, p2):
    device = p1.device
    tensor = p1 - p1
    tensor = torch.FloatTensor(p1) - torch.FloatTensor(p2)
    val = tensor.mul(tensor)
    val = val.sum()
    return val

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

if __name__ == '__main__':

    # generatCharpterFourePCForCat("03636649")
    generatPCForCat("03636649")


