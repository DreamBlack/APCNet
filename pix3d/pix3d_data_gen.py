import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import cv2
import json
from DatasetGeneration import  dividePointCloud
import matplotlib
data_dir_imgs = '/home/dream/study/codes/PCCompletion/datasets/pix3d/pix3d'
data_out_father='/home/dream/study/codes/PCCompletion/datasets/my_pix3d/four'
HEIGHT = 128
WIDTH = 128
PAD = 35
choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
                  torch.Tensor([-1, 1, 0])]
def get_pix3d_models(cat):
    with open(os.path.join(data_dir_imgs, 'pix3d.json'), 'r') as f:
        models_dict = json.load(f)
    models = []

    cats = [cat] # 'sofa'['chair', 'table']

    # Check for truncation and occlusion before adding a model to the evaluation list
    for d in models_dict:
        if d['category'] in cats:
            if not d['truncated'] and not d['occluded'] and not d['slightly_occluded']:
                models.append(d)

    print('Total models = {}\n'.format(len(models)))
    return models


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
catname_lower={'Chair':"chair","Table":'table'}
def generateFour(class_choice):
    models = get_pix3d_models(catname_lower[class_choice])
    cnt=0
    for ind in range(len(models)):
        model_path, file = os.path.split(os.path.join(data_dir_imgs, models[ind]['model']))

        _dict = models[ind]
        img_path = os.path.join(data_dir_imgs, _dict['img'])
        mask_path = os.path.join(data_dir_imgs, _dict['mask'])
        bbox = _dict['bbox']  # [width_from, height_from, width_to, height_to]
        pcl_path_1K = model_path.split('.')[0] + "/model.pts"  # point cloud path
        ip_image = cv2.imread(img_path)
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path) != 0
        ip_image = ip_image * mask_image
        ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        current_size = ip_image.shape[:2]  # current_size is in (height, width) format
        ratio = float(HEIGHT - PAD) / max(current_size)
        new_size = tuple([int(x * ratio) for x in current_size])
        ip_image = cv2.resize(ip_image,
                              (new_size[1], new_size[0]))  # new_size should be in (width, height) format
        delta_w = WIDTH - new_size[1]
        delta_h = HEIGHT - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        ip_image = cv2.copyMakeBorder(ip_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        xangle = np.pi / 180. * -90
        yangle = np.pi / 180. * -90
        #pcl_2025 = rotate(rotate(np.loadtxt(pcl_path_1K).astype(np.float32), xangle, yangle), xangle)
        pcl_2025 = np.loadtxt(pcl_path_1K).astype(np.float32)
        pcl_2025 = pc_normalize(pcl_2025)
        pcl_2025 = torch.from_numpy(pcl_2025)

        pcl_256 = torch.FloatTensor(256, 3)
        start = 0
        num_each_group = [50, 50, 50, 50, 56]
        left_pc_num = 1280
        pcl_1024 = pcl_2025.float()
        for i in range(len(choice)):
            left_pc_num = left_pc_num - num_each_group[i]
            part_gt, pcl_1024 = dividePointCloud(pcl_1024, choice[i], num_each_group[i], left_pc_num)
            for index in range(start, start + num_each_group[i]):
                pcl_256.data[index] = part_gt[index - start]
            start = start + num_each_group[i]

        # To Pytorch
        incomplete = pcl_1024
        gt = pcl_256
        #print(model_path)
        gt_filename = os.path.join(data_out_father, 'pc256', str(ind)+".pts")
        incomplete_filename = os.path.join(data_out_father, 'pc1024', str(ind)+".pts" )
        pc2025_filename = os.path.join(data_out_father, 'pc2025', str(ind) + ".pts")
        image_filename= os.path.join(data_out_father, 'image_clean', str(ind) + ".png")

        np.savetxt(gt_filename, gt.numpy())
        np.savetxt(incomplete_filename, incomplete.numpy())
        np.savetxt(pc2025_filename, pcl_2025.numpy())
        matplotlib.image.imsave(image_filename,ip_image)
        cnt = cnt + 1
        if cnt % 100 == 0:
            print("Succeed generate gt and incomplete for %s [%d/%d]" % (class_choice, cnt, len(models)))


class Pix3DSingleDataset(data.Dataset):
    def __init__(self,  class_choice=None):
        self.cat = class_choice
        models = get_pix3d_models(catname_lower[self.cat])
        self.models=models

    def __getitem__(self, index):
        ind=int(index/5)
        center=index%5
        model_path, file = os.path.split(os.path.join(data_dir_imgs, self.models[ind]['model']))

        _dict = self.models[ind]
        img_path = os.path.join(data_dir_imgs, _dict['img'])
        mask_path = os.path.join(data_dir_imgs, _dict['mask'])
        bbox = _dict['bbox']  # [width_from, height_from, width_to, height_to]
        pcl_path_1K = model_path.split('.')[0] + "/model.pts"  # point cloud path
        ip_image = cv2.imread(img_path)
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path) != 0
        ip_image = ip_image * mask_image
        ip_image = ip_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        current_size = ip_image.shape[:2]  # current_size is in (height, width) format
        ratio = float(HEIGHT - PAD) / max(current_size)
        new_size = tuple([int(x * ratio) for x in current_size])
        ip_image = cv2.resize(ip_image,
                              (new_size[1], new_size[0]))  # new_size should be in (width, height) format
        delta_w = WIDTH - new_size[1]
        delta_h = HEIGHT - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        ip_image = cv2.copyMakeBorder(ip_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        xangle = np.pi / 180. * -90
        yangle = np.pi / 180. * -90
        #pcl_2025 = rotate(rotate(np.loadtxt(pcl_path_1K).astype(np.float32), xangle, yangle), xangle)
        pcl_2025=np.loadtxt(pcl_path_1K).astype(np.float32)
        pcl_2025=torch.from_numpy(pcl_2025)
        pcl_256, pcl_1024 = dividePointCloud(pcl_2025.float(), choice[center])

        # To Pytorch
        incomplete = pcl_1024
        gt = pcl_256
        image = image = torch.from_numpy(np.transpose(ip_image, (2, 0, 1)))
        return incomplete, gt, image,pcl_2025,img_path


    def __len__(self):
        return len(self.models)*5

import matplotlib.pyplot as plt
if __name__ == '__main__':
    generateFour("Table")
    #    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    # print(len(dset))
    # incomplete,gt, image, filename= dset[1000]
    #
    # print(incomplete.size())
    # print(gt.size())
    # print(image.size())
#    print(incomplete)
#    ps = ps.numpy()
    #np.savetxt('ps'+'.xyz', incomplete, fmt = "%f %f %f")
