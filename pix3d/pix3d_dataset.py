import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import cv2
import json
from DatasetGeneration import  dividePointCloud

data_dir_imgs = '/home/dream/study/codes/PCCompletion/datasets/pix3d/pix3d'

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

def rotate(xyz, xangle=0, yangle=0, zangle=0):
	rotmat = np.eye(3)

	rotmat=rotmat.dot(np.array([
		[1.0,0.0,0.0],
		[0.0,np.cos(xangle),-np.sin(xangle)],
		[0.0,np.sin(xangle),np.cos(xangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(yangle),0.0,-np.sin(yangle)],
		[0.0,1.0,0.0],
		[np.sin(yangle),0.0,np.cos(yangle)],
		]))

	rotmat=rotmat.dot(np.array([
		[np.cos(zangle),-np.sin(zangle),0.0],
		[np.sin(zangle),np.cos(zangle),0.0],
		[0.0,0.0,1.0]
		]))

	return xyz.dot(rotmat)


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
catname_lower={'Chair':"chair","Table":'table'}
data_out_father='/home/dream/study/codes/PCCompletion/datasets/my_pix3d/four'
class Pix3DMultiDataset(data.Dataset):
    def __init__(self,  class_choice=None):
        self.cat = class_choice
        models = get_pix3d_models(catname_lower[self.cat])
        self.models=models

    def __getitem__(self, index):
        ind=index
        model_path, file = os.path.split(os.path.join(data_dir_imgs, self.models[ind]['model']))

        pcl_2025 = np.loadtxt(os.path.join(data_out_father,catname_lower[self.cat],'pc2025',str(ind)+'.pts')).astype(np.float32)
        pcl_2025=torch.from_numpy(pcl_2025)

        pcl_256 = np.loadtxt(os.path.join(data_out_father,catname_lower[self.cat], 'pc256', str(ind) + '.pts')).astype(np.float32)
        pcl_256 = torch.from_numpy(pcl_256)

        pcl_1024 = np.loadtxt(os.path.join(data_out_father,catname_lower[self.cat], 'pc1024', str(ind) + '.pts')).astype(np.float32)
        pcl_1024 = torch.from_numpy(pcl_1024)

        image_path=os.path.join(data_out_father,catname_lower[self.cat], 'image_clean', str(ind) + '.png')
        ip_image = cv2.imread(image_path)
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

        incomplete = pcl_1024
        gt = pcl_256
        image = torch.from_numpy(np.transpose(ip_image, (2, 0, 1)))
        return incomplete, gt, image,pcl_2025,image_path


    def __len__(self):
        return len(self.models)


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
        pcl_2025 = rotate(rotate(np.loadtxt(pcl_path_1K).astype(np.float32), xangle, yangle), xangle)
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
    dset = Pix3DMultiDataset(class_choice='table')
    assert dset
    dataloader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False, num_workers=16)
    # for epoch in range(0, 10):
    #     for i, data in enumerate(dataloader, 0):
    #         incomplete, gt, image,  cmpgt = data
    #         np.savetxt('cmppc' + '.pts', cmpgt[0], fmt = "%f %f %f")
    #         break
    incomplete, gt, image,  cmpgt = dset[200]

    print(cmpgt.size())
    print(gt.size())
    print(image.size())
    plt.imshow(image)
    plt.show()
    np.savetxt('cmppc' + '.pts', cmpgt, fmt="%f %f %f")
    np.savetxt('incomplete' + '.pts', incomplete, fmt="%f %f %f")
    np.savetxt('gt' + '.pts', gt, fmt="%f %f %f")
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
