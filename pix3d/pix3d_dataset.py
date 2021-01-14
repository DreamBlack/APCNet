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
data_out_father='/home/dream/study/codes/PCCompletion/datasets/my_pix3d/'
class Pix3DMultiDataset(data.Dataset):
    def __init__(self,  class_choice=None):
        self.cat = class_choice
        models = get_pix3d_models(catname_lower[self.cat])
        self.models=models
        self.father_path=os.path.join(data_out_father,"four")

    def __getitem__(self, index):
        ind=index
        model_path, file = os.path.split(os.path.join(data_dir_imgs, self.models[ind]['model']))

        pcl_2025 = np.loadtxt(os.path.join(self.father_path,catname_lower[self.cat],'pc2025',str(ind)+'.pts')).astype(np.float32)
        pcl_2025=torch.from_numpy(pcl_2025)

        pcl_256 = np.loadtxt(os.path.join(self.father_path,catname_lower[self.cat], 'pc256', str(ind) + '.pts')).astype(np.float32)
        pcl_256 = torch.from_numpy(pcl_256)

        pcl_1024 = np.loadtxt(os.path.join(self.father_path,catname_lower[self.cat], 'pc1024', str(ind) + '.pts')).astype(np.float32)
        pcl_1024 = torch.from_numpy(pcl_1024)

        image_path=os.path.join(self.father_path,catname_lower[self.cat], 'image_clean', str(ind) + '.png')
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
        self.father_path = os.path.join(data_out_father, "three")

    def __getitem__(self, index):
        ind=int(index/5)
        center=index%5

        model_path, file = os.path.split(os.path.join(data_dir_imgs, self.models[ind]['model']))

        pcl_2025 = np.loadtxt(
            os.path.join(self.father_path, catname_lower[self.cat], 'pc2025', str(ind) + '.pts')).astype(np.float32)
        pcl_2025 = torch.from_numpy(pcl_2025)

        pcl_256 = np.loadtxt(os.path.join(self.father_path, catname_lower[self.cat], 'pc256', str(index) + '.pts')).astype(
            np.float32)
        pcl_256 = torch.from_numpy(pcl_256)

        pcl_1024 = np.loadtxt(
            os.path.join(self.father_path, catname_lower[self.cat], 'pc1024', str(index) + '.pts')).astype(np.float32)
        pcl_1024 = torch.from_numpy(pcl_1024)

        image_path = os.path.join(self.father_path, catname_lower[self.cat], 'image_clean', str(ind) + '.png')
        ip_image = cv2.imread(image_path)
        ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)

        incomplete = pcl_1024
        gt = pcl_256
        image = torch.from_numpy(np.transpose(ip_image, (2, 0, 1)))
        return incomplete, gt, image, pcl_2025, image_path


    def __len__(self):
        return len(self.models)*5

import matplotlib.pyplot as plt
if __name__ == '__main__':
    dset = Pix3DSingleDataset(class_choice='Table')
    assert dset
    #dataloader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False, num_workers=16)
    # for epoch in range(0, 10):
    #     for i, data in enumerate(dataloader, 0):
    #         incomplete, gt, image,  cmpgt = data
    #         np.savetxt('cmppc' + '.pts', cmpgt[0], fmt = "%f %f %f")
    #         break
    incomplete, gt, image, cmpgt, image_path = dset[0]

    print(cmpgt.size())
    print(gt.size())
    print(image.size())
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
