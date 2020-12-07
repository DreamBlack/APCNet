import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from DatasetGeneration import farthest_point_sample,index_points

#  /home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points
help_path='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0'
dsimage_path='/home/dream/study/codes/densepcr/datasets/ShapeNet/ShapeNetRendering/'
dataset_pathT='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerate/'
dataset_pathF='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/datagenerateForFour/'
plt.figure()
class MyDataset(data.Dataset):
    def __init__(self, root=dataset_pathT, three=True, classification=False, class_choice=None, split='train'):
        if not three:
            root=dataset_pathF
        self.root = root # pointcloud path
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.image_path=dsimage_path
        self.help_path=help_path

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()  # line.strip() remove the head and tail  or tab
                self.cat[ls[0]] = ls[1]  # ls[0]=name, ls[1]=cat_id

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}  # meta是个字典，key是 incomplete点云路径，gt点云路径，图片路径，类别名，objid的列表
        # train_id val test are all obj_id
        # 如果是沙发，花瓶，柜子，ｔｒａｉｎ　ｓｐｌｉｔ这些文件地址需要改一下
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # set() 函数创建一个无序不重复元素集,para is  可迭代对象对象；
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:  # item is name eg. lamp
            self.meta[item] = []

            #  self.cat[item] is id
            dir_point = os.path.join(self.root, self.cat[item])
            dir_image = os.path.join(self.image_path, self.cat[item])
            dir_help=os.path.join(self.help_path, self.cat[item],'points')

            fns = sorted(os.listdir(dir_help))  # listdir返回目录下包含的文件和文件夹名字列表

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                sys.exit(-1)

            # table 数据集太大了，选其中几个就好了
            # if class_choice == 'Table':
            #     if split == 'train':
            #         fns = fns[:2658]
            #     elif split == 'val':
            #         fns = fns[:396]
            #     elif split == 'test':
            #         fns = fns[:704]
            #     else:
            #         print('Unknown split: %s. Exiting..' % (split))

            print(split+"_size %d"%(len(fns)))  # 1118*5*8=44720
            for fn in fns:  # 点云路径，图片路径，类别名，objid
                token = (os.path.splitext(os.path.basename(fn))[0])  # basename返回最后的文件名，token获得的是前缀文件名，也就是objid
                image_view = ['00', '09', '11', '18', '22']

                good_center=[]
                if class_choice == 'Table':
                    image_view = ['00', '09', '18']

                if three:
                    if class_choice=='Lamp':
                        good_center=[0,1,2,6,7]
                    else:
                        good_center = [0, 1, 2, 3, 4]
                    for center in good_center:
                        for view in image_view:
                            self.meta[item].append(
                                (os.path.join(dir_point, str(center) + 'incomplete' + token + '.pts'),
                                 os.path.join(dir_point, str(center) + 'gt' + token + '.pts'),
                                 os.path.join(dir_image, token, 'rendering', view + '.png'),
                                 self.cat[item],
                                 token))
                    # for center in range(8):
                    #     for view in image_view:
                    #         self.meta[item].append(
                    #             (os.path.join(dir_point, str(center) + 'incomplete' + token + '.pts'),
                    #              os.path.join(dir_point, str(center) + 'gt' + token + '.pts'),
                    #              os.path.join(dir_image, token, 'rendering', view + '.png'),
                    #              self.cat[item],
                    #              token))
                else:
                    for view in image_view:
                        self.meta[item].append(
                            (os.path.join(dir_point, 'incomplete' + token + '.pts'),
                             os.path.join(dir_point, 'gt' + token + '.pts'),
                             os.path.join(dir_image, token, 'rendering', view + '.png'),
                             self.cat[item],
                             token))

        self.datapath = []  #  所有path的总和
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3],fn[4]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))  # zip(a,b)将两个元组打包为列表909033,0的形式，dict又将其变为字典

        # 不知道这个classification函数是用来干嘛的，应该是作为bianbie器，这个我暂时不需要，不改了
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        # print(self.num_seg_classes)
        self.cache = {}  # from index to (in, gt,image) tuple
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            #            in, gt,image= self.cache[index]
            incomplete, gt, image, foldername, filename= self.cache[index]
        else:
            fn = self.datapath[index]  # filename
            cls = self.classes[self.datapath[index][0]]
            #            cls = np.array([cls]).astype(np.int32)
            incomplete = np.loadtxt(fn[1]).astype(np.float32)

            gt = np.loadtxt(fn[2]).astype(np.float32)
            image = cv2.imread(fn[3])[4:-5, 4:-5, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            foldername = fn[4]
            filename = fn[5]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (incomplete, gt, image, foldername, filename)

        # print(point_set.shape, seg.shape)
        # choice = np.random.choice(len(seg), self.npoints, replace=True)  # random.choic(a，size，true),返回a中大小为size的随机一项，有放回
        # resample
        # point_set = point_set[choice, :]
        # seg = seg[choice]

        # To Pytorch
        incomplete = torch.from_numpy(incomplete)
        gt = torch.from_numpy(gt)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        if self.classification:
            return incomplete, gt, image, filename
        else:
            return incomplete, gt, image, filename

    def __len__(self):
        return len(self.datapath)

    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


if __name__ == '__main__':
    dset = MyDataset( classification=True,three=True,
                       class_choice='Table', split='test')
    #    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    incomplete,gt, image, filename= dset[1000]

    print(incomplete.size())
    print(gt.size())
    print(image.size())
#    print(incomplete)
#    ps = ps.numpy()
    #np.savetxt('ps'+'.xyz', incomplete, fmt = "%f %f %f")
