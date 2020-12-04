#  ["shape_data/02691156/be02fd401a12d23822362be7f5226e91",]
import random
import os
import json
import shutil
print(random.randint(0,1))
cat="03001627" #04530566,02933112
val=[]
test=[]
train=[]
pc_dir='/home/dream/study/codes/PCCompletion/datasets/dataFromPFNet/shapenet_part/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/'
pc_dir=os.path.join(pc_dir,cat,"points")
image_dir='/home/dream/study/codes/densepcr/datasets/rendering'
source_dir='/home/dream/study/codes/densepcr/datasets/ShapeNet/ShapeNetRendering'
image_dir=os.path.join(image_dir,cat)
source_dir=os.path.join(source_dir,cat)
fns = sorted(os.listdir(pc_dir))  # listdir返回目录下包含的文件和文件夹名字列表
for i,fn in enumerate(fns):
    obj_id = (os.path.splitext(os.path.basename(fn))[0])
    sourcepath=os.path.join(source_dir,obj_id,"rendering")
    destpath=os.path.join(image_dir,obj_id,"rendering")
    image_view = ['00', '09', '11', '18', '22']
    for view in image_view:

        nowsourcepath=os.path.join(sourcepath,view+'.png')
        if not os.path.exists(destpath):
            os.makedirs(destpath)
        nowdestpath = os.path.join(destpath, view + '.png')

        shutil.copyfile(nowsourcepath,nowdestpath)


