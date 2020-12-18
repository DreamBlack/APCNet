#!/bin/bash
source activate pytorch-gpu  &&
cd /home/dream/study/codes/PCCompletion/APCNet  &&

python Train_PFNet_vanilia_Four.py --class_choice=Lamp --expdir=/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/vanilia/lamp &&

python Train_PFNet_withimage.py --class_choice=Lamp --expdir=/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/with_image/lamp &&

python Train_PFNet_vanilia_Four.py --class_choice=Car --expdir=/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/vanilia/car &&

python Train_PFNet_withimage.py --class_choice=Car --expdir=/home/dream/study/codes/PCCompletion/best_four_exp/pfnet/with_image/car

