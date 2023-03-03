#!/bin/bash
# Usage: Create pdf files from input (wrapper script)
# Author: Vivek Gite <Www.cyberciti.biz> under GPL v2.x+
#---------------------------------------------------------

#Input file
_db="shell/corruption_list.txt"

# read it

while IFS=':' read -r corruption s
do
  severities=($s)
  for level in 0 1 2 3 4
    do
    echo $corruption ${severities[$level]}
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3.7 teco_kinetics_prior.py --threed_data --dataset mini_kinetics400 --frames_per_group 1 \
    --groups 16 \
    --logdir snapshots/ \
    --pretrained /home/yichenyu/action-recognition-pytorch-master/snapshots/mini_kinetics400-rgb-i3d-resnet-18-ts-f16-cosine-bs32-e50-v1/model_best.pth.tar \
    --backbone_net i3d_resnet_non_local -d 18 -b 32 -j 16 -e \
    --name MiniKinetics-TeCo-b32-lre-3-beta1-layer2-prior0.2 --lr 0.001 \
    --corruption $corruption \
    --severity ${severities[$level]} \
    --beta 1 \
    --prior 0.2
  done
done <"$_db"
