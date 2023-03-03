# TeCo-Temporal Coherent Test-Time Optimization for Robust Video Classification (ICLR2023)

This repositery contains the code for [**Temporal Coherent Test-Time Optimization for Robust Video Classification, ICLR2023**](https://arxiv.org/pdf/2302.14309v1.pdf).

## Test Time Optimization and Evaluation
Adapt the pretrained model with test data. The command sample adapts the model with data corrupted by shot noise with severity of 3.

```buildoutcfg
python3 teco_kinetics.py --threed_data --dataset mini_kinetics400 --frames_per_group 1 \
    --groups 16 \
    --logdir snapshots/ \
    --pretrained /path/to/pretrained/model \
    --data_folder /path/to/corrupted/data \
    --backbone_net i3d_resnet_non_local -d 18 -b 32 -j 16 -e \
    --name MiniKinetics-TeCo-b32-lre-3 --lr 0.001 \
    --corruption shot_noise \
    --severity 3 \
    --beta 1 \
    --prior 0.2
```

## Dataset 

Go to another repository for creating video corruption dataset [link](https://github.com/Newbeeyoung/Video-Corruption-Robustness).

To Do: evaluate with Mini Kinetics Test Data which corrupted Shot_Noise-3, please download from [link](). 

