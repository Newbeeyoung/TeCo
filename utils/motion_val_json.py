import argparse
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd

import h5py


def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])

def get_n_frames_hdf5(video_path):
    with h5py.File(video_path, 'r') as f:
        video_data = f['video']
        return len(video_data)

def get_n_motion_frames_hdf5(video_path,n):
    with h5py.File(video_path, 'r') as f:
        video_data = f['video']
        motion_magnitudes=[]
        for motion_map in video_data:
            motion_mag = np.asarray(motion_map).sum()
            motion_magnitudes.append(motion_mag)

        largest_n = sorted(range(len(motion_magnitudes)), key=lambda x: motion_magnitudes[x])[-n:]
        return len(video_data), largest_n

def generate_json_from_folder(json_path, video_dir_path,dst_json_path):

    dst_data = {}
    # dst_data['labels'] = os.listdir(video_dir_path)
    # labels = load_labels(train_csv_path)
    # dst_data['labels'] = [label.replace(" ","_") for label in labels]
    # class_list=load_classes("mini-kinetics-200-classes.txt")
    class_list=load_classes("../data/mini-kinetics-200-classes.txt")
    # dst_data['labels']=class_list
    # dst_data['database'] = {}

    with json_path.open('r') as f:
        data = json.load(f)

    n=0
    for label in sorted(os.listdir(str(video_dir_path))):
    # for label in class_list:
    #     print(label)
        for clip_name_path in os.listdir(os.path.join(str(video_dir_path),label)):
            if clip_name_path[-5:]==".hdf5":
                clip_name = clip_name_path[:-5]
                # dst_data['database'][clip_name]={"subset":"validation"}

                video_path = video_dir_path / label / clip_name_path
                if video_path.exists():
                    n_frames, largest_n = get_n_motion_frames_hdf5(video_path, 16)
                    # print(n_frames)
                    # dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 6),'motion_index_16':largest_n}
                    # dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 1)}
                    data['database'][clip_name]['annotations']['motion_index_16'] = largest_n
                n+=1
                if n%500==0:
                    print(n)
                #
                # try:
                #     clip_name=clip_name_path[:-5]
                #     # dst_data['database'][clip_name]={"subset":"validation"}
                #
                #     video_path = video_dir_path / label / clip_name_path
                #     if video_path.exists():
                #
                #         n_frames, largest_n= get_n_motion_frames_hdf5(video_path,16)
                #         # print(n_frames)
                #         # dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 6),'motion_index_16':largest_n}
                #         # dst_data['database'][clip_name]['annotations']={'label':label,'segment':(1, n_frames + 1)}
                #         data['database'][clip_name]['annotation']['motion_index_16']=largest_n
                #
                # except:
                #     print(video_dir_path,label,clip_name)
                #     pass
    with dst_json_path.open('w') as dst_file:
        json.dump(data, dst_file)

def load_classes(classes_txt_path):

    class_list=[]
    f=open(classes_txt_path,'r')

    for classname in f.readlines():
        class_list.append(classname.strip("\n"))
    return class_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    # parser.add_argument('video_path',
    #                     default=None,
    #                     type=Path,
    #                     help=('Path of video directory (jpg or hdf5).'
    #                           'Using to get n_frames of each video.'))
    # parser.add_argument('video_type',
    #                     default='hdf5',
    #                     type=str,
    #                     help=('jpg or hdf5'))
    # parser.add_argument('dst_path',
    #                     default=None,
    #                     type=Path,
    #                     help='Path of dst json file.')
    #
    args = parser.parse_args()

    args.video_type='hdf5'
    assert args.video_type in ['jpg', 'hdf5']

    MOTION_BASE_DIR="/home/yichenyu/Dataset/kinetics/mini_kinetics200-c_motion/motion_map"
    VAL_JSON_BASE_DIR = "/home/yichenyu/action-recognition-pytorch-master/data/mini_kinetics/"
    DST_DIR="/home/yichenyu/git/VideoTestTime/data/mini_kinetics"
    # BASE_DIR="/home/yichenyu/Dataset/ssv2/mini_ssv287-c"
    # DST_DIR="/home/yichenyu/action-recognition-pytorch-master/data/mini_ssv2"

    corruption_list=['shot_noise', 'fog']
    # corruption_list = ['clean', 'fog']
    for corruption in corruption_list:
        for severity in os.listdir(os.path.join(MOTION_BASE_DIR,corruption)):
            print(corruption,severity)
            args.video_path=os.path.join(MOTION_BASE_DIR,corruption,severity)
            args.dst_path=os.path.join(DST_DIR,"mini-kinetics200_motion_{}_{}_v2.json".format(corruption,severity))
            generate_json_from_folder(Path(os.path.join(VAL_JSON_BASE_DIR,"mini-kinetics200_{}_{}.json".format(corruption,severity))),Path(args.video_path), Path(args.dst_path))