import json
from pathlib import Path

import torch
import torch.utils.data as data
import h5py
import os
import pdb
import numpy as np
import io
from PIL import Image

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDatasetHdf5(data.Dataset):

    def __init__(self,
                 root_path,
                 dst_path,
                 annotation_path,
                 subset,
                 corruption_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: 'image_{:05d}.jpg'.format(x),
                 target_type='label'):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.dst_path=dst_path
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]

            if not video_path.exists():
                print(video_path)
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]-2))

            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        # if self.spatial_transform is not None:
            # self.spatial_transform.randomize_parameters()
            # clip = [self.spatial_transform(img) for img in clip]
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.corruption_transform is not None:
            clip = [self.corruption_transform(img) for img in clip]
            # clip=self.corruption_transform(clip)
        # print(clip[0].size)
        # clip=self.spatial_transform(clip)
        # print(len(clip))

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        # if self.temporal_transform is not None:
        #     frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        classname=str(path).split("/")[-2]

        hdf5_path = os.path.join(self.dst_path,classname,path.name)

        if not os.path.exists(os.path.join(self.dst_path,classname)):
            os.mkdir(os.path.join(self.dst_path, classname))

        dst_dir_path=Path(hdf5_path[:-5])
        dst_dir_path.mkdir(exist_ok=True)

        for n,img in enumerate(clip):
            img.save(os.path.join(dst_dir_path,'image_{:05d}.jpg'.format(n)),'JPEG',quality=80)


        with h5py.File(hdf5_path, 'w') as f:
            dtype = h5py.special_dtype(vlen='uint8')
            video = f.create_dataset('video',
                                     (len(frame_indices),),
                                     dtype=dtype)

        for i, file_path in enumerate(sorted(dst_dir_path.glob('*.jpg'))):
            with file_path.open('rb') as f:
                data = f.read()
            with h5py.File(hdf5_path, 'r+') as f:
                video = f['video']
                video[i] = np.frombuffer(data, dtype='uint8')

        for file_path in dst_dir_path.glob('*.jpg'):
            file_path.unlink()
        dst_dir_path.rmdir()


        clip=self.spatial_transform(clip)

        return clip,target


    def __len__(self):
        return len(self.data)