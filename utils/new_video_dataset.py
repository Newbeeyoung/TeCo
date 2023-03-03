import json
import pdb
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np

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


class NewVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
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
                 target_type='label',
                 average=False):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

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

            frame_indices = list(range(segment[0], max(2,segment[1]-2)))

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

        clip=self.spatial_transform(clip)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        # try:
        #     if self.temporal_transform is not None:
        #         frame_indices = self.temporal_transform(frame_indices)
        # except:
        #     print(path)
        #     print(frame_indices)
        #     pass

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if not self.average:
            clip = self.__loading(path, frame_indices)
        else:
            video_frame_indices=[]
            for clip_frame_indice in frame_indices:
                video_frame_indices+=clip_frame_indice
            clip=self.__loading(path, video_frame_indices)

            # print(len(clip))
            # print(clip[0].shape)
            # # clip=torch.cat(clip,dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # if clip.shape[1]!=64:
        #     print(path,frame_indices)
        return clip, target

    def __len__(self):
        return len(self.data)


class ShotVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
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
                 target_type='label',
                 average=False):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

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

            frame_indices = list(range(segment[0], max(2,segment[1]-2)))

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

        clip=self.spatial_transform(clip)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        # try:
        #     if self.temporal_transform is not None:
        #         frame_indices = self.temporal_transform(frame_indices)
        # except:
        #     print(path)
        #     print(frame_indices)
        #     pass

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if not self.average:
            clip = self.__loading(path, frame_indices)
        else:
            video_frame_indices=[]
            for clip_frame_indice in frame_indices:
                video_frame_indices+=clip_frame_indice
            clip=self.__loading(path, video_frame_indices)

            # print(len(clip))
            # print(clip[0].shape)
            # # clip=torch.cat(clip,dim=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # if clip.shape[1]!=64:
        #     print(path,frame_indices)
        return clip, target,index

    def __len__(self):
        return len(self.data)

class TwoStreamVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 augment_root_path,
                 annotation_path,
                 augment_annotation_path,
                 subset,
                 corruption_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: 'image_{:05d}.jpg'.format(x),
                 target_type='label',
                 average=False,
                 shift=0):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        self.augment_data, self.augment_class_names = self.__make_dataset(
            augment_root_path, augment_annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.shift = shift

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

            # if i<68800:
            #     continue

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

            frame_indices = list(range(segment[0], max(2,segment[1]-2)))

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

        clip=self.spatial_transform(clip)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        augment_path = self.augment_data[index]['video']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
        except:
            print(path)
            print(frame_indices)
            pass

        augment_frame_indices = self.augment_data[index]['frame_indices']

        shift_augment_frame_indices =[]
        for augment_ind in augment_frame_indices:
            shift_augment_frame_indices.append(augment_ind+self.shift)

        try:
            if self.temporal_transform is not None:
                augment_frame_indices = self.temporal_transform(shift_augment_frame_indices)
        except:
            print(augment_path)
            print(augment_frame_indices)
            pass
        # if self.temporal_transform is not None:
        #     frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)
        augment_clip = self.__loading(augment_path, augment_frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, augment_clip, target

    def __len__(self):
        return len(self.data)

class ThreeStreamVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 augment_root_path,
                 augmix_root_path,
                 annotation_path,
                 augment_annotation_path,
                 subset,
                 corruption_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: 'image_{:05d}.jpg'.format(x),
                 target_type='label',
                 average=False):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        self.augment_data, self.augment_class_names = self.__make_dataset(
            augment_root_path, augment_annotation_path, subset, video_path_formatter)
        self.augmix_data, self.augmix_class_names = self.__make_dataset(
            augmix_root_path, augment_annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

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

            # if i<68800:
            #     continue

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

            frame_indices = list(range(segment[0], max(2,segment[1]-2)))

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

        clip=self.spatial_transform(clip)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        augment_path = self.augment_data[index]['video']
        augmix_path =  self.augmix_data[index]['video']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
        except:
            print(path)
            print(frame_indices)
            pass

        augment_frame_indices = self.augment_data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                augment_frame_indices = self.temporal_transform(augment_frame_indices)
        except:
            print(augment_path)
            print(augment_frame_indices)
            pass

        augmix_frame_indices = self.augmix_data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                augmix_frame_indices = self.temporal_transform(augmix_frame_indices)
        except:
            print(augmix_path)
            print(augmix_frame_indices)
            pass
        # if self.temporal_transform is not None:
        #     frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)
        augment_clip = self.__loading(augment_path, augment_frame_indices)
        augmix_clip = self.__loading(augmix_path, augmix_frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, augment_clip, augmix_clip, target

    def __len__(self):
        return len(self.data)

class MotionVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
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
                 target_type='label',
                 average=False,
                 shift=0):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.shift = shift

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

            # if i<68800:
            #     continue

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

            # frame_indices = list(range(segment[0], max(2,segment[1]-2)))
            frame_indices = list(range(segment[0], max(2, segment[1])))
            motion_index_16 = annotations[i]['motion_index_16']
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id,
                'motion_index_16':sorted(motion_index_16)
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

        clip=self.spatial_transform(clip)

        return clip

    def key_frame_sampling(self, key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) + 1 for j in range(frame_size)]
        return index

    def __getitem__(self, index):
        path = self.data[index]['video']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
        except:
            print(path)
            print(frame_indices)
            pass

        motion_frame_indices = self.data[index]['motion_index_16']

        #Hardcode 16 frame length
        if len(motion_frame_indices)< 16:
            motion_frame_indices = self.key_frame_sampling(len(motion_frame_indices),16)

        clip = self.__loading(path, frame_indices)
        motion_clip = self.__loading(path, motion_frame_indices)
        # motion_clip = clip
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, motion_clip, target

    def __len__(self):
        return len(self.data)


class DUVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 corruption_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 temporal_transform2=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: 'image_{:05d}.jpg'.format(x),
                 target_type='label',
                 average=False,
                 shift=0):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.temporal_transform2 = temporal_transform2
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.shift = shift

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

            # if i<68800:
            #     continue

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

            # frame_indices = list(range(segment[0], max(2,segment[1]-2)))
            frame_indices = list(range(segment[0], max(2, segment[1])))
            # motion_index_16 = annotations[i]['motion_index_16']
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id,
                # 'motion_index_16':sorted(motion_index_16)
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.corruption_transform is not None:
            clip = [self.corruption_transform(img) for img in clip]

        clip=self.spatial_transform(clip)

        return clip

    def key_frame_sampling(self, key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) + 1 for j in range(frame_size)]
        return index

    def __getitem__(self, index):
        path = self.data[index]['video']

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                uniform_frame_indices = self.temporal_transform(frame_indices)
        except:
            print(path)
            print(uniform_frame_indices)
            pass

        # motion_frame_indices = self.data[index]['motion_index_16']
        dense_frame_indices = self.temporal_transform2(frame_indices)
        #Hardcode 16 frame length
        if len(dense_frame_indices)< 16:
            motion_frame_indices = self.key_frame_sampling(len(dense_frame_indices),16)

        clip = self.__loading(path, uniform_frame_indices)
        dense_clip = self.__loading(path, dense_frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, dense_clip, target

    def __len__(self):
        return len(self.data)


class DUMotionVideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 motion_path,
                 motion_annotation_path,
                 subset,
                 corruption_transform=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 temporal_transform2=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: 'image_{:05d}.jpg'.format(x),
                 target_type='label',
                 average=False,
                 shift=0):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)
        self.motion_data, self.motion_class_names = self.__make_dataset(
            motion_path, motion_annotation_path, subset, video_path_formatter)
        print("Dataset Size:",len(self.data))
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.temporal_transform2 = temporal_transform2
        self.target_transform = target_transform
        self.corruption_transform=corruption_transform
        self.average=average

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
        self.shift = shift

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

            # if i<68800:
            #     continue

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

            # frame_indices = list(range(segment[0], max(2,segment[1]-2)))
            frame_indices = list(range(segment[0], max(2, segment[1]-2)))
            # motion_index_16 = annotations[i]['motion_index_16']
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id,
                # 'motion_index_16':sorted(motion_index_16)
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if np.array(clip[0]).ndim==2:
            print("Converting Type")
            clip = [img.convert('RGB') for img in clip]

        clip=self.spatial_transform(clip)

        return clip

    def key_frame_sampling(self, key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        index = [int(j / factor) + 1 for j in range(frame_size)]
        return index

    def __getitem__(self, index):
        path = self.data[index]['video']
        motion_path = self.motion_data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        try:
            if self.temporal_transform is not None:
                uniform_frame_indices = self.temporal_transform(frame_indices)
        except:
            print(path)
            print(frame_indices)
            # pass

        # print(uniform_frame_indices)
        # motion_frame_indices = self.data[index]['motion_index_16']
        motion_frame_indices = self.temporal_transform2(frame_indices)
        # print("Dense:",motion_frame_indices)
        #Hardcode 16 frame length
        if len(motion_frame_indices)< 16:
            motion_frame_indices = self.key_frame_sampling(len(motion_frame_indices),16)

        clip = self.__loading(path, uniform_frame_indices)
        dense_clip = self.__loading(path, motion_frame_indices)
        motion_clip = self.__loading(motion_path,motion_frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, dense_clip,motion_clip, target

    def __len__(self):
        return len(self.data)