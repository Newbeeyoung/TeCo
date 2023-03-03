import pdb
import shutil
import os
import time
import multiprocessing
from torch.autograd import Variable
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)
from torch.cuda.amp import GradScaler,autocast
from torchvision.utils import save_image
import random
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k=correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))

def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list

def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        elif version == 'v3':
            augments += [
                GroupRandomScale([224,256]),
                GroupRandomCrop(image_size),
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        else:
            augments += [
                GroupRandomScale([224,256]),
                GroupRandomCrop(image_size),
            ]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor

def get_augmentor_withGaussianNoise(is_train, image_size, mean=None,
                  std=None,g_std=0.2, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    def Gaussian_Noise(x,std):
        return x+torch.zeros_like(x).data.normal_(0,std)

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        elif version == 'v3':
            augments += [
                GroupRandomScale([224,256]),
                GroupRandomCrop(image_size),
            ]
            augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
        else:
            augments += [
                GroupRandomScale([224,256]),
                GroupRandomCrop(image_size),
            ]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        transforms.Lambda(lambda x:Gaussian_Noise(x,g_std)),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor

def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    #debugging
    # shuffle=False

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def temporal_consistency_loss(x,distance):
    return F.l1_loss(x[:,distance:],x[:,:-distance])

def train_temporal_consistency_loss(data_loader, model, criterion, optimizer, epoch, a,b,c,d,e,distance, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0

    with tqdm(total=len(data_loader)) as t_bar:
        for i, (clip, target) in enumerate(data_loader):
            # measure data loading time
            # pdb.set_trace()
            data_time.update(time.time() - end)
            # compute output

            # if i<20:
            #     print(clean_clip.size())
            #     save_image(clean_clip[0,:,0,:,:],"visualization/clean_sample_{}.jpg".format(i))
            #     save_image(augment_clip[0, :, 0, :, :], "visualization/augment_sample_{}.jpg".format(i))

            # if gpu_id is not None:
            #     clip = clip.cuda(gpu_id, non_blocking=True)
            clip=clip.cuda()
            bs = clip.size()[0]
            output, layer1_input,layer2_input,layer3_input,layer4_input,fc_input = model(clip)

            target = target.cuda(gpu_id, non_blocking=True)

            loss = softmax_entropy(output).mean(0)

            if a:
                content_loss1=temporal_consistency_loss(layer1_input,distance)
                loss+= a* content_loss1
            if b:
                content_loss2 =temporal_consistency_loss(layer2_input,distance)
                loss+=b*content_loss2
            if c:
                content_loss3 = temporal_consistency_loss(layer3_input,distance)
                loss+=c*content_loss3
            if d:
                content_loss4 = temporal_consistency_loss(layer4_input,distance)
                loss+=d*content_loss4
            # if e:
            #     content_loss5 = temporal_consistency_loss(layer1_input,distance)
            #     loss+=e*content_loss5

            target = target.cuda(gpu_id, non_blocking=True)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)
            # pdb.set_trace()
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), clip.size(0))
            top1.update(prec1[0], clip.size(0))
            top5.update(prec5[0], clip.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch

def partial_temporal_consistency_loss(x,distance,r_mask):
    bs,c,_,_,_=x.size()
    _,nsplit=r_mask.size()
    csplit=c//nsplit

    loss=0

    for split in range(nsplit):
        l1_loss=torch.mean(torch.abs(x[:, csplit * split + distance:csplit * (split + 1)]- x[:, csplit * split:csplit * (split + 1) - distance]).mean(dim=(1,2,3,4),keepdim=True).squeeze()*r_mask[:,split])
        loss+=l1_loss

    return loss

def train_partial_temporal_consistency_loss(data_loader, model, criterion, optimizer, epoch, a,b,c,d,e,distance, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0


    with tqdm(total=len(data_loader)) as t_bar:
        for i, (clip, target) in enumerate(data_loader):
            # measure data loading time
            # pdb.set_trace()
            data_time.update(time.time() - end)
            # compute output

            # if i<20:
            #     print(clean_clip.size())
            #     save_image(clean_clip[0,:,0,:,:],"visualization/clean_sample_{}.jpg".format(i))
            #     save_image(augment_clip[0, :, 0, :, :], "visualization/augment_sample_{}.jpg".format(i))

            # if gpu_id is not None:
            #     clip = clip.cuda(gpu_id, non_blocking=True)
            clip=clip.cuda()
            # concat_clips = torch.cat((clean_clip.cuda(), augment_clip.cuda()), 0)

            bs,c,f,w,h=clip.size()
            output, layer1_input,layer2_input,layer3_input,layer4_input,fc_input = model(clip)

            target = target.cuda(gpu_id, non_blocking=True)

            loss = softmax_entropy(output).mean(0)

            nsplit=2

            overall_input_gradient = torch.abs(clip[:,:,distance:]-clip[:,:,:-distance]).mean(dim=(1,2,3,4),keepdim=True)
            r_mask = torch.zeros(bs,nsplit).cuda()
            for split in range(nsplit):
                fsplit=f//nsplit
                input_gradient_by_split=torch.abs(clip[:,:,split*fsplit+distance:fsplit*(split+1)]-clip[:,:,split*fsplit:fsplit*(split+1)-distance]).mean(dim=(1,2,3,4),keepdim=True)

                r_mask[:,split] = torch.floor((input_gradient_by_split-overall_input_gradient).squeeze()+1).detach()

            if a:
                content_loss1=partial_temporal_consistency_loss(layer1_input,distance,r_mask)
                loss+= a* content_loss1
            if b:
                content_loss2 =partial_temporal_consistency_loss(layer2_input,distance,r_mask)
                loss+=b*content_loss2
            if c:
                content_loss3 = partial_temporal_consistency_loss(layer3_input,distance,r_mask)
                loss+=c*content_loss3
            if d:
                content_loss4 = partial_temporal_consistency_loss(layer4_input,distance,r_mask)
                loss+=d*content_loss4
            # if e:
            #     content_loss5 = temporal_consistency_loss(layer1_input,distance)
            #     loss+=e*content_loss5

            target = target.cuda(gpu_id, non_blocking=True)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)
            # pdb.set_trace()
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), clip.size(0))
            top1.update(prec1[0],clip.size(0))
            top5.update(prec5[0],clip.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch

def validate_content_loss(data_loader, model, criterion, gpu_id=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output, layer1_input,layer2_input,layer3_input,layer4_input,fc_input  = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    return top1.avg, top5.avg, losses.avg, batch_time.avg

def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            # pdb.set_trace()
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images)
            target = target.cuda(gpu_id, non_blocking=True)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    return top1.avg, top5.avg, losses.avg, batch_time.avg
