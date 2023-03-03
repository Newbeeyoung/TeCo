import copy
import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as T
from torchvision import transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import logging

from models import build_model
from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
# from utils.video_dataset import VideoDataSet
from utils.temporal_transform import *
from utils.new_video_dataset import NewVideoDataset,DUVideoDataset
from utils.dataset_config import get_dataset_config
from utils.loader import VideoLoaderHDF5
# from utils.common_corruption_imagenet import *
from opts import arg_parser
from pdb import set_trace
import teco
import bn

logger = logging.getLogger(__name__)

def adapt_batchnorm(model):
    model.eval()
    parameters = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            print("Batch 3D Module")
            parameters.extend(module.parameters())
            module.train()
    return parameters

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.
    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """

    return optim.SGD(params,
                     lr=0.00025,
                     momentum=0.9,
                     dampening=0,
                     weight_decay=0,
                     nesterov=True)

def reset(model,model_state):

    return model.load_state_dict(copy.deepcopy(model_state),strict=True)

def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])

        data = data.cuda()
        result,_ = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result

def temporal_consistency_loss(x,distance):
    return F.l1_loss(x[:,:,distance:],x[:,:,-distance])

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print("Number of GPUs:", torch.cuda.device_count())

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)

    args.num_classes = num_classes

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    model = model.cuda()
    vanilla_model = torch.nn.DataParallel(model)

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')

        vanilla_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> creating model '{}'".format(arch_name))

    vanilla_model = bn.set_all_bn_to_bayesian(vanilla_model,args.prior)
    model = teco.SELF_LEARING(vanilla_model)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr
    )
    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    # Data loading code
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}".format(args.num_clips))

    temporal_transform = []
    temporal_transform.append(Uniform_Sampling_Fixoffset(n_samples=args.groups))
    temporal_transform = Compose(temporal_transform)

    temporal_transform2 = []
    temporal_transform2.append(TemporalRandomCrop(size=args.groups))
    temporal_transform2 = Compose(temporal_transform2)

    video_path_formatter = (lambda root_path, label, video_id: root_path /
                                                               label / '{}.hdf5'.format(video_id))

    corruption_transform = None

    crop_list = ['motion_blur', 'packet_loss_hdf5', 'packet_drop_hdf5', 'h265_crf_hdf5', 'h264_abr_hdf5',
                 'h264_crf_hdf5', 'h265_abr_hdf5', 'frame_rate']

    key = args.corruption
    severity = args.severity
    print("Corruption:", key, "Severity:", severity)

    val_video_path = Path("../{}}/{}/{}".format(args.data_folder,key, severity))
    val_annotation_path = Path(
        "data/mini_kinetics/mini-kinetics200_{}_{}.json".format(
            key, severity))
    print(val_annotation_path)
    augments = []
    if key in crop_list:
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_size)
        ]

    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = T.Compose(augments)

    val_dataset = DUVideoDataset(val_video_path,
                                  val_annotation_path,
                                  'validation',
                                  corruption_transform=corruption_transform,
                                  spatial_transform=augmentor,
                                  temporal_transform=temporal_transform,
                                  temporal_transform2=temporal_transform2,
                                  target_transform=None,
                                  video_loader=VideoLoaderHDF5(),
                                  video_path_formatter=video_path_formatter)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name, args.name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        top1 = AverageMeter()
        top5 = AverageMeter()

    total_outputs = 0
    outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
    # switch to evaluate mode
    total_batches = len(data_loader)
    with tqdm(total=total_batches) as t_bar:
        for i, (data_u, data_d, label) in enumerate(data_loader):

            batch_size = data_u.shape[0]

            data_u = data_u.cuda()
            data_d = data_d.cuda()

            all_data = torch.cat((data_u,data_d),0)
            output,nlb_feature = model(all_data)
            output_u,output_d=torch.split(output,batch_size)
            ent_loss = teco.softmax_entropy(output_u).mean(0)
            tc_loss = temporal_consistency_loss(torch.split(nlb_feature, batch_size)[1], 1)
            loss = ent_loss+args.beta*tc_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t_bar.update(1)
            if i%20==0:
                print("Entropy Loss:",ent_loss.item(),"TC Loss:",tc_loss.item(),"Loss:",loss.item())

    model.eval()
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video,video2, label) in enumerate(data_loader):
            output = eval_a_batch(video, model, num_clips=args.num_clips, num_crops=args.num_crops,
                                  threed_data=args.threed_data)

            label = label.cuda(non_blocking=True)
            # measure accuracy
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            top1.update(prec1[0], video.size(0))
            top5.update(prec5[0], video.size(0))
            output = output.data.cpu().numpy().copy()
            batch_size = output.shape[0]
            outputs[total_outputs:total_outputs + batch_size, :] = output

            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        print("Predict {} videos.".format(total_outputs), flush=True)

    if args.evaluate:
        print(
            'Val@{}({}) (# corruption ={}, #severity ={}, # crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
                args.input_size, scale_size, args.corruption, severity, args.num_crops, args.num_clips,
                top1.avg, top5.avg),
            flush=True)
        print(
            'Val@{}({}) (# corruption ={}, #severity ={}, # crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
                args.input_size, scale_size, args.corruption, severity, args.num_crops, args.num_clips,
                top1.avg, top5.avg),
            flush=True, file=logfile)

    logfile.close()

if __name__ == '__main__':
    main()
