# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

""" Batch norm variants
"""
import pdb

import torch
from torch import nn
from torch.nn import functional as F


def adapt_ema(model: nn.Module):
    return EMABatchNorm.adapt_model(model)


def adapt_parts(model: nn.Module, adapt_mean: bool, adapt_var: bool):
    return PartlyAdaptiveBN.adapt_model(model, adapt_mean, adapt_var)


def adapt_bayesian(model: nn.Module, prior: float):
    return BayesianBatchNorm.adapt_model(model, prior=prior)

def adapt_bayesian_3d(model: nn.Module, prior: float):
    return BayesianBatchNorm3D.adapt_model(model, prior=prior)

def adapt_bayesian_3dspatial(model: nn.Module, prior: float):
    return BayesianBatchNorm3DSpatial.adapt_model(model, prior=prior)

def adapt_bayesian_3dspatialtemporal(model: nn.Module, prior: float):
    return BayesianBatchNorm3DSpatialTemporal.adapt_model(model, prior=prior)


class PartlyAdaptiveBN(nn.Module):
    @staticmethod
    def find_bns(parent, estimate_mean, estimate_var):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = PartlyAdaptiveBN(child, estimate_mean, estimate_var)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(
                    PartlyAdaptiveBN.find_bns(child, estimate_mean,
                                              estimate_var)
                )

        return replace_mods

    @staticmethod
    def adapt_model(model, adapt_mean, adapt_var):
        replace_mods = PartlyAdaptiveBN.find_bns(model, adapt_mean, adapt_var)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, estimate_mean=True, estimate_var=True):
        super().__init__()
        self.layer = layer

        self.estimate_mean = estimate_mean
        self.estimate_var = estimate_var

        self.register_buffer("source_mean", layer.running_mean.data)
        self.register_buffer("source_var", layer.running_var.data)

        self.register_buffer(
            "estimated_mean",
            torch.zeros(layer.running_mean.size(),
                        device=layer.running_mean.device),
        )
        self.register_buffer(
            "estimated_var",
            torch.ones(layer.running_var.size(),
                       device=layer.running_mean.device),
        )

    def reset(self):
        self.estimated_mean.zero_()
        self.estimated_var.fill_(1)

    @property
    def running_mean(self):
        if self.estimate_mean:
            return self.estimated_mean
        return self.source_mean

    @property
    def running_var(self):
        if self.estimate_var:
            return self.estimated_var
        return self.source_var

    def forward(self, input):
        # Estimate training set statistics
        self.reset()
        F.batch_norm(
            input,
            self.estimated_mean,
            self.estimated_var,
            None,
            None,
            True,
            1.0,
            self.layer.eps,
        )

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0.0,
            self.layer.eps,
        )


class EMABatchNorm(nn.Module):
    @staticmethod
    def reset_stats(module):
        module.reset_running_stats()
        module.momentum = None
        return module

    @staticmethod
    def find_bns(parent):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm3d):
                module = EMABatchNorm.reset_stats(child)
                module = EMABatchNorm(module)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(EMABatchNorm.find_bns(child))

        return replace_mods

    @staticmethod
    def adapt_model(model):
        replace_mods = EMABatchNorm.find_bns(model)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # store statistics, but discard result
        self.layer.train()
        self.layer(x)
        # store statistics, use the stored stats
        self.layer.eval()
        return self.layer(x)


class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )
        self.norm=self.norm.cuda()
        self.prior = prior


    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )

class BayesianBatchNorm3D(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
        # for name, child in parent.named_modules():
            child.requires_grad_(False)
            print(name,child)
            if isinstance(child, nn.BatchNorm3d):
                module = BayesianBatchNorm3D(child, prior)
                replace_mods.append((parent, name, module))
            elif isinstance(child, nn.Sequential):
                replace_mods.extend(BayesianBatchNorm.find_bns(child.modules(), prior))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm3D.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")

        for (parent, name, child) in replace_mods:
            # print(name)
            # print(parent,name,child)
            if name[:5]=='layer':
                blk = name.split('.')[1]
                name = name.replace(".{}".format(blk),"[{}]".format(blk))
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        print("*** prior *** ",prior)
        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm3d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )
        self.norm=self.norm.cuda()
        self.prior = prior

    def forward(self, input):
        self.norm(input)

        print(self.layer.running_mean)
        print(self.norm.running_mean)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )

def get_all_parent_layers(net, type):
    layers = []

    for name, l in net.named_modules():
        if isinstance(l, type):
            tokens = name.strip().split('.')

            layer = net
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1],l])

    return layers

def set_all_bn_to_bayesian(net,prior):
    for parent_layer, last_token, child_layer in get_all_parent_layers(net, nn.BatchNorm3d):
        new_module = BayesianBatchNorm3DRe(child_layer,prior)
        setattr(parent_layer, last_token, new_module)
    return net

class BayesianBatchNorm3DRe(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        print("*** prior *** van",prior)
        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm3d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )
        self.norm=self.norm.cuda()
        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0.1,
            self.layer.eps,
        )

class BayesianBatchNorm3DSpatial(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_modules():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm3d):
                module = BayesianBatchNorm3DSpatial(child, prior)
                replace_mods.append((parent, name, module))
            # else:
            #     replace_mods.extend(BayesianBatchNorm3DSpatial.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):

        replace_mods = BayesianBatchNorm3DSpatial.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")

        print("For spatial only")

        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )
        self.norm=self.norm.cuda()
        self.prior = prior

    def forward(self, input):
        tmp_input = input[:,:,8]

        self.norm(tmp_input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )

class BayesianBatchNorm3DSpatialTemporal(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_modules():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm3d):
                module = BayesianBatchNorm3DSpatial(child, prior)
                replace_mods.append((parent, name, module))
            # else:
            #     replace_mods.extend(BayesianBatchNorm3DSpatial.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):

        replace_mods = BayesianBatchNorm3DSpatial.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")

        print("For spatial only")

        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )

        self.spnorm=nn.BatchNorm3d(
            self.layer.num_features,
            affine=False,
            momentum=1.0,
        )

        self.norm = self.norm.cuda()
        self.spnorm = self.spnorm.cuda()

        self.prior = prior

    def forward(self, input):
        tmp_input = input[:, :, 8]

        tmp_spinput = input

        self.norm(tmp_input)
        self.spnorm(tmp_spinput)

        running_mean = (
                self.prior * self.spnorm.running_mean
                + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
                self.prior * self.spnorm.running_var
                + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )