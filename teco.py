from copy import deepcopy

import torch
import torch.nn as nn


class Norm(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_3d_model(model, eps, momentum, reset_stats,
                                     no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


class LayerNorm(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_transformer(model, eps, momentum, reset_stats,
                                        no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

class SELF_LEARING(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        para = adapt_batchnorm(self.model)
        # para2 = adapt_conv(self.model)
        self.model_state=deepcopy(self.model.state_dict())
        # self.optimizer_state=deepcopy(self.optimizer.state_dict())

    def forward(self,x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state,strict=True)

class SELF_LEARING_TC_TRANS(nn.Module):
    def __init__(self,model,freeze_layer):
        super().__init__()
        self.model = model
        para = adapt_teco(self.model,freeze_layer)
        self.model_state=deepcopy(self.model.state_dict())

    def forward(self,x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state,strict=True)


def adapt_teco(model,freeze_layer):
    model.train()

    start_freeze = freeze_layer
    for i in range(start_freeze,16):
        for param in model.module.blocks[i].parameters():
            print("Freeze Block {}".format(i))
            param.requires_grad = False

    parameters = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            print("LayerNorm 2D/3D Module")
            parameters.extend(module.parameters())
            module.requires_grad_(True)
            module.train()

    return parameters


def adapt_batchnorm(model):
    model.train()

    for param in model.module.layer4.parameters():
        param.requires_grad = False
    for param in model.module.fc.parameters():
        param.requires_grad = False

    parameters = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d) or isinstance(module, torch.nn.BatchNorm2d):
            print("Batch 2D/3D Module")
            parameters.extend(module.parameters())
            module.requires_grad_(True)
            module.train()

    return parameters

def collect_stats(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names

def collect_stats_3d(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            state = m.state_dict()
            # if m.affine:
            #     del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names

def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def configure_3d_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            print("Batch Norm Module")
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def configure_transformer(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            print("Layer Norm Module")
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)