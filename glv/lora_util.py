import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save
    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0,num_loras=50, lora_stride=4,
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.num_loras = num_loras
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.ModuleList([nn.Linear(in_features, r, bias=False) for i in range(num_loras)])
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.ModuleList([nn.Linear(r, out_features, bias=False) for i in range(num_loras)])
        self.scale = scale
        self.selector = nn.Identity()
        self.lora_stride = lora_stride

        for i in range(num_loras):
            nn.init.normal_(self.lora_down[i].weight, std=1 / r)
            nn.init.zeros_(self.lora_up[i].weight)

    def forward(self, input,lora_id):
        if lora_id is None or sum(lora_id>=0) == 0:
            return self.linear(input)
        lora_id = lora_id//self.lora_stride
        
        if len(lora_id) == 1:
            assert lora_id < self.num_loras
            
            if lora_id < 0:
                return self.linear(input)
            return (
                self.linear(input)
                + self.dropout(self.lora_up[lora_id](self.selector(self.lora_down[lora_id](input))))
                * self.scale
            )
        else:
            outputs = [ ]
            input_lst = input.chunk(len(lora_id))
            for data_idx, lora_idx in enumerate(lora_id):
                input_tmp = input_lst[data_idx]
                if lora_idx < 0:
                    outputs.append(self.linear(input_tmp))
                else:
                    outputs.append(self.linear(input_tmp)
                    + self.dropout(self.lora_up[lora_idx](self.selector(self.lora_down[lora_idx](input_tmp))))
                    * self.scale)
            outputs = torch.cat(outputs,dim=0)
            return outputs

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
        num_loras = 50,
        lora_stride = 4,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.num_loras = num_loras
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.ModuleList([nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ) for i in range(num_loras)])
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.ModuleList([nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ) for i in range(num_loras)])
        self.selector = nn.Identity()
        self.scale = scale
        self.lora_stride = lora_stride

        for i in range(num_loras):
            nn.init.normal_(self.lora_down[i].weight, std=1 / r)
            nn.init.zeros_(self.lora_up[i].weight)

    def forward(self, input, lora_id):
        if lora_id is None:
            return self.conv(input)
        lora_id = lora_id//self.lora_stride
        
        if len(lora_id) == 1:
            assert lora_id < self.num_loras
            # batch 实现
            if lora_id < 0:
                return self.conv(input)
            return (
                self.conv(input)
                + self.dropout(self.lora_up[lora_id](self.selector(self.lora_down[lora_id](input))))
                * self.scale
            )
        else:
            outputs = [ ]
            input_lst = input.chunk(len(lora_id))
            for data_idx,lora_idx in enumerate(lora_id):
                input_tmp = input_lst[data_idx]
                if lora_idx < 0:
                    outputs.append(self.conv(input_tmp))
                else:
                    outputs.append(self.conv(input_tmp)
                    + self.dropout(self.lora_up[lora_idx](self.selector(self.lora_down[lora_idx](input_tmp))))
                    * self.scale)
            outputs = torch.cat(outputs,dim=0)
            return outputs


    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))
    return ret


_find_modules = _find_modules_v2


def get_lora(
    module: nn.Module,
    r: int = 4,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    stride: int = 4,
    num_loras: int = 50,
):
    
    weight = module.weight
    bias = module.bias
    _tmp = LoraInjectedLinear(
        module.in_features,
        module.out_features,
        module.bias is not None,
        r=r,
        dropout_p=dropout_p,
        scale=scale,
        lora_stride = stride,
        num_loras=num_loras
    )
    _tmp.linear.weight = weight
    if bias is not None:
        _tmp.linear.bias = bias
    # switch the module
    _tmp.to(module.weight.device).to(module.weight.dtype)
    return _tmp

def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
    stride: int = 4,
    num_loras: int = 50,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            dropout_p=dropout_p,
            scale=scale,
            lora_stride = stride,
            num_loras=num_loras
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)
        for i in range(num_loras):
            _module._modules[name].lora_up[i].weight.requires_grad = True
            _module._modules[name].lora_down[i].weight.requires_grad = True
        names.append(name)

    return require_grad_params, names