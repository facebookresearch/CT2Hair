# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import inspect
import numpy as np

from typing import Dict, List, Optional, Tuple
from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm

class LearnedPE(torch.nn.Module):
    def __init__(self, in_channels, num_encoding_functions, logsampling):
        super(LearnedPE, self).__init__()
        self.in_channels = in_channels
        self.num_encoding_functions = num_encoding_functions
        self.logsampling = logsampling

        out_channels = in_channels * self.num_encoding_functions * 2
        self.conv = torch.nn.Linear(in_channels, int(out_channels / 2), bias=True).cuda()  #in the case we set the weight ourselves

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            num_input = self.in_channels
            self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input))
            # print("weight is ", self.conv.weight.shape) #60x3

            # we make the same as the positonal encoding, which is mutiplying each coordinate with this linespaced frequencies
            lin = 2.0 ** torch.linspace(0.0,
                                        self.num_encoding_functions - 1,
                                        self.num_encoding_functions,
                                        dtype=torch.float32,
                                        device=torch.device("cuda"))
            lin_size = lin.shape[0]
            weight = torch.zeros([self.in_channels, self.num_encoding_functions * self.in_channels], dtype=torch.float32, device=torch.device("cuda"))
            for i in range(self.in_channels):
                weight[i : i + 1,   i * lin_size : i * lin_size + lin_size ] = lin

            weight = weight.t().contiguous()

            self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        x_proj = self.conv(x)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj), x], -1).contiguous()

class BlockSiren(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activ=torch.sin, is_first_layer=False, scale_init=90):
        super(BlockSiren, self).__init__()
        self.bias = bias
        self.activ = activ
        self.is_first_layer = is_first_layer
        self.scale_init = scale_init

        self.conv = torch.nn.Linear(in_channels, out_channels, bias=self.bias).cuda()

        # if self.activ==torch.sin or self.activ==None:
        with torch.no_grad():
            if self.activ == torch.sin:
                    num_input = in_channels
                    # See supplement Sec. 1.5 for discussion of factor 30
                    if self.is_first_layer:
                        self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input))
                    else:
                        self.conv.weight.uniform_(-np.sqrt(6 / num_input) , np.sqrt(6 / num_input))
            elif self.activ == None:
                # self.conv.weight.normal_(0, 0.1)
                swish_init(self.conv, True)

    def forward(self, x):
        x = self.conv(x)

        if self.activ == torch.sin:
            if self.is_first_layer:
                x = self.scale_init * x
            else:
                x = x * 1
            x = self.activ(x)

        elif self.activ is not None:
            x = self.activ(x)

        return x

def check_args_shadowing(name, method, arg_names):
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")

# For backward compatibility.
class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ):
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.
        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.
            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


def weight_norm_wrapper(cls, name="weight", g_dim=0, v_dim=0):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.
    Args:
        name: str
        Name of the parameter to apply weight normalization.
        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.
        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = torch.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    return Wrap

def is_weight_norm_wrapped(module):
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False

LinearWN = weight_norm_wrapper(torch.nn.Linear, g_dim=0, v_dim=None)
Conv1dWN = weight_norm_wrapper(torch.nn.Conv1d, g_dim=0, v_dim=None)

def swish_init(m, is_linear, scale=1):
    # normally relu has a gain of sqrt(2)
    # however swish has a gain of sqrt(2.952) as per the paper https://arxiv.org/pdf/1805.08266.pdf
    gain=np.sqrt(2.952)
    # gain=np.sqrt(2)

    if is_linear:
        gain = 1
        # gain = np.sqrt(2.0 / (1.0 + 1 ** 2))

    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt(n1 * ksize)

    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        # std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
        std = gain / np.sqrt(n1 * ksize)

    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain / np.sqrt((n1))

    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.normal_(0, std*scale)
    if m.bias is not None:
        m.bias.data.zero_()

    if is_wnw:
        m.unfuse()