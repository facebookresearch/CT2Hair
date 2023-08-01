# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dGaussian(nn.Module):
    '''
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    '''
    def __init__(self,
                 size: int,                
                 sigma=3,
                 gamma_y=1.0,
                 gamma_z=1.0,
                 padding=None,
                 device='cuda'):
        super().__init__()

        self.size = size
        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        self.sigma = sigma

        self.gamma_y = gamma_y
        self.gamma_z = gamma_z

        self.kernels = self.init_kernel()

    def init_kernel(self):
        sigma_x = self.sigma
        sigma_y = self.sigma * self.gamma_y
        sigma_z = self.sigma * self.gamma_z

        c_max, c_min = int(self.size / 2), -int(self.size / 2)
        (x, y, z) = torch.meshgrid(torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1), indexing='ij') # for future warning

        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        kernel = torch.exp(-.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2 + z ** 2 / sigma_z ** 2))

        # normalize
        kernel = F.normalize(kernel)

        return kernel.reshape(1, 1, self.size, self.size, self.size).contiguous()

    def forward(self, x):
        with torch.no_grad():
           x = F.conv3d(x, weight=self.kernels, padding=self.padding)
        return x

class Conv3dLaplacian():
    '''
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    '''
    def __init__(self,
                 padding=None,
                 device='cuda'):
        super().__init__()

        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        self.kernels = self.init_kernel()

    def init_kernel(self):
        kernel = torch.ones((3, 3, 3), device=self.device) * -1
        kernel[1, 1, 1] = 26
        return kernel.reshape(1, 1, 3, 3, 3)

    def forward(self, x):
        with torch.no_grad():
            x = F.conv3d(x, weight=self.kernels, padding=self.padding)
            mask = x[0, 0] > 0
        return mask.float()

class Conv3dErosion(nn.Module):
    '''
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    '''
    def __init__(self,
                 size=3,
                 padding=None,
                 device='cuda'):
        super().__init__()

        self.size = size
        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        self.kernels = self.init_kernel()

    def init_kernel(self):
        kernel = torch.ones((self.size, self.size, self.size), device=self.device)
        return kernel.reshape(1, 1, self.size, self.size, self.size)
    
    def forward(self, x, ration=1):
        with torch.no_grad():
            x = F.conv3d(x, weight=self.kernels, padding=self.padding)
            mask = x[0, 0] >= self.size ** 3 * ration
        return mask.float()

class Conv3dGabor():
    '''
    Applies a 3d convolution over an input signal using Gabor filter banks.
    WARNING: the size of the kernel must be an odd number otherwise it'll be shifted with respect to the origin
    Refer to https://github.com/m-evdokimov/pytorch-gabor3d
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 size: int,                
                 sigma=3,
                 gamma_y=0.5,
                 gamma_z=0.5,
                 lambd=6,
                 psi=0.,
                 padding=None,
                 device='cuda'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = in_channels * out_channels
        self.size = size
        self.device = device

        if padding:
            self.padding = padding
        else:
            self.padding = 0

        # all additional axes are made for correct broadcast
        # the bounds of uniform distribution adjust manually for every size (rn they're adjusted for 5x5x5 filters)
        # for better understanding: https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

        self.sigma = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * sigma

        self.gamma_y = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * gamma_y
        self.gamma_z = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * gamma_z

        self.lambd = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * lambd

        self.psi = torch.ones(size=(self.num_filters, 1, 1, 1)).to(self.device) * psi

        self.angles = torch.zeros(size=(self.num_filters, 3)).to(self.device)
        num_angles_per_axis = round(math.sqrt(self.num_filters))
        angle_step = math.pi / num_angles_per_axis
        # use polar coordinate, theta round with x, phi round with y
        for i_theta in range(num_angles_per_axis):
            for j_phi in range(num_angles_per_axis):
                rot_angle = torch.tensor([0, j_phi * angle_step, i_theta * angle_step]).to(self.device)
                self.angles[i_theta * num_angles_per_axis + j_phi] = rot_angle

        self.kernels = self.init_kernel()

    def init_kernel(self):
        '''
        Initialize a gabor kernel with given parameters
        Returns torch.Tensor with size (out_channels, in_channels, size, size, size)
        '''
        lambd = self.lambd
        psi = self.psi

        sigma_x = self.sigma
        sigma_y = self.sigma * self.gamma_y
        sigma_z = self.sigma * self.gamma_z
        R = self.get_rotation_matrix().reshape(self.num_filters, 3, 3, 1, 1, 1)

        c_max, c_min = int(self.size / 2), -int(self.size / 2)
        (x, y, z) = torch.meshgrid(torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1),
                                   torch.arange(c_min, c_max + 1), indexing='ij') # for future warning

        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        # meshgrid for every filter
        x = x.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        y = y.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)
        z = z.unsqueeze(0).repeat(self.num_filters, 1, 1, 1)

        x_prime = z * R[:, 2, 0] + y * R[:, 2, 1] + x * R[:, 2, 2]
        y_prime = z * R[:, 1, 0] + y * R[:, 1, 1] + x * R[:, 1, 2]
        z_prime = z * R[:, 0, 0] + y * R[:, 0, 1] + x * R[:, 0, 2]

        yz_prime = torch.sqrt(y_prime ** 2 + z_prime ** 2)

        # gabor formula
        kernel = torch.exp(-.5 * (x_prime ** 2 / sigma_x ** 2 + y_prime ** 2 / sigma_y ** 2 + z_prime ** 2 / sigma_z ** 2)) \
                 * torch.cos(2 * math.pi * yz_prime / (lambd + 1e-6) + psi)

        return kernel.reshape(self.out_channels, self.in_channels, self.size, self.size, self.size).contiguous()

    def get_rotation_matrix(self):
        '''
        Makes 3d rotation matrix.
        R_x = torch.Tensor([[cos_a, -sin_a, 0],
                           [sin_a, cos_a,  0],
                           [0,     0,      1]],)
        R_y = torch.Tensor([[cos_b,  0, sin_b],
                           [0    ,  1,    0],
                           [-sin_b, 0, cos_b]])
        R_z = torch.Tensor([[1,  0,     0],
                           [0,  cos_g, -sin_g],
                           [0,  sin_g, cos_g]])
        '''

        sin_a, cos_a = torch.sin(self.angles[:, 0]), torch.cos(self.angles[:, 0])
        sin_b, cos_b = torch.sin(self.angles[:, 1]), torch.cos(self.angles[:, 1])
        sin_g, cos_g = torch.sin(self.angles[:, 2]), torch.cos(self.angles[:, 2])

        R_x = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_x[:, 0, 0] = cos_a
        R_x[:, 0, 1] = -sin_a
        R_x[:, 1, 0] = sin_a
        R_x[:, 1, 1] = cos_a
        R_x[:, 2, 2] = 1

        R_y = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_y[:, 0, 0] = cos_b
        R_y[:, 0, 2] = sin_b
        R_y[:, 2, 0] = -sin_b
        R_y[:, 2, 2] = cos_b
        R_y[:, 1, 1] = 1

        R_z = torch.zeros(size=(self.num_filters, 3, 3)).to(self.device)
        R_z[:, 1, 1] = cos_g
        R_z[:, 1, 2] = -sin_g
        R_z[:, 2, 1] = sin_g
        R_z[:, 2, 2] = cos_g
        R_z[:, 0, 0] = 1

        return R_x @ R_y @ R_z

    def forward(self, x):
        with torch.no_grad():
            x = F.conv3d(x, weight=self.kernels, padding=self.padding)
        return x