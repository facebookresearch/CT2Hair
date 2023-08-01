# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv

import torch
import torch.nn as nn

from modules.networks import *
from utils.utils import batched_index_select
from utils.strandsutils import natural_cubic_spline_coeffs, NaturalCubicSpline

class StrandEncoder1dCNN(nn.Module):
    def __init__(self, do_vae, num_pts=100, out_channels=64):
        super(StrandEncoder1dCNN, self).__init__()

        self.do_vae = do_vae
        self.num_pts = num_pts

        self.training = False

        out_channels *= 2 # not that we do vae the features are dobules so that we can get mean and variance

        in_channels = 0
        in_channels += 3 # 3 for the xyz
        in_channels += 3 # 3 for the direction
        if num_pts == 100:
            self.cnn_encoder = torch.nn.Sequential(
                Conv1dWN(in_channels, 32, kernel_size=4, stride=2, padding=1, padding_mode='zeros'), torch.nn.SiLU(),
                Conv1dWN(32, 32, kernel_size=4, stride=2, padding=1, padding_mode='zeros'), torch.nn.SiLU(),
                Conv1dWN(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='zeros'), torch.nn.SiLU(),
                Conv1dWN(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'), torch.nn.SiLU(),
                Conv1dWN(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'), torch.nn.SiLU())

            # after runnign cnn we still end up with some elments per strand, and we want to pool over them with something better than an avg pool
            self.final_cnn_aggregator=torch.nn.Sequential(
                LinearWN(128*3, 128), torch.nn.SiLU(),
                LinearWN(128, out_channels))

        elif num_pts == 256:
            self.cnn_encoder = torch.nn.Sequential(
                Conv1dWN(in_channels, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
                Conv1dWN(32, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
                Conv1dWN(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
                Conv1dWN(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
                Conv1dWN(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
                Conv1dWN(256, 256, kernel_size=3, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU())

            self.final_cnn_aggregator = torch.nn.Sequential(
                LinearWN(256 * int(self.num_pts / 64), 256), torch.nn.SiLU(),
                LinearWN(256, out_channels))

        else:
            print("Number of points %d is not supported."%(num_pts))
            exit(0)

        self.pred_mean = torch.nn.Sequential(
            torch.nn.SiLU(),
            LinearWN(out_channels, out_channels), torch.nn.SiLU(),
            LinearWN(out_channels, int(out_channels / 2))
        )
        self.pred_logstd = torch.nn.Sequential(
            torch.nn.SiLU(),
            LinearWN(out_channels, out_channels), torch.nn.SiLU(),
            LinearWN(out_channels, int(out_channels / 2))
        )

        self.apply(lambda x: swish_init(x, False))
        swish_init(self.pred_mean, True)
        swish_init(self.pred_logstd, True)
        self.pe = LearnedPE(in_channels=1, num_encoding_functions=5, logsampling=True)

    def forward(self, points):
        points = points.view(-1, self.num_pts, 3) # nr_strands, points_per_strand, xyz
        original_points = points
        points = points.permute(0, 2, 1) ## nr_strands, xyz, 100
        nr_strands = points.shape[0]

        ### get also the direciton from point to the next
        cur_points = original_points[:, 0:self.num_pts - 1, : ]
        next_points = original_points[:, 1:self.num_pts, :]
        direction = next_points - cur_points
        # pad_zero=torch.zeros(nr_strands,1,3).cuda()
        # direction = torch.cat([direction,pad_zero],1) # make the direction nr_strands, 100, 3
        last_dir = direction[:, self.num_pts - 2:self.num_pts - 1, :]
        direction = torch.cat([direction, last_dir],1) # make the direction nr_strands, 100, 3
        direction = direction.permute(0, 2, 1) # nr_strands, xyz, 100
        # direction=direction * 100 # (we multiply by the nr of segments so that the value is not so small and is closer to our desired range)

        per_point_features = torch.cat([points, direction] ,1)
        strand_features = self.cnn_encoder(per_point_features) # nr_strands, 128(nr_features), 3(elements per string)

        strand_features = strand_features.view(nr_strands, -1).contiguous()
        strand_features = self.final_cnn_aggregator(strand_features) # outputs nr_strands x 128
        
        s = self.pred_mean(strand_features)

        s_mean_and_logstd_dict = {}
        if self.do_vae:
            s_mean = s
            # print("s_mean has mean std ", s_mean.mean(), s_mean.std())
            s_logstd = 0.1 * self.pred_logstd(strand_features)
            s_mean_and_logstd_dict["mean"] = s_mean
            s_mean_and_logstd_dict["logstd"] = s_logstd
            # print("s_logstd has mean std ", s_logstd.mean(), s_logstd.std())
            if self.training:
                std = torch.exp(s_logstd)
                eps = torch.empty_like(std).normal_()
                s = s + std * eps
                # print("strand std min max", std.min(), " ", std.max())
        
        return s, s_mean_and_logstd_dict

class StrandGeneratorSiren(nn.Module):
    # a siren network which predicts various direction vectors along the strand similar ot FakeODE.
    # the idea is that siren works well when periodic thing needs to be predicted and the strand can be seen as some periodic direction vectors being repeted at some points on the strand
    # the idea is similar to modulation siren https://arxiv.org/pdf/2104.03960.pdf
    def __init__(self, in_channels, modulation_hidden_dim, siren_hidden_dim, scale_init, decode_direct_xyz, decode_random_verts, num_pts=100):
        super(StrandGeneratorSiren, self).__init__()

        self.num_pts = num_pts

        self.decode_direct_xyz = decode_direct_xyz
        self.decode_random_verts = decode_random_verts

        self.swish = torch.nn.SiLU()
        self.tanh = torch.nn.Tanh()

        self.nr_layers = 3
        cur_nr_channels = in_channels
        # cur_nr_channels+=1 #+1 for the time t
        self.modulation_layers = torch.nn.ModuleList([])
        for i in range(self.nr_layers):
            self.modulation_layers.append(LinearWN(cur_nr_channels, modulation_hidden_dim))
            cur_nr_channels = modulation_hidden_dim+in_channels  # at the end we concatenate the input z

        self.decode_dir = LinearWN(siren_hidden_dim, 3)

        self.apply(lambda x: swish_init(x, False))
        swish_init(self.decode_dir, True)

        self.siren_layers = torch.nn.ModuleList([])
        self.siren_layers.append(BlockSiren(in_channels=1, out_channels=siren_hidden_dim, is_first_layer=True, scale_init=scale_init))
        for i in range(self.nr_layers-1):
            self.siren_layers.append(BlockSiren(in_channels=siren_hidden_dim, out_channels=siren_hidden_dim))

    def forward(self, strand_features):
        nr_verts_to_create = self.num_pts - 1 # we create only 99 because the frist one is just the origin
        if self.decode_random_verts:
            nr_verts_to_create = 1

        nr_strands = strand_features.shape[0]
        strand_features = strand_features.view(nr_strands, 1, -1).repeat(1, nr_verts_to_create, 1) # nr_strands x 100 x nr_channels

        # sampling t
        t = torch.linspace(0, 1, self.num_pts).cuda()
        t = t.view(self.num_pts, 1)
        if self.decode_direct_xyz:
            t = t[1:self.num_pts, :] # we don't create the root because it's already given
        else: # we are decoding direction therefore the first direction should be computed but the last direction should be ingored because the tip doesnt need a direction
            t = t[0:self.num_pts - 1, :]

        # repeat strand featues to be nr_strands x nr_vert x nr_channels
        # concat for each vertex the positional encoding
        t = t.view(1, self.num_pts - 1, -1).repeat(nr_strands, 1, 1) #nrstrands, nr_verts, nr_channels
        # strand_features_with_time=torch.cat([strand_features,t],2)

        point_indices = None
        if self.decode_random_verts:
            # choose a random t for each strand
            # we can create only up until the very last vertex, except the tip, we need to be able to sample the next vertex so as to get a direction vector
            probability = torch.ones([nr_strands, self.num_pts - 2], dtype=torch.float32, device=torch.device("cuda")) 
            point_indices = torch.multinomial(probability, nr_verts_to_create, replacement=False) # size of the chunk size we selected
            # add also the next vertex on the strand so that we can compute directions
            point_indices = torch.cat([point_indices, point_indices + 1], 1)

            t = batched_index_select(t, 1, point_indices)

        # decode xyz
        h_siren = t
        # z_scaling=0.001 #this has to be initialized so that the h_modulation is something like 0.2.If its lower, 
        # then no gradient will flow into Z and then the network will not be optimized. You might need to do one run and check the gradients of the network with model.summary to see if the gradients don't vanish
        z_scaling = 1.0
        z = strand_features
        z_initial = z * z_scaling
        z = z * z_scaling
        with_checkpointing = True
        for i in range(self.nr_layers):
            h_modulation = self.swish( self.modulation_layers[i](z))
            s = self.siren_layers[i](h_siren)
            h_siren = (1 - h_modulation) * s
            # for next iter
            z = torch.cat([z_initial, h_modulation], 2)
        if self.decode_direct_xyz:
            points_dir = self.decode_dir(h_siren) * 0.1
            if self.decode_random_verts:
                pred_strands = points_dir
            else:
                start_positions = torch.zeros(nr_strands, 1, 3).cuda()
                pred_strands = torch.cat([start_positions, points_dir], 1)
        else:
            # divide by the nr of points on the strand otherwise the direction will have norm=1 and then when integrated you end up with a gigantic strand that has 100 units
            hair_dir = self.decode_dir(h_siren) * 0.01 
            pred_strands = torch.cumsum(hair_dir, dim=1) # nr_strands, nr_verts-1, 3
            # we know that the first vertex is 0,0,0 so we just concatenate that one
            start_positions = torch.zeros(nr_strands, 1, 3).cuda()
            pred_strands = torch.cat([start_positions, pred_strands], 1)

        return pred_strands, point_indices

'''
uses only one Z tensor and predicts the strands using SIREN. There is no normalization apart from moving the strands to origin
is used to predict and regress only strand data, with no scalp
'''
class StrandCodec(nn.Module):
    def __init__(self, do_vae, decode_direct_xyz, decode_random_verts, train_params, is_train=True):
        super(StrandCodec, self).__init__()

        self.do_vae = do_vae
        self.decode_direct_xyz = decode_direct_xyz
        self.decode_random_verts = decode_random_verts
        self.nr_verts_per_strand = train_params['num_pts']
        if self.decode_random_verts:
            self.nr_verts_per_strand = 2

        self.cosine_embed_loss = nn.CosineEmbeddingLoss()

        if is_train:
            self.weight_pts = train_params['weight_pts']
            self.weight_dir = train_params['weight_dir'] # 0.001
            self.weight_kl= train_params['weight_kl'] # 0.0001

        # encode
        self.strand_encoder_for_shape = StrandEncoder1dCNN(self.do_vae, self.nr_verts_per_strand, train_params['code_channels']) # predicts 64 vector of shape, gets the inputs after they were normalized

        # decoder
        self.strand_generator = StrandGeneratorSiren(in_channels=train_params['code_channels'], modulation_hidden_dim=32, siren_hidden_dim=32,
                                                     scale_init=5, decode_direct_xyz=decode_direct_xyz, decode_random_verts=decode_random_verts,
                                                     num_pts=self.nr_verts_per_strand)  # generate a whoel strand from 64 dimensional shape vector
                                                     
    def save(self, root_folder, experiment_name, iter_nr):
        models_path = os.path.join(root_folder, experiment_name, str(iter_nr), "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "strand_codec.pt"))

        # write csv with some info
        out_info_path = os.path.join(models_path, "strand_codec_info.csv")
        with open(out_info_path, "w") as f: #we need to put the writer in a block so that it closes the file automaticaly afterwards
            w = csv.writer(f)
            w.writerow(["do_vae", self.do_vae])
            w.writerow(["decode_direct_xyz", self.decode_direct_xyz])
            w.writerow(["decode_random_verts", self.decode_random_verts])

    def diff_spline(self, hair_data_dict):
        points = hair_data_dict["points"].cuda()
        times = hair_data_dict["times"].cuda()
        coeffs = natural_cubic_spline_coeffs(times, points)
        spline = NaturalCubicSpline(coeffs)
        time_pts = torch.arange(self.nr_verts_per_strand).cuda() / (self.nr_verts_per_strand - 1)
        time_pts = time_pts.repeat(points.shape[0], 1)
        self.splined_points = spline.evaluate(time_pts)
        self.splined_points = self.splined_points.detach()

    def encode(self):
        s_shape, s_shape_mean_and_logstd_dict = self.strand_encoder_for_shape(self.splined_points)

        encoded_dict = {}
        encoded_dict["s_shape"] = s_shape
        encoded_dict["s_shape_mean_and_logstd_dict"] = s_shape_mean_and_logstd_dict

        return encoded_dict

    def decode(self, encoded_dict):
        s_shape = encoded_dict["s_shape"]

        # generate the strand points
        pred_points, point_indices = self.strand_generator(s_shape)

        prediction_dict = {}
        prediction_dict["pred_points"] = pred_points
        prediction_dict["point_indices"] = point_indices

        return prediction_dict

    def compute_loss(self, prediction_dict, encoded_dict):
        loss_l2 = self.compute_loss_l2(prediction_dict)
        loss_dir = self.compute_loss_dir(prediction_dict)
        loss_kl = self.compute_loss_kl(encoded_dict)

        loss = self.weight_pts * loss_l2 + self.weight_dir * loss_dir + self.weight_kl * loss_kl
        # loss = loss_l2 + loss_dir * 0.01 + loss_kl * 0.001
        # loss = loss_l2 + loss_dir * 0.1 + loss_kl * 0.001 # this gives the lowest kl and the autodecoding looks nice
        
        loss_dict = {}
        loss_dict['loss'] = loss
        loss_dict['loss_l2'] = loss_l2
        loss_dict['loss_dir'] = loss_dir
        loss_dict['loss_kl'] = loss_kl
        return loss_dict

    def compute_loss_l2(self, prediction_dict):
        pred_points = prediction_dict["pred_points"].view(-1, self.nr_verts_per_strand, 3)
        loss_l2 = ((pred_points - self.splined_points) ** 2).mean()

        return loss_l2

    def compute_loss_dir(self, prediction_dict):
        pred_points = prediction_dict["pred_points"].view(-1, self.nr_verts_per_strand, 3)

        # get also a loss for the direciton, we need to compute the direction
        cur_points = pred_points[:, 0:self.nr_verts_per_strand - 1, : ]
        next_points = pred_points[:, 1:self.nr_verts_per_strand, :]
        pred_deltas = next_points - cur_points
        pred_deltas = pred_deltas.view(-1, 3)

        gt_cur_points = self.splined_points[:, 0:self.nr_verts_per_strand - 1, : ]
        gt_next_points = self.splined_points[:, 1:self.nr_verts_per_strand, :]
        gt_dir = gt_next_points - gt_cur_points
        gt_dir = gt_dir.view(-1, 3)
        loss_dir = self.cosine_embed_loss(pred_deltas, gt_dir, torch.ones(gt_dir.shape[0]).cuda())

        return loss_dir

    def compute_loss_kl(self, encoded_dict):
        #get input data
        kl_loss = 0

        if self.do_vae:
            #kl loss
            s_shape_mean_and_logstd_dict = encoded_dict["s_shape_mean_and_logstd_dict"]

            kl_shape = self.kl( s_shape_mean_and_logstd_dict["mean"], s_shape_mean_and_logstd_dict["logstd"])
            # free bits from IAF-VAE. so that if the KL drops below a certan value, then we stop reducing the KL
            kl_shape = torch.clamp(kl_shape, min=0.25)

            kl_loss = kl_shape.mean()

        return kl_loss

    def kl(self, mean, logstd):
        kl = (-0.5 - logstd + 0.5 * mean ** 2 + 0.5 * torch.exp(2 * logstd))
        return kl

    def forward(self, hair_data_dict):
        self.diff_spline(hair_data_dict)
        encoded_dict=self.encode()
        prediction_dict=self.decode(encoded_dict)

        return prediction_dict, encoded_dict