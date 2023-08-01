# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import numpy as np

from tqdm import tqdm
from sklearn.cluster import MeanShift
from scipy.spatial import KDTree

from utils.meshutils import strands_world2tbn, make_closest_uv_barys
from datautils.dataloaders import TbnStrandsBinDataset
from modules.strands_codec import StrandCodec

class NeuralStrands():
    def __init__(self, is_resampled=True):
        self.is_resampled = is_resampled
        self.texture_height = 1024
        self.texture_width = 1024
        self.feature_channels = 128 # 64 for old, 128 for new
        self.num_strds_points = 256 # 100 for old, 256 for new
        self.neural_texture = np.zeros((self.texture_width, self.texture_height, self.feature_channels))
        self.neural_texture_pca_rgb = np.zeros((self.texture_width, self.texture_height, 3))
        self.strds_idx_map = np.zeros((self.texture_width, self.texture_height, 1), dtype=np.uint32)

        train_param = {"num_pts": self.num_strds_points, "code_channels": self.feature_channels}

        self.ckpt_path = 'CT2Hair/ckpt/neuralstrands_model.pt'
        self.model = StrandCodec(do_vae=True, decode_direct_xyz=False, decode_random_verts=False, train_params=train_param, is_train=False).to("cuda")
        checkpoint = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

    def prep_strands_data(self, strands, head_mesh, scalp_mesh, scalp_faces_idx):
        self.original_strands = strands
        self.head_mesh = head_mesh

        self.tbn_strands, barys, interp_idxs, face_idxs, self.valid_strds_idxs, \
        self.tangents, self.bitangents, self.normals = strands_world2tbn(strands, head_mesh, scalp_mesh, scalp_faces_idx)

        self.head_index_map, self.head_bary_map = make_closest_uv_barys(torch.tensor(head_mesh.visual.uv), torch.tensor(head_mesh.faces),
                                                                        [self.texture_height, self.texture_width])  # type: ignore for avoiding pylance error
        
        # get uv coords for hair strands roots
        head_interp_idxs = head_mesh.faces[scalp_faces_idx][face_idxs]
        head_uv_coords = head_mesh.visual.uv    # num_vertices x 2
        v0 = head_uv_coords[head_interp_idxs[:, 0]]
        v1 = head_uv_coords[head_interp_idxs[:, 1]]
        v2 = head_uv_coords[head_interp_idxs[:, 2]]
        b0, b1, b2 = np.split(barys, 3, axis=1)
        self.strds_uv_coords = b0 * v0 + b1 * v1 + b2 * v2
        # try to save a texture map for demonstration
        self.strds_texel_coords = self.strds_uv_coords * [self.texture_height, self.texture_width]
        self.strds_texel_coords = np.around(self.strds_texel_coords).astype(np.int32)

        tbn_strds_dataset = TbnStrandsBinDataset(self.tbn_strands, is_resampled=self.is_resampled, num_strds_points=self.num_strds_points)
        self.tbn_strds_dataloader = tbn_strds_dataset.get_dataloader()

    def decode(self, strds_code):
        strds_code_dict = {}
        strds_code_dict['s_shape'] = strds_code
        pred_dict = self.model.decode(strds_code_dict)
        pred_points = pred_dict["pred_points"]

        return pred_points

    def get_neural_representations(self, iter_opt=0, lr=1e-4):
        # loss_writer = SummaryWriter('log/neural_rep/')
        self.regular_strands = torch.zeros((0, self.num_strds_points, 3)).cuda()   # valid strands in TBN space with the unified number of points
        self.strds_features = torch.zeros((0, self.feature_channels)).cuda()
        hair_loss_l2 = []
        hair_loss_dir = []
        loop = tqdm(enumerate(self.tbn_strds_dataloader, 0))
        for i_data, input_data in loop:
            self.model.diff_spline(input_data)
            encoded_dict = self.model.encode()
            strds_code = encoded_dict['s_shape'].clone().detach()

            # setup optimization
            strds_code = strds_code.requires_grad_(True)
            strds_code_dict = {}
            strds_code_dict['s_shape'] = strds_code
            code_optimizer = torch.optim.Adam([strds_code_dict['s_shape']], lr=lr)

            if iter_opt == 0:
                prediction_dict = self.model.decode(strds_code_dict)

                loss_l2 = self.model.compute_loss_l2(prediction_dict)
                loss_dir = self.model.compute_loss_dir(prediction_dict)
                loss = loss_l2 + loss_dir * 1e-4
                hair_loss_l2.append(loss_l2.item())
                hair_loss_dir.append(loss_dir.item())
            else:
                for i_iter in range(iter_opt):
                    self.model.train()
                    prediction_dict = self.model.decode(strds_code_dict)

                    loss_l2 = self.model.compute_loss_l2(prediction_dict)
                    loss_dir = self.model.compute_loss_dir(prediction_dict)
                    loss = loss_l2 + loss_dir * 0.001

                    code_optimizer.zero_grad()
                    loss.backward()
                    code_optimizer.step()

                    hair_loss_l2.append(loss_l2.item())
                    hair_loss_dir.append(loss_dir.item())

            loop.set_description("Getting neural representations, batch loss: l2: %f, dir: %f"%(loss_l2.item(), loss_dir.item()))

            self.regular_strands = torch.concat((self.regular_strands, self.model.splined_points), dim=0)
            self.strds_features = torch.concat((self.strds_features, strds_code_dict['s_shape']), dim=0)

        hair_loss_l2 = np.array(hair_loss_l2)
        hair_loss_dir = np.array(hair_loss_dir)
        print('Average reconstruction errors: l2: %f, dir: %f'%(np.mean(hair_loss_l2), np.mean(hair_loss_dir)))
        
        self.regular_strands = self.regular_strands.reshape(-1, self.num_strds_points, 3).detach().cpu().numpy()
        self.strds_features = self.strds_features.reshape(-1, self.feature_channels).detach().cpu().numpy()
        self.neural_texture[np.clip(self.texture_height - self.strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                            np.clip(self.strds_texel_coords[:, 0], 0, self.texture_width - 1), :] = self.strds_features[:, :]

        strds_idxs = np.arange(self.strds_features.shape[0]) + 1
        self.strds_idx_map[np.clip(self.texture_height - self.strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                           np.clip(self.strds_texel_coords[:, 0], 0, self.texture_width - 1), 0] = strds_idxs

        valid_texel_in_map = np.where(self.strds_idx_map > 0)
        self.texel_strds_idxs = self.strds_idx_map[valid_texel_in_map[0], valid_texel_in_map[1], 0] - 1

        self.used_strands = self.original_strands[self.valid_strds_idxs][self.texel_strds_idxs]

    def denoise_on_strds(self, num_init_clusters=16, num_iters=64, max_cls_thres=2., num_strds_thres=64):
        '''
        Denoising on regular TBN strands.
        Return denoised strands index on valid strands (tbn strands).
        '''
        # Try classic K-means on points
        valid_texel_in_map = np.where(self.strds_idx_map > 0)
        texel_strds_idxs = self.strds_idx_map[valid_texel_in_map[0], valid_texel_in_map[1], 0] - 1
        num_texel_strds = texel_strds_idxs.shape[0]
        init_centers_idxs = np.arange(num_init_clusters) * (num_texel_strds // num_init_clusters)

        init_strds_centriods = self.regular_strands[init_centers_idxs]

        num_clusters = num_init_clusters
        strds_centriods = init_strds_centriods

        adaptive_iter = 0
        while(True):
            repeated_strds = self.regular_strands[:, None, :, :].repeat(num_clusters, axis=1)   # type: ignore for avoiding pylance error

            for i_iter in range(num_iters):
                pts_dis_centriods = np.sqrt(np.sum((repeated_strds - strds_centriods) ** 2, axis=-1, keepdims=False))
                strd_dis_centriods = np.sum(pts_dis_centriods, axis=-1, keepdims=False) # naive sum without weights
                strd_clusters = np.argmin(strd_dis_centriods, axis=-1)

                # update means
                pre_strds_centroids = copy.deepcopy(strds_centriods)
                for j_cls in range(num_clusters):
                    cluster_strds = self.regular_strands[np.where(strd_clusters == j_cls)[0]]
                    strds_centriods[j_cls] = np.sum(cluster_strds, axis=0, keepdims=False) / cluster_strds.shape[0]

                # centroid_dis = np.sum(np.sqrt(np.sum((strds_centriods - pre_strds_centroids) ** 2, axis=-1, keepdims=False)))
                # print(centroid_dis)

            # recalculate strands cluster use the final center
            pts_dis_centriods = np.sqrt(np.sum((repeated_strds - strds_centriods) ** 2, axis=-1, keepdims=False))
            strd_dis_centriods = np.sum(pts_dis_centriods, axis=-1, keepdims=False) # naive sum without weights
            strd_clusters = np.argmin(strd_dis_centriods, axis=-1)

            # calculate the max distances in clusters
            strd_clusters_dis = np.min(strd_dis_centriods, axis=-1)

            num_currt_clusters = num_clusters
            for i_cls in range(num_currt_clusters):
                strd_cluster_idx = np.where(strd_clusters == i_cls)[0]
                cluster_dis = strd_clusters_dis[strd_cluster_idx]
                max_cls_dis = np.max(cluster_dis)
                max_strd_idx = np.argmax(cluster_dis)

                if max_cls_dis > max_cls_thres:
                    num_clusters += 1
                    strds_centriods = np.concatenate((strds_centriods, self.regular_strands[strd_cluster_idx][max_strd_idx:max_strd_idx+1]), axis=0)

            if num_clusters == num_currt_clusters:
                break
            
            num_iters = num_iters // 2

            if num_iters < 1:
                break
            
            adaptive_iter += 1
            print('Adaptive K-means iter %d...'%(adaptive_iter))

        denoised_strds_idxs = []    # for valid tbn_strands
        for i_cls in range(num_clusters):
            cluster_idxs = np.where(strd_clusters == i_cls)[0].tolist()
        
            if len(cluster_idxs) >= num_strds_thres:   # type: ignore for avoiding pylance error
                denoised_strds_idxs.extend(cluster_idxs)

            # # temp visualization
            # cluster_strds = world_strands[cluster_idxs]
            # cluster_rgb = strd_clusters_rgb[cluster_idxs]
            # save_color_strands('../temp/KMeans/kmeans_strands_cls_%d.cin'%(i_cls), cluster_strds, cluster_rgb)

        print('Final number of clusters: %d, remove noise strands: %d.'%(num_clusters, self.regular_strands.shape[0] - len(denoised_strds_idxs)))

        self.denoised_regular_strds = self.regular_strands[denoised_strds_idxs]
        self.denoised_strds_features = self.strds_features[denoised_strds_idxs]
        self.denoised_strds_texel_coords = self.strds_texel_coords[denoised_strds_idxs]

        self.denoised_neural_texture = np.zeros((self.texture_height, self.texture_width, self.feature_channels))
        self.denoised_strds_idx_map = np.zeros((self.texture_height, self.texture_width, 1), dtype=np.uint32)
        self.denoised_neural_texture[np.clip(self.texture_height - self.denoised_strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                            np.clip(self.denoised_strds_texel_coords[:, 0], 0, self.texture_width - 1), :] = self.denoised_strds_features[:, :]

        strds_idxs = np.arange(self.denoised_strds_features.shape[0]) + 1
        self.denoised_strds_idx_map[np.clip(self.texture_height - self.denoised_strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                           np.clip(self.denoised_strds_texel_coords[:, 0], 0, self.texture_width - 1), 0] = strds_idxs

        return denoised_strds_idxs, strd_clusters

    def interpolation_on_strds(self, texel_roots_map, interp_kernel_size=5, interp_neig_pts=3, max_dis_thres=16):
        steps_height = self.texture_height // interp_kernel_size
        steps_width = self.texture_width // interp_kernel_size

        # build kd-tree for points
        texel_strds_pts = np.where(self.denoised_strds_idx_map > 0)
        texel_strds_pts = np.concatenate((texel_strds_pts[0][:, None], texel_strds_pts[1][:, None]), axis=1)
        texel_pts_kdtree = KDTree(texel_strds_pts)

        texel_strds_mask = self.denoised_strds_idx_map > 0
        texel_roots_mask = texel_roots_map > 0.5

        interped_strands = []
        interped_strds_face_idxs = []
        interped_strds_face_barys = []
        self.interp_count = 0
        for i_h in range(steps_height):
            for i_w in range(steps_width):
                cen_h = i_h * interp_kernel_size + (interp_kernel_size // 2)
                cen_w = i_w * interp_kernel_size + (interp_kernel_size // 2)

                if texel_roots_mask[cen_h, cen_w] == False or texel_strds_mask[cen_h, cen_w] == True:
                    continue
                
                num_existing_features = np.sum(texel_strds_mask[cen_h - (interp_kernel_size // 2) : cen_h + (interp_kernel_size // 2) + 1,
                                                                   cen_w - (interp_kernel_size // 2) : cen_w + (interp_kernel_size // 2) + 1].astype(np.int16))
                if num_existing_features > 0:
                    continue
                
                dis, idx = texel_pts_kdtree.query(np.array([cen_h, cen_w]), interp_neig_pts)
                dis = np.array(dis)
                if  np.sum(dis) > max_dis_thres * 3:
                    continue

                dis = 1. / dis
                normalized_dis = dis / np.linalg.norm(dis)  # add np.array for avoiding pylance error
                
                knn_strds_idxs = self.denoised_strds_idx_map[texel_strds_pts[idx, 0], texel_strds_pts[idx, 1], 0] # type: ignore for avoiding pylance error # for valid strands in TBN space 
                knn_strands = self.regular_strands[knn_strds_idxs]

                if interp_neig_pts == 1:
                    interped_strand = knn_strands
                else:
                    interped_strand = np.average(knn_strands, axis=0, weights=normalized_dis)

                interped_strands.append(interped_strand)
                interped_strds_face_idxs.append(self.head_index_map[cen_h, cen_w].detach().numpy())
                interped_strds_face_barys.append(self.head_bary_map[cen_h, cen_w].detach().numpy())

                self.interp_count += 1

        interped_strands = np.array(interped_strands)
        interped_strds_face_idxs = np.array(interped_strds_face_idxs)
        interped_strds_face_barys = np.array(interped_strds_face_barys)

        return interped_strands, interped_strds_face_idxs, interped_strds_face_barys 

    def denoise_neural_texture(self, num_del_cls=4, do_denoise=True):
        if do_denoise:
            clustering = MeanShift().fit(self.strds_features)
            num_cls = np.max(clustering.labels_) + 1

            strds_cls = clustering.labels_
            cls_amount = np.zeros(num_cls)
            for i_cls in range(num_cls):
                cls_idx = np.where(strds_cls == i_cls)[0]
                cls_amount[i_cls] = cls_idx.shape[0]

            argsort_cls_idx = np.argsort(cls_amount)
            if num_del_cls == 0:
                num_del_cls = num_cls - 1
            denoised_cls_idx = argsort_cls_idx[num_del_cls:]
            num_denoised_cls = denoised_cls_idx.shape[0]

            denoised_strds_idxs = []
            for i_cls in range(num_denoised_cls):
                strds_idx = np.where(strds_cls == denoised_cls_idx[i_cls])[0].tolist()
                denoised_strds_idxs.extend(strds_idx)
        else:
            denoised_strds_idxs = np.arange(self.strds_features.shape[0]).tolist()

        self.denoised_strds_features = self.strds_features[denoised_strds_idxs]
        self.denoised_strds_texel_coords = self.strds_texel_coords[denoised_strds_idxs]

        self.denoised_neural_texture = np.zeros((self.texture_height, self.texture_width, self.feature_channels))
        self.denoised_strds_idx_map = np.zeros((self.texture_height, self.texture_width, 1), dtype=np.uint32)
        self.denoised_neural_texture[np.clip(self.texture_height - self.denoised_strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                            np.clip(self.denoised_strds_texel_coords[:, 0], 0, self.texture_width - 1), :] = self.denoised_strds_features[:, :]

        strds_idxs = np.arange(self.denoised_strds_features.shape[0]) + 1
        self.denoised_strds_idx_map[np.clip(self.texture_height - self.denoised_strds_texel_coords[:, 1], 0, self.texture_height - 1), 
                           np.clip(self.denoised_strds_texel_coords[:, 0], 0, self.texture_width - 1), 0] = strds_idxs

        return denoised_strds_idxs

    def interpolation_local_average(self, texel_roots_map, interp_kernel_size=5, interp_neig_radius=16):
        steps_height = self.texture_height // interp_kernel_size
        steps_width = self.texture_width // interp_kernel_size

        texel_strds_mask = self.denoised_strds_idx_map > 0
        texel_roots_mask = texel_roots_map > 0.5
        
        self.interp_neural_texture = np.zeros_like(self.denoised_neural_texture)
        self.interp_strds_idx_map = np.zeros_like(self.denoised_strds_idx_map, dtype=np.uint32)

        self.interp_count = 0
        for i_h in range(steps_height):
            for i_w in range(steps_width):
                cen_h = i_h * interp_kernel_size + (interp_kernel_size // 2)
                cen_w = i_w * interp_kernel_size + (interp_kernel_size // 2)

                if texel_roots_mask[cen_h, cen_w] == False or texel_strds_mask[cen_h, cen_w] == True:
                    continue
                
                num_existing_features = np.sum(texel_strds_mask[cen_h - (interp_kernel_size // 2) : cen_h + (interp_kernel_size // 2) + 1,
                                                                   cen_w - (interp_kernel_size // 2) : cen_w + (interp_kernel_size // 2) + 1].astype(np.int16))
                if num_existing_features > 0:
                    continue
                
                # get the neighbor for centroid using neig_radius
                neig_ul = np.clip(np.array([cen_h, cen_w]) - interp_neig_radius, 0, [self.texture_height, self.texture_width])
                neig_br = np.clip(np.array([cen_h, cen_w]) + interp_neig_radius, 0, [self.texture_height, self.texture_width])

                neig = self.neural_texture[neig_ul[0]:neig_br[0], neig_ul[1]:neig_br[1]]
                num_features = np.sum(texel_strds_mask[neig_ul[0]:neig_br[0], neig_ul[1]:neig_br[1]].astype(np.int16))

                if num_features == 0:
                    continue
                
                self.interp_neural_texture[cen_h, cen_w] = np.sum(np.sum(neig, axis=1, keepdims=False), axis=0, keepdims=False) / num_features
                self.interp_strds_idx_map[cen_h, cen_w] = self.interp_count + 1
                self.interp_count += 1

    def interpolation_knn(self, texel_roots_map, interp_kernel_size=5, interp_neig_pts=3, is_bilateral=True, max_dis_thres=16):
        steps_height = self.texture_height // interp_kernel_size
        steps_width = self.texture_width // interp_kernel_size

        # build kd-tree for points
        texel_strds_pts = np.where(self.denoised_strds_idx_map > 0)
        texel_strds_pts = np.concatenate((texel_strds_pts[0][:, None], texel_strds_pts[1][:, None]), axis=1)
        texel_pts_kdtree = KDTree(texel_strds_pts)

        texel_strds_mask = self.denoised_strds_idx_map > 0
        texel_roots_mask = texel_roots_map > 0.5

        self.interp_neural_texture = np.zeros_like(self.denoised_neural_texture)
        self.interp_strds_idx_map = np.zeros_like(self.denoised_strds_idx_map, dtype=np.uint32)

        self.interp_count = 0
        for i_h in range(steps_height):
            for i_w in range(steps_width):
                cen_h = i_h * interp_kernel_size + (interp_kernel_size // 2)
                cen_w = i_w * interp_kernel_size + (interp_kernel_size // 2)

                if texel_roots_mask[cen_h, cen_w] == False or texel_strds_mask[cen_h, cen_w] == True:
                    continue
                
                num_existing_features = np.sum(texel_strds_mask[cen_h - (interp_kernel_size // 2) : cen_h + (interp_kernel_size // 2) + 1,
                                                                   cen_w - (interp_kernel_size // 2) : cen_w + (interp_kernel_size // 2) + 1].astype(np.int16))
                if num_existing_features > 0:
                    continue
                
                dis, idx = texel_pts_kdtree.query(np.array([cen_h, cen_w]), interp_neig_pts)
                dis = np.array(dis)
                if  np.sum(dis) > max_dis_thres * 3:
                    continue

                dis = 1. / dis
                normalized_dis = dis / np.linalg.norm(dis)

                knn_strds_codes = self.denoised_neural_texture[texel_strds_pts[idx, 0], texel_strds_pts[idx, 1]]   # for valid strands in TBN space 
                
                nn_strds_code = knn_strds_codes[0]
                similarities = np.abs(np.dot(knn_strds_codes, nn_strds_code.T)
                                      / (np.linalg.norm(knn_strds_codes, axis=-1) * np.linalg.norm(nn_strds_code, axis=-1)))

                if is_bilateral:
                    interp_weigths = similarities * normalized_dis
                    interp_weigths = interp_weigths / np.linalg.norm(interp_weigths)
                else:
                    interp_weigths = normalized_dis

                if interp_neig_pts == 1:
                    self.interp_neural_texture[cen_h, cen_w] = knn_strds_codes
                else:
                    self.interp_neural_texture[cen_h, cen_w] = np.average(knn_strds_codes, axis=0, weights=interp_weigths)

                self.interp_strds_idx_map[cen_h, cen_w] = self.interp_count + 1
                self.interp_count += 1
        
        print('Interpolation done!')

    def world_strands_from_tbn(self, strands, face_idxs, face_barys):
        if not torch.is_tensor(strands):
            strands = torch.tensor(strands, dtype=torch.float32).cuda()

        if not torch.is_tensor(face_barys):
            face_barys = torch.tensor(face_barys, dtype=torch.float32)

        tbn_basis = torch.stack((self.tangents[0], self.bitangents[0], self.normals[0]), dim=2)[face_idxs]
        # basis change
        orig_points = torch.matmul(tbn_basis.float().cuda(), strands.permute(0, 2, 1)).permute(0, 2, 1)
        # scale
        orig_points = orig_points * 1000. # m -> mm

        # translate to world space with brad and triangle vertices
        triangled_vertices = torch.tensor(self.head_mesh.vertices[self.head_mesh.faces, :])
        roots_triangles = triangled_vertices[face_idxs]
        roots_positions = roots_triangles[:, 0] * face_barys[:, 0:1] + \
                          roots_triangles[:, 1] * face_barys[:, 1:2] + \
                          roots_triangles[:, 2] * face_barys[:, 2:3]
        strds_points = orig_points + roots_positions[:, None, :].cuda()
        return strds_points

    def world_strands_from_texels(self, neural_texture, strds_idx_map, batch_size=300):
        texel_idx = np.where(strds_idx_map > 0)
        strds_codes = neural_texture[texel_idx[0], texel_idx[1], :]
        num_interped = strds_codes.shape[0]

        if not torch.is_tensor(strds_codes):
            strds_codes = torch.tensor(strds_codes, dtype=torch.float32).cuda()
        
        pred_points = torch.zeros((num_interped, self.num_strds_points, 3)).cuda()
        num_batches = math.ceil(num_interped / batch_size)

        loop = tqdm(range(num_batches))
        loop.set_description('Decoding strands')
        for i_b in loop:
            i_start = i_b * batch_size
            i_end = min((i_b + 1) * batch_size, num_interped)
            pred_points[i_start:i_end] = self.decode(strds_codes[i_start:i_end])

        face_idxs = self.head_index_map[texel_idx[0], texel_idx[1]]
        tbn_basis = torch.stack((self.tangents[0], self.bitangents[0], self.normals[0]), dim=2)[face_idxs]
        
        # basis change
        orig_points = torch.matmul(tbn_basis.float().cuda(), pred_points.permute(0, 2, 1)).permute(0, 2, 1)
        # scale
        orig_points = orig_points * 1000. # m -> mm

        # translate to world space with brad and triangle vertices
        triangled_vertices = torch.tensor(self.head_mesh.vertices[self.head_mesh.faces, :])
        roots_triangles = triangled_vertices[face_idxs]
        face_barys = self.head_bary_map[texel_idx[0], texel_idx[1]]
        roots_positions = roots_triangles[:, 0] * face_barys[:, 0:1] + \
                          roots_triangles[:, 1] * face_barys[:, 1:2] + \
                          roots_triangles[:, 2] * face_barys[:, 2:3]
        strds_points = orig_points + roots_positions[:, None, :].cuda()
        return strds_points