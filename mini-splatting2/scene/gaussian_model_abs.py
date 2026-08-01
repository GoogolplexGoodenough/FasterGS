#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import SH2RGB

try:
    from diff_gaussian_rasterization_ms_2 import SparseGaussianAdam
except:
    pass

def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest    
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        if (len(extra_f_names)!=3*(self.max_sh_degree + 1) ** 2 - 3):
            features_extra = np.zeros((xyz.shape[0], 3, 0))
            self.active_sh_degree = 0

        else:
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self.active_sh_degree = self.max_sh_degree

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum = torch.zeros_like(self.xyz_gradient_accum[valid_points_mask])
        self.xyz_gradient_accum_abs = torch.zeros_like(self.xyz_gradient_accum_abs[valid_points_mask])

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._culling = self._culling[valid_points_mask]
        self.factor_culling = self.factor_culling[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # grads = self.xyz_gradient_accum / self.denom
        grads = self.xyz_gradient_accum / (1 - self.denom)
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def add_densification_stats_culling(self, viewspace_point_tensor, update_filter, factor):
        self.xyz_gradient_accum[update_filter] += (torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)*factor[update_filter])
        self.xyz_gradient_accum_abs[update_filter] += (torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)*factor[update_filter])
        self.denom[update_filter] += 1        


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        new_culling = self._culling[selected_pts_mask].repeat(N,1)
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))                  

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone_mask(self, grads, max_grad, mask):
        
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, mask)

        self._opacity[selected_pts_mask] = self.inverse_opacity_activation(1 - torch.sqrt(1 - self.get_opacity[selected_pts_mask]))

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))


    def densify_and_prune_mask(self, max_grad, min_opacity, extent, max_screen_size, mask_split):
        grads = self.xyz_gradient_accum / self.denom
        # grads = self.xyz_gradient_accum / (1 - self.denom)
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask_split)
        # print(mask_split.shape)
        
        # self.densify_and_clone_mask(grads, max_grad, mask_split)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()



    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)
        

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        new_culling = self._culling[selected_pts_mask].repeat(N,1)
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask].repeat(N,1)
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))          

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)



    def reinitial_pts(self, pts, rgb):

        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  



    def init_culling(self, num_views):
        self._culling=torch.zeros((self._xyz.shape[0], num_views), dtype=torch.bool, device='cuda')
        self.factor_culling=torch.ones((self._xyz.shape[0],1), device='cuda')




    def depth_reinit(self, scene, render_depth, iteration, num_depth, args, pipe, background):

        out_pts_list=[]
        gt_list=[]
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            gt = view.original_image[0:3, :, :]

            render_depth_pkg = render_depth(view, self, pipe, background, culling=self._culling[:,view.uid])


            out_pts = render_depth_pkg["out_pts"]
            accum_alpha = render_depth_pkg["accum_alpha"]


            prob=1-accum_alpha

            prob = prob/prob.sum()
            prob = prob.reshape(-1).cpu().numpy()

            factor=1/(gt.shape[1]*gt.shape[2]*len(views)/num_depth)

            N_xyz=prob.shape[0]
            num_sampled=int(N_xyz*factor)

            indices = np.random.choice(N_xyz, size=num_sampled, 
                                        p=prob,replace=False)
            
            out_pts = out_pts.permute(1,2,0).reshape(-1,3)
            gt = gt.permute(1,2,0).reshape(-1,3)

            out_pts_list.append(out_pts[indices])
            gt_list.append(gt[indices])       


        out_pts_merged=torch.cat(out_pts_list)
        gt_merged=torch.cat(gt_list)

        return out_pts_merged, gt_merged
    

    def interesction_sampling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
        
        imp_score[accum_area_max==0]=0
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()


        factor=args.sampling_factor
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        mask = np.zeros(N_xyz, dtype=bool)
        mask[indices] = True

        self.prune_points(mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    def interesction_preserving(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights
            
        imp_score[accum_area_max==0]=0
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
        self.prune_points(non_prune_mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    def importance_pruning(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            imp_score=imp_score+accum_weights
            
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
        self.prune_points(non_prune_mask==False)

        return self._xyz, SH2RGB(self._features_dc+0)[:,0]


    

    def visibility_culling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1

            imp_score=imp_score+accum_weights

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 

        self.factor_culling=count_vis/(count_rad+1e-1)

        mask = (count_vis<=1)[:,0]
        mask = torch.logical_or(mask, non_prune_mask==False)
        self.prune_points(mask) 


    def aggressive_clone(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        for view in views:
            # render_pkg = render_simp(view, self, pipe, background)
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])

            accum_weights = render_pkg["accum_weights"]
            area_max = render_pkg["area_max"]

            imp_score=imp_score+accum_weights
            accum_area_max = accum_area_max+area_max

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 
        self.prune_points(~non_prune_mask) 

        imp_score[accum_area_max==0]=0
        intersection_pts_mask = init_cdf_mask(importance=imp_score, thres=0.99)
        intersection_pts_mask=intersection_pts_mask[non_prune_mask]
        self.clone(intersection_pts_mask)


    # aggressive_clone with visibility_culling
    def culling_with_clone(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            # render_pkg = render_simp(view, self, pipe, background)
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])

            accum_weights = render_pkg["accum_weights"]
            area_max = render_pkg["area_max"]

            imp_score=imp_score+accum_weights
            accum_area_max = accum_area_max+area_max

            visibility_mask = init_cdf_mask(importance=accum_weights, thres=0.99)
            self._culling[:,view.uid]=(visibility_mask==False)
            count_rad[render_pkg["radii"]>0] += 1
            count_vis[visibility_mask] += 1

        self.factor_culling=count_vis/(count_rad+1e-1)

        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.999) 
        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


        imp_score[accum_area_max==0]=0
        intersection_pts_mask = init_cdf_mask(importance=imp_score, thres=0.99)

        intersection_pts_mask=intersection_pts_mask[~prune_mask]
        self.clone(intersection_pts_mask)


    def clone(self, selected_pts_mask):
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        temp_opacity_old = self.get_opacity[selected_pts_mask]
        new_opacity = 1-(1-temp_opacity_old)**0.5

        temp_scale_old = self.get_scaling[selected_pts_mask]
        new_scaling = (temp_opacity_old / (2*new_opacity-0.5**0.5*new_opacity**2)) * temp_scale_old

        new_opacity = torch.clamp(new_opacity, max=1.0 - torch.finfo(torch.float32).eps, min=0.0051)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling)   


        self._opacity[selected_pts_mask] = new_opacity
        self._scaling[selected_pts_mask] = new_scaling


        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        new_culling = self._culling[selected_pts_mask]
        self._culling = torch.cat((self._culling, new_culling))
        new_factor_culling = self.factor_culling[selected_pts_mask]
        self.factor_culling = torch.cat((self.factor_culling, new_factor_culling))
    

    # interesction_preserving with visibility_culling
    def culling_with_interesction_preserving(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights            

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        imp_score[accum_area_max==0]=0
        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 

        self.factor_culling=count_vis/(count_rad+1e-1)


        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


    # interesction_sampling with visibility_culling
    def culling_with_interesction_sampling(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]
            area_proj = render_pkg["area_proj"]
            area_max = render_pkg["area_max"]

            accum_area_max = accum_area_max+area_max

            if args.imp_metric=='outdoor':
                mask_t=area_max!=0 
                temp=imp_score+accum_weights/area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score=imp_score+accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        imp_score[accum_area_max==0]=0
        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()

        factor=args.sampling_factor
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        non_prune_mask = np.zeros(N_xyz, dtype=bool)
        non_prune_mask[indices] = True


        self.factor_culling=count_vis/(count_rad+1e-1)

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, torch.tensor(non_prune_mask==False, device='cuda'))
        self.prune_points(prune_mask) 


    # importance_pruning with visibility_culling
    def culling_with_importance_pruning(self, scene, render_simp, iteration, args, pipe, background):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        views = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        self._culling=torch.zeros((self._xyz.shape[0], len(views)), dtype=torch.bool, device='cuda')

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self, pipe, background, culling=self._culling[:,view.uid])
            accum_weights = render_pkg["accum_weights"]

            imp_score=imp_score+accum_weights      

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)

            self._culling[:,view.uid]=(non_prune_mask==False)

            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 

        self.factor_culling=count_vis/(count_rad+1e-1)

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


    def extend_features_rest(self):

        features = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))








