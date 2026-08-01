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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class Buffer:
    def __init__(self, device='cuda', MAX_NUM_RENDERED=2**24, MAX_NUM_TILES=2**20, **kwargs):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = _C.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=device, dtype=torch.int32)

        # 40 bytes
        self.splat_buffer = torch.zeros([MAX_NUM_RENDERED, 32], device=device, dtype=torch.float32)
        self.device = device


def rasterize_gaussians(
    means3D,
    dc,
    shs,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    culling,
    xyz_gradient_accum,
    xyz_gradient_accum_abs,
    max_radii,
    denom
):
    return _RasterizeGaussians.apply(
        means3D,
        dc,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        culling,
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        max_radii,
        denom
    )



class _RasterizeGaussians(torch.autograd.Function):
    buffer = Buffer()

    @staticmethod
    def forward(
        ctx,
        means3D,
        dc,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        culling,
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        max_radii,
        denom
    ):
        buffer = _RasterizeGaussians.buffer
        splat_buffer = _RasterizeGaussians.buffer.splat_buffer
        # Restructure arguments the way that the C++ lib expects them
        zFar = 100.0
        zNear = 0.01

        N = means3D.shape[0]
        
        buffer.curr_offset.fill_(0)
        radii = _C.ops.preprocess(
            raster_settings.sh_degree,
            means3D, colors_precomp, dc, shs, opacities, scales, rotations, raster_settings.scale_modifier,
            raster_settings.image_width, raster_settings.image_height, 16, 16,
            raster_settings.campos, raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy, zFar, zNear,
            buffer.gaussian_keys_unsorted, buffer.gaussian_values_unsorted,
            buffer.gaussian_values_sorted,
            buffer.curr_offset, raster_settings.mult, splat_buffer, culling
        )
        # torch.cuda.synchronize()
        num_rendered = int(buffer.curr_offset.cpu()[0])
        if num_rendered >= buffer.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"
        
        _C.ops.sort_gaussian(
            num_rendered, raster_settings.image_width, raster_settings.image_height, 16, 16,
            buffer.list_sorting_space,
            buffer.gaussian_keys_unsorted, buffer.gaussian_values_unsorted,
            buffer.gaussian_keys_sorted, buffer.gaussian_values_sorted
        )
        
        # out_color = torch.zeros((raster_settings.image_height, raster_settings.image_width, 3), device=buffer.device, dtype=torch.float)
        
        last_contributor = torch.zeros((raster_settings.image_height, raster_settings.image_width, 1), device=buffer.device, dtype=torch.int)
        Ts_final = torch.zeros((raster_settings.image_height, raster_settings.image_width, 1), device=buffer.device, dtype=torch.float)

        bucket_sum, out_color, img_buffer, smp_buffer = _C.ops.render_16x16(
            N, num_rendered, raster_settings.image_width, raster_settings.image_height,
            splat_buffer,
            buffer.gaussian_keys_sorted, buffer.gaussian_values_sorted,
            buffer.ranges, raster_settings.bg, 
            last_contributor, Ts_final
        )

        
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.buffer = buffer
        ctx.bucket_sum = bucket_sum
        ctx.save_for_backward(
            means3D, scales, rotations, dc, shs, opacities, out_color,
            last_contributor, Ts_final, radii, splat_buffer, img_buffer, smp_buffer,
            
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            max_radii,
            denom
        )
        return out_color, radii
    

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        buffer = ctx.buffer
        bucket_sum = ctx.bucket_sum
        zFar = 100.0
        zNear = 0.01
        means3D, scales, rotations, dc, shs, opacities, out_color, last_contributor, Ts_final, radii, splat_buffer, img_buffer, smp_buffer, xyz_gradient_accum, xyz_gradient_accum_abs, max_radii, denom = ctx.saved_tensors

        grad_mean2D, grad_color, grad_opacity, grad_mean3D, grad_cov3D, grad_dc, grad_shs, grad_scales, grad_rotations = _C.ops.backward(
            raster_settings.sh_degree,
            bucket_sum,
            means3D, radii, dc, shs, opacities,
            scales, rotations, raster_settings.scale_modifier, 
            raster_settings.image_width, raster_settings.image_height, 16, 16,
            raster_settings.tanfovx, raster_settings.tanfovy, zFar, zNear,
            raster_settings.campos, raster_settings.viewmatrix, raster_settings.projmatrix,
            img_buffer, smp_buffer, splat_buffer,
            buffer.gaussian_values_sorted,
            buffer.ranges, raster_settings.bg,
            last_contributor, Ts_final,
            grad_out_color,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            max_radii,
            denom
        )

        return grad_mean3D, grad_dc, grad_shs, grad_color, grad_opacity, grad_scales, grad_rotations, grad_cov3D, None, None, None, None, None, None
    
        

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    mult : float
    prefiltered : bool
    debug : bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, **kwargs):
        super().__init__()
        self.raster_settings = raster_settings


    def forward(self, means3D, opacities, dc = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, culling = None, xyz_gradient_accum = None, xyz_gradient_accum_abs = None, max_radii = None, denom = None):
        
        raster_settings = self.raster_settings
        # buffer = self.buffer

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if dc is None:
            dc = torch.tensor([])
        if shs is None:
            shs = torch.tensor([])
        if culling is None:
            culling = torch.tensor([], dtype=torch.bool)
        if colors_precomp is None:
            colors_precomp = torch.tensor([])

        if scales is None:
            scales = torch.tensor([])
        if rotations is None:
            rotations = torch.tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.tensor([])
        if xyz_gradient_accum is None:
            xyz_gradient_accum = torch.tensor([])
        if xyz_gradient_accum_abs is None:
            xyz_gradient_accum_abs = torch.tensor([])
        if max_radii is None:
            max_radii = torch.tensor([])
        if denom is None:
            denom = torch.tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            dc,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            culling,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            max_radii,
            denom
        )
    
    
class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None or torch.prod(torch.tensor(param.grad.shape))==0:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]

            # compensate lr for sparse adam, (1-b2**step)**0.5/(1-b1**step)
            state['step']+=1
            step=state['step']

            M = param.numel() // N
            _C.ops.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr*(1-0.999**step)**0.5/(1-0.9**step), 0.9, 0.999, eps, N, M)