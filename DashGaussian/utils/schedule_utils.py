# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math
import torch
from arguments import OptimizationParams, PipelineParams

from scene.gaussian_model import GaussianModel
from utils.general_utils import get_expon_lr_func


class TrainingScheduler():
	"""
	DashGaussian training scheduler of resolution and primitive number.
	"""
	def __init__(self, opt: OptimizationParams, pipe: PipelineParams, gaussians: GaussianModel, original_images: list) -> None:

		self.max_steps = opt.iterations
		self.init_n_gaussian = gaussians.get_xyz.shape[0]

		self.densify_mode = pipe.densify_mode
		self.densify_until_iter = opt.densify_until_iter
		self.densification_interval = opt.densification_interval

		self.resolution_mode = pipe.resolution_mode

		self.start_significance_factor = 4
		self.max_reso_scale = 8
		self.reso_sample_num = 32 # Must be no less than 2
		self.max_densify_rate_per_step = 0.2
		self.reso_scales = None
		self.reso_level_significance = None
		self.reso_level_begin = None
		self.increase_reso_until = self.densify_until_iter
		self.next_i = 2

		if pipe.max_n_gaussian > 0:
			self.max_n_gaussian = pipe.max_n_gaussian
			self.momentum = -1
		else:
			self.momentum = 5 * self.init_n_gaussian
			self.max_n_gaussian = self.init_n_gaussian + self.momentum
			self.integrate_factor = 0.98
			self.momentum_step_cap = 1000000
		
		# Generate schedulers.
		self.init_reso_scheduler(original_images)

		if self.resolution_mode == "freq":
			gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale, 
													         lr_final=opt.position_lr_final*gaussians.spatial_lr_scale, 
															 lr_delay_mult=opt.position_lr_delay_mult, 
															 max_steps=opt.position_lr_max_steps, 
															 decay_from_iter=self.lr_decay_from_iter())
	
	def update_momentum(self, momentum_step):
		if self.momentum == -1:
			return
		self.momentum = max(self.momentum, 
					        int(self.integrate_factor * self.momentum + min(self.momentum_step_cap, momentum_step)))
		self.max_n_gaussian = self.init_n_gaussian + self.momentum

	def get_res_scale(self, iteration):
		if self.resolution_mode == "const":
			return 1
		elif self.resolution_mode == "freq":
			if iteration >= self.increase_reso_until:
				return 1
			if iteration < self.reso_level_begin[1]:
				return self.reso_scales[0]
			while iteration >= self.reso_level_begin[self.next_i]:
				# If the index is out of the range of 'reso_level_begin', there must be something wrong with the scheduler.
				self.next_i += 1
			i = self.next_i - 1
			i_now, i_nxt = self.reso_level_begin[i: i + 2]
			s_lst, s_now = self.reso_scales[i - 1: i + 1]
			scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
			return int(scale)
		else:
			raise NotImplementedError("Resolution mode '{}' is not implemented.".format(self.resolution_mode))
	
	def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
		if self.densify_mode == "free":
			return 1.0
		elif self.densify_mode == "freq":
			assert cur_scale is not None
			if self.densification_interval + iteration < self.increase_reso_until:
				next_n_gaussian = int((self.max_n_gaussian - self.init_n_gaussian) / cur_scale**(2 - iteration / self.densify_until_iter)) + self.init_n_gaussian
			else:
				next_n_gaussian = self.max_n_gaussian
			return min(max((next_n_gaussian - cur_n_gaussian) / cur_n_gaussian, 0.), self.max_densify_rate_per_step)
		else:
			raise NotImplementedError("Densify mode '{}' is not implemented.".format(self.densify_mode))
	
	def lr_decay_from_iter(self):
		if self.resolution_mode == "const":
			return 1
		for i, s in zip(self.reso_level_begin, self.reso_scales):
			if s < 2:
				return i
		raise Exception("Something is wrong with resolution scheduler.")

	def init_reso_scheduler(self, original_images):
		if self.resolution_mode != "freq":
			print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(self.resolution_mode))
			return

		def compute_win_significance(significance_map: torch.Tensor, scale: float):
			h, w = significance_map.shape[-2:]
			c = ((h + 1) // 2, (w + 1) // 2)
			win_size = (int(h / scale), int(w / scale))
			win_significance = significance_map[..., c[0]-win_size[0]//2: c[0]+win_size[0]//2, c[1]-win_size[1]//2: c[1]+win_size[1]//2].sum().item()
			return win_significance
		
		def scale_solver(significance_map: torch.Tensor, target_significance: float):
			L, R, T = 0., 1., 64
			for _ in range(T):
				mid = (L + R) / 2
				win_significance = compute_win_significance(significance_map, 1 / mid)
				if win_significance < target_significance:
					L = mid
				else:
					R = mid
			return 1 / mid
		
		print("[ INFO ] Initializing resolution scheduler...")

		self.max_reso_scale = 8
		self.next_i = 2
		scene_freq_image = None
		
		for img in original_images:
			img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
			img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
			scene_freq_image = img_fft_centered_mod if scene_freq_image is None else scene_freq_image + img_fft_centered_mod

			e_total = img_fft_centered_mod.sum().item()
			e_min = e_total / self.start_significance_factor
			self.max_reso_scale = min(self.max_reso_scale, scale_solver(img_fft_centered_mod, e_min))

		modulation_func = math.log

		self.reso_scales = []
		self.reso_level_significance = []
		self.reso_level_begin = []
		scene_freq_image /= len(original_images)
		E_total = scene_freq_image.sum().item()
		E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
		self.reso_level_significance.append(E_min)
		self.reso_scales.append(self.max_reso_scale)
		self.reso_level_begin.append(0)
		for i in range(1, self.reso_sample_num - 1):
			self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
			self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
			self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
			self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
		self.reso_level_significance.append(modulation_func(E_total / E_min))
		self.reso_scales.append(1.)
		self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
		self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
		self.reso_level_begin.append(self.increase_reso_until)

		# print("================== Resolution Scheduler ==================")
		# for idx, (e, s, i) in enumerate(zip(self.reso_level_significance, self.reso_scales, self.reso_level_begin)):
		# 	print(" - idx: {:02d}; scale: {:.2f}; significance: {:.2f}; begin: {}".format(idx, s, e, i))
		# print("==========================================================")
