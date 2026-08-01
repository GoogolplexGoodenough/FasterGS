#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include "glm/glm.hpp"
#include <functional>
#include <torch/extension.h>

#define DEBUG_TIME 0
#define DEBUG_MEM 0
#define DEBUG_RENDER 0
#define DEBUG_PIX_X 283
#define DEBUG_PIX_Y 190

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f};

constexpr int FLASHGS_WARP_SIZE = 32;

#define FLASHGS_CHECK_CUDA(x)                                                         \
	{                                                                                 \
		cudaError_t status = x;                                                       \
		if (status != cudaSuccess)                                                    \
		{                                                                             \
			fprintf(stderr, "%s\nline = %d\n", cudaGetErrorString(status), __LINE__); \
			exit(1);                                                                  \
		}                                                                             \
	}

#if DEBUG_MEM
#define CHECK_CUDA(msg)                                                    \
	do                                                                     \
	{                                                                      \
		cudaError_t err__ = cudaGetLastError();                            \
		if (err__ != cudaSuccess)                                          \
		{                                                                  \
			fprintf(stderr, "[CUDA] After %s launch: %s (%s:%d)\n",        \
					(msg), cudaGetErrorString(err__), __FILE__, __LINE__); \
		}                                                                  \
		err__ = cudaDeviceSynchronize();                                   \
		if (err__ != cudaSuccess)                                          \
		{                                                                  \
			fprintf(stderr, "[CUDA] After %s sync: %s (%s:%d)\n",          \
					(msg), cudaGetErrorString(err__), __FILE__, __LINE__); \
		}                                                                  \
	} while (0)
#else
#define CHECK_CUDA(msg) \
	do                  \
	{                   \
		(void)(msg);    \
	} while (0)
#endif

namespace faster
{
	void preprocess(
		int P, int D, int max_coeffs,
		glm::vec3 *positions, float *color_precomp,
		float *dc, float *shs, float *opacities,
		float *scales, float *rotations, float scale_modifier,
		int width, int height, int block_x, int block_y,
		glm::vec3 *cam_position,
		float *view_matrix, float *proj_matrix,
		float tan_fovx, float tan_fovy, float zFar, float zNear,
		float *splat_buffer,
		uint64_t *gaussian_keys_unsorted,
		uint32_t *gaussian_values_unsorted,
		uint32_t *gaussian_values_sorted,
		int *curr_offset, int *radii, float mult, bool *culling,
		cudaStream_t stream = 0);

	void sort_gaussian(
		int num_rendered,
		int width, int height, int block_x, int block_y,
		char *list_sorting_space, size_t sorting_size,
		uint64_t *gaussian_keys_unsorted, uint32_t *gaussian_values_unsorted,
		uint64_t *gaussian_keys_sorted, uint32_t *gaussian_values_sorted, cudaStream_t stream = 0);

	size_t get_sort_buffer_size(int num_rendered, cudaStream_t stream = 0);

	uint32_t render_16x16(
		int P, int num_rendered,
		int width, int height,
		float *splat_buffer,
		uint64_t *gaussian_keys_sorted, uint32_t *gaussian_values_sorted,
		uint2 *ranges, float3 *bg_color, float *out_color,
		int *last_contributor, float *Ts_final,

		std::function<char *(size_t)> img_func,
		std::function<char *(size_t)> smp_func,
		cudaStream_t stream = 0);

	void preprocess_backward(
		int P, int D, int M, const float3 *means3D,
		const int *radii, const float *dc, const float *shs,
		const float *splat_buffer,
		const glm::vec3 *scales,
		const glm::vec4 *rotations, const float scale_modifier,
		const float *viewmatrix,
		const float *projmatrix, const float focal_x,
		float focal_y, const float tan_fovx, float tan_fovy,
		const glm::vec3 *campos, const float4 *dL_dmean2D,
		const float *dL_dconic, glm::vec3 *dL_dmean3D,
		float *dL_dcolor, float *dL_dcov3D, float *dL_ddc, float *dL_dsh,
		glm::vec3 *dL_dscale, glm::vec4 *dL_drot,
		cudaStream_t stream = 0);

	void render_backward(
		uint32_t bucket_sum, int width, int height,
		const uint32_t *point_list, const float *bg_color,
		const float *splat_buffer,
		const float *final_Ts,
		const uint32_t *n_contrib,
		char *image_buffer,
		char *sample_buffer,
		const float *dL_dpixels,
		float4 *dL_dmean2D, float4 *dL_dconic2D,
		float *dL_dopacity, float *dL_dcolor,
		cudaStream_t stream = 0);

	void add_densification_stats(
		int P,
		const int *radii,
		const float4 *dL_dmean2D,
		float *xyz_gradient_accum,
		float *xyz_gradient_accum_abs,
		float *max_radii,
		float *denom,
		cudaStream_t stream = 0);

	uint32_t render_16x16_simp(int P, int num_rendered, int width, int height,
							   float *splat_buffer, uint64_t *gaussian_keys_sorted,
							   uint32_t *gaussian_values_sorted, uint2 *ranges,
							   float *accum_weights_ptr, int *accum_weights_count,
							   float *accum_max_count, cudaStream_t stream = 0);

	void render_16x16_depth(int P, int num_rendered, int width, int height,
							float *splat_buffer, uint64_t *gaussian_keys_sorted,
							uint32_t *gaussian_values_sorted, uint2 *ranges,
							float *means3D, glm::vec3 *scales, glm::vec4 *rotations,
							float *projmatrix, glm::vec3 *campos, float *out_pts,
							float *out_depth, float *accum_alpha, int *gidx,
							float *discriminants,
							cudaStream_t stream = 0);

} // namespace faster