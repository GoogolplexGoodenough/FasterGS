/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "../glm/glm.hpp"

#ifndef FULL_MASK
#define FULL_MASK 0xffffffffu
#endif

#define BUCKET_SIZE 128
#define WARP_SIZE 32

// buffer idx
#define POINT_XY 0
#define RGBD 2
#define CON_O 6
#define RECT_MIN 10
#define RECT_MAX 12
#define CONIC 14
#define P_VIEW 17
#define POWER 20
#define RADII 21
#define CLAMPED 22
#define COV3D 25

constexpr int large_threshold = 128;
constexpr int large_split_tile_count = 128;

constexpr float log2e = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;
constexpr float ONE_OF_255 = 1.f / 255.f;

template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
	std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

template <typename T>
constexpr size_t default_align() {
    if constexpr (sizeof(T) <= 4)  return 4;
    if constexpr (sizeof(T) <= 8)  return 8;
    if constexpr (sizeof(T) <= 16) return 16;
    return 128;
}


template <typename T>
static void obtain_auto(char*& chunk, T*& ptr, size_t count)
{
    constexpr size_t align = default_align<T>();
    std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(chunk);
    std::uintptr_t aligned = (raw + align - 1) & ~(align - 1);

    ptr = reinterpret_cast<T*>(aligned);
    chunk = reinterpret_cast<char*>(ptr + count);
}


struct ImageState
{
	uint32_t *bucket_count;
	uint32_t *bucket_offsets;
	float3* pixel_colors;
	uint32_t* max_contrib;

	
	size_t bucket_count_scan_size;
	char * bucket_count_scanning_space;

	static ImageState fromChunk(char*& chunk, size_t N);
};


struct SampleState
{
	uint32_t *bucket_to_tile;
    int *idx_max;
    float *weight_max;
	float *T;
	float* accum_T;
	uint2* bucket_ranges;
	float4 *ar;
	float4* accum_ar;
	static SampleState fromChunk(char*& chunk, size_t C);
};

template<typename T> 
size_t required(size_t P)
{
	char* size = nullptr;
	T::fromChunk(size, P);
	return ((size_t)size) + 128;
}


struct render_load_info
{
	const void* data[WARP_SIZE] = { nullptr };
	int lg2_scale[WARP_SIZE] = { 0 };

	
	render_load_info(const uint32_t* point_list, const float* buffer)
	{
		
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 4:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 8:
				data[lane] = &buffer[0]; // point_xy
				lg2_scale[lane] = 7;
				break;
			case 9:
				data[lane] = &buffer[1];
				lg2_scale[lane] = 7;
				break;
			case 12:
				data[lane] = &buffer[0]; 
				lg2_scale[lane] = 7;
				break;
			case 13:
				data[lane] = &buffer[1];
				lg2_scale[lane] = 7;
				break;
			case 16:
				data[lane] = &buffer[2]; //rgbd
				lg2_scale[lane] = 7;
				break;
			case 17:
				data[lane] = &buffer[3]; 
				lg2_scale[lane] = 7;
				break;
			case 18:
				data[lane] = &buffer[4];
				lg2_scale[lane] = 7;
				break;
			case 19:
				data[lane] = &buffer[5];
				lg2_scale[lane] = 7;
				break;
			case 20:
				data[lane] = &buffer[2];
				lg2_scale[lane] = 7;
				break;
			case 21:
				data[lane] = &buffer[3];
				lg2_scale[lane] = 7;
				break;
			case 22:
				data[lane] = &buffer[4];
				lg2_scale[lane] = 7;
				break;
			case 23:
				data[lane] = &buffer[5];
				lg2_scale[lane] = 7;
				break;
			case 24:
				data[lane] = &buffer[6]; // con_o
				lg2_scale[lane] = 7;
				break;
			case 25:
				data[lane] = &buffer[7];
				lg2_scale[lane] = 7;
				break;
			case 26:
				data[lane] = &buffer[8];
				lg2_scale[lane] = 7;
				break;
			case 27:
				data[lane] = &buffer[9];
				lg2_scale[lane] = 7;
				break;
			case 28:
				data[lane] = &buffer[6]; // con_o
				lg2_scale[lane] = 7;
				break;
			case 29:
				data[lane] = &buffer[7];
				lg2_scale[lane] = 7;
				break;
			case 30:
				data[lane] = &buffer[8];
				lg2_scale[lane] = 7;
				break;
			case 31:
				data[lane] = &buffer[9];
				lg2_scale[lane] = 7;
				break;
			}
		}
	}
};

__forceinline__ __device__ void get_gaussian_features(float2& xy, float3& rgb, float4& con_o, float buf, int offset)
{
	xy = {
		__shfl_sync(~0, buf, 8 + offset),
		__shfl_sync(~0, buf, 9 + offset)
	};
	rgb = {
		__shfl_sync(~0, buf, 16 + offset),
		__shfl_sync(~0, buf, 17 + offset),
		__shfl_sync(~0, buf, 18 + offset)
	};
	con_o = {
		__shfl_sync(~0, buf, 24 + offset),
		__shfl_sync(~0, buf, 25 + offset),
		__shfl_sync(~0, buf, 26 + offset),
		__shfl_sync(~0, buf, 27 + offset)
	};
}


__device__ __forceinline__ float warp_sum_int(int v, unsigned mask = 0xFFFFFFFFu)
{
    // 32-lane warp reduction -> lane 0
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v; // valid in lane 0
}

__device__ __forceinline__ float warp_sum_float(float v, unsigned mask = 0xFFFFFFFFu)
{
    // 32-lane warp reduction -> lane 0
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v; // valid in lane 0
}

__device__ __forceinline__ float2 warp_sum_float2(float2 v, unsigned mask = 0xFFFFFFFFu)
{
    v.x = warp_sum_float(v.x, mask);
    v.y = warp_sum_float(v.y, mask);
    return v; // valid in lane 0
}

__device__ __forceinline__ float3 warp_sum_float3(float3 v, unsigned mask = 0xFFFFFFFFu)
{
    v.x = warp_sum_float(v.x, mask);
    v.y = warp_sum_float(v.y, mask);
    v.z = warp_sum_float(v.z, mask);
    return v; // valid in lane 0
}

__device__ __forceinline__ float4 warp_sum_float4(float4 v, unsigned mask = 0xFFFFFFFFu)
{
    v.x = warp_sum_float(v.x, mask);
    v.y = warp_sum_float(v.y, mask);
    v.z = warp_sum_float(v.z, mask);
    v.w = warp_sum_float(v.w, mask);
    return v; // valid in lane 0
}

__device__ __forceinline__ float warp_sum_float_alllanes(float v, unsigned mask = 0xFFFFFFFFu)
{
    float s = warp_sum_float(v, mask);
    return __shfl_sync(mask, s, 0); // broadcast from lane 0
}

__device__ __forceinline__ float3 warp_sum_float3_alllanes(float3 v, unsigned mask = 0xFFFFFFFFu)
{
    float3 s = warp_sum_float3(v, mask);
    s.x = __shfl_sync(mask, s.x, 0);
    s.y = __shfl_sync(mask, s.y, 0);
    s.z = __shfl_sync(mask, s.z, 0);
    return s;
}


__device__ __forceinline__  void store_to_buffer(
    float* tmp_buffer,
    float2 point_xy,
    float4 rgb_depth,
    float4 conic_opacity
)
{
    // point_xy
    tmp_buffer[0] = point_xy.x;
    tmp_buffer[1] = point_xy.y;

    // rgb_depth
    tmp_buffer[2] = rgb_depth.x;
    tmp_buffer[3] = rgb_depth.y;
    tmp_buffer[4] = rgb_depth.z;
    tmp_buffer[5] = rgb_depth.w;

    // conic_opacity
    tmp_buffer[6] = conic_opacity.x;
    tmp_buffer[7] = conic_opacity.y;
    tmp_buffer[8] = conic_opacity.z;
    tmp_buffer[9] = conic_opacity.w;
}


__device__ __forceinline__ void load_from_buffer(
    const float* tmp_buffer,
    float2& point_xy,
    float4& rgb_depth,
    float4& conic_opacity
)
{
    // point_xy
    point_xy.x = tmp_buffer[0];
    point_xy.y = tmp_buffer[1];

    // rgb_depth
    rgb_depth.x = tmp_buffer[2];
    rgb_depth.y = tmp_buffer[3];
    rgb_depth.z = tmp_buffer[4];
    rgb_depth.w = tmp_buffer[5];

    // conic_opacity
    conic_opacity.x = tmp_buffer[6];
    conic_opacity.y = tmp_buffer[7];
    conic_opacity.z = tmp_buffer[8];
    conic_opacity.w = tmp_buffer[9];
}

__device__ __forceinline__  void store_to_buffer(
    float* tmp_buffer,
    float2 point_xy,
    float4 rgb_depth,
    float4 conic_opacity,
    float2 rect_min,
    float2 rect_max,
    float3 conic,
    float3 p_view,
    float power
)
{
    // point_xy
    tmp_buffer[0] = point_xy.x;
    tmp_buffer[1] = point_xy.y;

    // rgb_depth
    tmp_buffer[2] = rgb_depth.x;
    tmp_buffer[3] = rgb_depth.y;
    tmp_buffer[4] = rgb_depth.z;
    tmp_buffer[5] = rgb_depth.w;

    // conic_opacity
    tmp_buffer[6] = conic_opacity.x;
    tmp_buffer[7] = conic_opacity.y;
    tmp_buffer[8] = conic_opacity.z;
    tmp_buffer[9] = conic_opacity.w;

    // rect_min
    tmp_buffer[10] = rect_min.x;
    tmp_buffer[11] = rect_min.y;

    // rect_max
    tmp_buffer[12] = rect_max.x;
    tmp_buffer[13] = rect_max.y;

    // conic
    tmp_buffer[14] = conic.x;
    tmp_buffer[15] = conic.y;
    tmp_buffer[16] = conic.z;

    // p_view
    tmp_buffer[17] = p_view.x;
    tmp_buffer[18] = p_view.y;
    tmp_buffer[19] = p_view.z;

    tmp_buffer[20] = power;
}


__device__ __forceinline__ void load_from_buffer(
    const float* tmp_buffer,
    float2& point_xy,
    float4& rgb_depth,
    float4& conic_opacity,
    float2& rect_min,
    float2& rect_max,
    float3& conic,
    float3& p_view,
    float& power
)
{
    // point_xy
    point_xy.x = tmp_buffer[0];
    point_xy.y = tmp_buffer[1];

    // rgb_depth
    rgb_depth.x = tmp_buffer[2];
    rgb_depth.y = tmp_buffer[3];
    rgb_depth.z = tmp_buffer[4];
    rgb_depth.w = tmp_buffer[5];

    // conic_opacity
    conic_opacity.x = tmp_buffer[6];
    conic_opacity.y = tmp_buffer[7];
    conic_opacity.z = tmp_buffer[8];
    conic_opacity.w = tmp_buffer[9];

    // rect_min
    rect_min.x = tmp_buffer[10];
    rect_min.y = tmp_buffer[11];

    // rect_max
    rect_max.x = tmp_buffer[12];
    rect_max.y = tmp_buffer[13];

    // conic
    conic.x = tmp_buffer[14];
    conic.y = tmp_buffer[15];
    conic.z = tmp_buffer[16];

    // p_view
    p_view.x = tmp_buffer[17];
    p_view.y = tmp_buffer[18];
    p_view.z = tmp_buffer[19];

    power = tmp_buffer[20];
}

__device__ __forceinline__  void write_tmp_to_global(
    const float* tmp_buffer,
    float* global_buffer,
    int tid, int tmp_buffer_size = 10)
{
    int base = tid << 5;
    // int base = tid * 32;
    for (int i = 0; i < tmp_buffer_size; ++i)
        global_buffer[base + i] = tmp_buffer[i];
}

__device__ __forceinline__  void read_global_to_tmp(
    const float* global_buffer,
    float* tmp_buffer,
    int tid, int tmp_buffer_size = 10)
{
    int base = tid << 5;
    // int base = tid * 32;

    for (int i = 0; i < tmp_buffer_size; ++i)
        tmp_buffer[i] = global_buffer[base + i];
}



__forceinline__ __device__ float fast_max_f32(float a, float b) {
  float d;
  asm volatile("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

__forceinline__ __device__ float fast_sqrt_f32(float x) {
  float y;
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float fast_rsqrt_f32(float x) {
  float y;
  asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ float fast_lg2_f32(float x) {
  float y;
  asm volatile("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}


__forceinline__ __device__ float ndc2Pix(float v, int S) {
  return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float Pix2ndc(float r, int S)
{
	return (2 * r + 1) / S - 1;
}

__forceinline__ __device__ float3 transformPoint4x3(const glm::vec3 &p,
                                                    const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const glm::vec3 &p,
                                                    const float *matrix) {
  float4 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

// __forceinline__ __device__ float4 transformPoint4x4(const float3 &p,
//                                                     const float *matrix) {
//   float4 transformed = {
//       matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
//       matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
//       matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
//       matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
//   return transformed;
// }

__forceinline__ __device__ void getRect(const float2 p, int width, int height,
                                        int2 &rect_min, int2 &rect_max,
                                        dim3 grid, int block_x, int block_y) {
  rect_min = {
      min((int)grid.x, max((int)0, (int)((p.x - width) / (float)block_x))),
      min((int)grid.y, max((int)0, (int)((p.y - height) / (float)block_y)))};
  rect_max = {
      min((int)grid.x, max((int)0, (int)((p.x + width) / (float)block_x) + 1)),
      min((int)grid.y,
          max((int)0, (int)((p.y + height) / (float)block_y) + 1))};
}