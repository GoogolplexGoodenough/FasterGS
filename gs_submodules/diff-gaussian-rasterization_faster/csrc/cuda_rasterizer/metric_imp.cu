#include "../ops.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rasterizer_imp.h"

namespace cg = cooperative_groups;

namespace faster {
namespace {


__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32; //32
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; //32
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1){
		ranges[currtile].y = L;
	}
}


__forceinline__ __device__ float fast_ex2_ftz_f32(float x)
{
	float y;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}


__forceinline__ __device__ void pixel_shader(float& T, int& metric_count, int2 pix, float2 xy, float4 con_o, float3 rgb, int metric)
{
	float2 d = { xy.x - (float)pix.x, xy.y - (float)pix.y };
	//float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
	float power = con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
	float alpha;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
	alpha = min(0.99f, alpha);
	float test_T = T * (1.f - alpha);
	if (test_T > 0.0001f && alpha > ONE_OF_255){
		T = test_T;
        if (metric == 1){
            metric_count += 1; 
        }
	}
}


template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ void metricCountCUDA(
	const int P,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	render_load_info info,
	float3* bg_color,
	float* __restrict__ out_color,
	int* __restrict__ last_contributor,
	float* __restrict__ Ts_final,
	bool get_flag, const int* __restrict__ metric_map, int* __restrict__ metric_count
)
{
	uint2 range = ranges[blockIdx.y * x_blocks + blockIdx.x];
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	// uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y };
	int2 pix[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				(int)blockIdx.x * BLOCK_X + (int)threadIdx.x * THREAD_X + j,
				(int)blockIdx.y * BLOCK_Y + (int)threadIdx.y * THREAD_Y + i
			};
		}
	}

    int local_map[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1){
				continue;
			}
            int pix_id = pix[i][j].y * width + pix[i][j].x;
			local_map[i][j] = metric_map[pix_id];
		}
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	int to_do = range.y - range.x;
	int offset = range.x;
	bool done = false;
	float buf, ldg_buf;
	float2 xy;
	float3 rgb;
	float4 rgbd;
	float4 con_o;
	int point_id, ldg_point_id, curr_id;
    int local_count = 0;
	bool load_enable = data != nullptr;
    int warp_accum_count = 0;
    
	if (to_do % 2 != 0) {
		point_id = point_list[offset];
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)point_id << lg2_scale));
		bool load_enable = data != nullptr;
		if (load_enable) buf = __ldg(ptr);
		get_gaussian_features(xy, rgb, con_o, buf, 0);

        local_count = 0;
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1){
					continue;
				}
				pixel_shader(T[i][j], local_count, pix[i][j], xy, con_o, rgb, local_map[i][j]);
			}
		}
        warp_accum_count = warp_sum_int(local_count);
        if (lane == 0 && point_id < P){
            atomicAdd(&metric_count[point_id], warp_accum_count);
        }

		to_do -= 1;
		offset += 1;
	}
	done = to_do == 0? true: false;
	offset = (lane & 4) == 0? offset: offset + 1;

	if (to_do > 0){
		point_id = point_list[offset];
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)point_id << lg2_scale));
		if (load_enable) buf = __ldg(ptr);
		offset += 2;
		to_do -= 2;
	}

	while(__any_sync(~0, to_do >= 0 && !done)){
        if (to_do > 0){
            ldg_point_id = point_list[offset];
            const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)ldg_point_id << lg2_scale));
            if (load_enable) ldg_buf = __ldg(ptr);
        }

		get_gaussian_features(xy, rgb, con_o, buf, 0);
        curr_id = __shfl_sync(~0, point_id, 0);
        local_count = 0;

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1){
					continue;
				}
				pixel_shader(T[i][j], local_count, pix[i][j], xy, con_o, rgb, local_map[i][j]);
			}
		}
        warp_accum_count = warp_sum_int(local_count);
        if (lane == 0 && curr_id < P){
            atomicAdd(&metric_count[curr_id], warp_accum_count);
        }
		
		get_gaussian_features(xy, rgb, con_o, buf, 4);
        curr_id = __shfl_sync(~0, point_id, 4);
        local_count = 0;
		
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1){
					continue;
				}
				pixel_shader(T[i][j], local_count, pix[i][j], xy, con_o, rgb, local_map[i][j]);
			}
		}
        warp_accum_count = warp_sum_int(local_count);
        if (lane == 0 && curr_id < P){
            atomicAdd(&metric_count[curr_id], warp_accum_count);
        }

		done = true;
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				done = done && T[i][j] < 0.0001f;
			}
		}
		
		to_do -= 2;
		offset += 2;
		buf = ldg_buf;
        point_id = ldg_point_id;
	}
}


template<int BLOCK_X, int BLOCK_Y>
void metricCount(
	int P,
	int num_rendered,
	int width, int height,
	float* splat_buffer,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	uint2* ranges, float3* bg_color, float* out_color, 
	int* last_contributor, float* Ts_final,
	bool get_flag, int* metric_map, int* metric_count,
	cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

    // Identify start and end of per-tile workloads in sorted list
    identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
        num_rendered,
        gaussian_keys_sorted,
        ranges);

	metricCountCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4><<<grid, dim3(8, 4, 1), 0, stream>>>(
		P,
		ranges,
		gaussian_values_sorted,
		width, height, grid.x,
		render_load_info(gaussian_values_sorted, splat_buffer),
		bg_color,
		out_color,
		last_contributor, Ts_final,
		get_flag, metric_map, metric_count
	);
	CHECK_CUDA("metricCountCUDA");
}

} // namespace

uint32_t getMetricCount(
	int P,
	int num_rendered,
	int width, int height,
	float* splat_buffer,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	uint2* ranges, float3* bg_color, float* out_color, 
	int* last_contributor, float* Ts_final,
	bool get_flag, int* metric_map, int* metric_count,
	cudaStream_t stream)
{
	metricCount<16, 16>(P, num_rendered, width, height, splat_buffer,
				gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, 
				last_contributor, Ts_final,
				get_flag, metric_map, metric_count,
				stream);
    return 0;
}


} // namespace flashgs