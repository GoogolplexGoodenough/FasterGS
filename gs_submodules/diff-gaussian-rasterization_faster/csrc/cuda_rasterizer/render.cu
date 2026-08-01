#include "../ops.h"
#include "rasterizer_imp.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace cg = cooperative_groups;

namespace faster {
namespace {

__global__ void perTileBucketCount(int T, uint2 *ranges,
                                   uint32_t *bucketCount) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= T)
    return;

  uint2 range = ranges[idx];
  int num_splats = range.y - range.x;
  int num_buckets = (num_splats + BUCKET_SIZE - 1) / BUCKET_SIZE;
  bucketCount[idx] = (uint32_t)num_buckets;
}

__global__ void
perBucketRange(int T, const uint2 *__restrict__ ranges,
               const uint32_t *__restrict__ bucketCount,
               const uint32_t *__restrict__ per_tile_bucket_offset,
               uint2 *__restrict__ bucket_ranges,
               uint32_t *__restrict__ bucket_to_tile) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= T)
    return;

  uint2 range = ranges[idx];
  uint32_t bbm = idx == 0 ? 0 : per_tile_bucket_offset[idx - 1];
  int num_splats = range.y - range.x;
  int offset = 0;

  while (num_splats > 0) {
    offset = num_splats > BUCKET_SIZE ? BUCKET_SIZE : num_splats;
    bucket_ranges[bbm].x = range.x;
    bucket_ranges[bbm].y = range.x + offset;
    bucket_to_tile[bbm] = idx;
    range.x += offset;
    num_splats -= offset;
    bbm += 1;
  }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys,
                                   uint2 *ranges) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L)
    return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key = point_list_keys[idx];
  uint32_t currtile = key >> 32; // 32
  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32; // 32
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1) {
    ranges[currtile].y = L;
  }
  // if (ranges[idx].y - ranges[idx].x != 0){
  // 	printf("ranges: idx %d, x %d, y %d\n", idx, ranges[idx].x,
  // ranges[idx].y);
  // }
}

__global__ void show_ranges(int L, const int2 *ranges) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("blockIdx.x %d, blockDim.x %d, theardIdx.x %d\n", blockIdx.x,
  // blockDim.x, threadIdx.x);
  if (idx >= L)
    return;
  if (ranges[idx].y - ranges[idx].x != 0) {
    printf("ranges: idx %d, x %d, y %d\n", idx, ranges[idx].x, ranges[idx].y);
  }
  // printf("ranges: idx %d, x %d, y %d\n", idx, ranges[idx].x, ranges[idx].y);
}

__forceinline__ __device__ float fast_ex2_ftz_f32(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__forceinline__ __device__ void pixel_shader(float3 &C, float &T, int2 pix,
                                             float2 xy, float4 con_o,
                                             float3 rgb) {
  float2 d = {xy.x - (float)pix.x, xy.y - (float)pix.y};
  // float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y
  // * d.x * d.y;
  float power =
      con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
  float alpha;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
  alpha = min(0.99f, alpha);
  if (alpha < ONE_OF_255) alpha = 0.f;
  
  C.x += rgb.x * (alpha * T);
  C.y += rgb.y * (alpha * T);
  C.z += rgb.z * (alpha * T);
  T -= alpha * T;

  // if (alpha > ONE_OF_255){
  //   C.x += rgb.x * (alpha * T);
  //   C.y += rgb.y * (alpha * T);
  //   C.z += rgb.z * (alpha * T);
  //   T -= alpha * T;
  // }
}

__forceinline__ __device__ uint8_t encode(float x) {
  return (uint8_t)min(max(0.0f, x * 255.0f), 255.0f);
}


template <int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ void
renderBucket(uint32_t bucket_num, int width, int height, render_load_info info,
             const uint32_t *__restrict__ point_list,

             // const uint32_t* __restrict__ per_tile_bucket_offset,
             const uint2 *__restrict__ bucket_ranges,
             const uint32_t *__restrict__ bucket_to_tile,

             float *__restrict__ sampled_T, float4 *__restrict__ sampled_ar) {
	int idx = blockIdx.x * blockDim.z + threadIdx.z;
	if (idx >= bucket_num) return;
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	int BLOCK_SIZE = BLOCK_X * BLOCK_Y;
	
	uint2 range = bucket_ranges[idx];
	int tile_id = bucket_to_tile[idx];
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];

	int horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	const int2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
	const int2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};
	int2 pix[THREAD_Y][THREAD_X];
	float T[THREAD_Y][THREAD_X];
	float3 C[THREAD_Y][THREAD_X];

	// bool debug = (tile.x == horizontal_blocks / 2) && (tile.y  == horizontal_blocks / 2);
	// bool debug = tile.x == 13 && tile.y == 10 && lane == 0;

#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				pix_min.x + (int)threadIdx.x * THREAD_X + j,
				pix_min.y + (int)threadIdx.y * THREAD_Y + i
			};
			// if (debug){
			// 	printf(
			// 		"i %d, j %d, pix[i][j]: %d %d, pix_min %d %d, tile_id %d, tile %d %d\n", 
			// 		i, j, pix[i][j].x, pix[i][j].y, pix_min.x, pix_min.y, tile_id, tile.x, tile.y
			// 	);
			// }
			T[i][j] = 1.0f;
			C[i][j] = { 0.0f, 0.0f, 0.0f };
		}
	}
	int point_id;
	float buf, ldg_buf;
	float2 xy;
	float3 rgb;
	float4 rgbd, con_o;
	bool load_enable = data != nullptr;
	int to_do = range.y - range.x;
	int offset = range.x;

	if (to_do % 2 != 0) {
		point_id = point_list[offset];
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)point_id << lg2_scale));
		bool load_enable = data != nullptr;
		if (load_enable) buf = __ldg(ptr);
		get_gaussian_features(xy, rgb, con_o, buf, 0);

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				pixel_shader(C[i][j], T[i][j], pix[i][j], xy, con_o, rgb);
			}
		}
		to_do -= 1;
		offset += 1;
	}

	offset = (lane & 4) == 0? offset: offset + 1;
	bool done = to_do == 0? true: false;

	if (to_do > 0){
		point_id = point_list[offset];
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)point_id << lg2_scale));
		if (load_enable) buf = __ldg(ptr);
		offset += 2;
		to_do -= 2;
	}

	while(__any_sync(~0, to_do >= 0 && !done)){
		if (to_do > 0){
			point_id = point_list[offset];
			const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)point_id << lg2_scale));
			if (load_enable) ldg_buf = __ldg(ptr);
		}

		get_gaussian_features(xy, rgb, con_o, buf, 0);

#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				pixel_shader(
					C[i][j], T[i][j], pix[i][j], xy, con_o, rgb
				);
			}
		}
		
		get_gaussian_features(xy, rgb, con_o, buf, 4);
		
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				pixel_shader(
					C[i][j], T[i][j], pix[i][j], xy, con_o, rgb
				);
			}
		}

		to_do -= 2;
		offset += 2;
		buf = ldg_buf;
	}
	
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			if (pix[i][j].x >= width || pix[i][j].y >= height) continue;
			int pix_idx = (int)threadIdx.x * THREAD_X + j + ((int)threadIdx.y * THREAD_Y + i) * BLOCK_X;
			int sample_idx = idx * BLOCK_SIZE + pix_idx;
			sampled_T[sample_idx] = T[i][j];
			sampled_ar[sample_idx].x = C[i][j].x;
			sampled_ar[sample_idx].y = C[i][j].y;
			sampled_ar[sample_idx].z = C[i][j].z;
		}
	}
}

__device__ __forceinline__ float3 ldg_float3_from_float4(const float4 *p) {
  float3 v;
  v.x = __ldg(&p->x);
  v.y = __ldg(&p->y);
  v.z = __ldg(&p->z);
  return v;
}

template <int BLOCK_X, int BLOCK_Y>
__global__ void mergeResults(
    int width, int height, const uint32_t *__restrict__ per_tile_bucket_offset,
    uint2 *__restrict__ bucket_ranges, const float *__restrict__ sampled_T,
    const float4 *__restrict__ sampled_ar, float *__restrict__ accum_T,
    float4 *__restrict__ accum_ar, const float3 *__restrict__ bg_color,
    float *__restrict__ out_color, float3 *__restrict__ pixel_color,
    int *__restrict__ last_contributor, float *__restrict__ Ts_final,
    uint32_t *__restrict__ max_contrib) {
  const float3 bg = *bg_color;
  // int lane = threadIdx.y * blockDim.x + threadIdx.x;
  int2 pix = {(int)threadIdx.x, (int)threadIdx.y};
  int pix_id = pix.y * BLOCK_X + pix.x;
  int2 pix_min = {(int)blockIdx.x * BLOCK_X, (int)blockIdx.y * BLOCK_Y};
  if (pix_min.x + pix.x >= width || pix_min.y + pix.y >= height)
    return;
  int global_pix_id = pix_min.x + pix.x + (pix_min.y + pix.y) * width;

  int horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
  int tile_id = (int)blockIdx.y * horizontal_blocks + (int)blockIdx.x;

  uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
  int to_do = (int)per_tile_bucket_offset[tile_id] - bbm;

  int BLOCK_SIZE = BLOCK_X * BLOCK_Y;
  const float *local_T = sampled_T + bbm * BLOCK_SIZE + pix_id;
  const float4 *local_ar = sampled_ar + bbm * BLOCK_SIZE + pix_id;

  float *local_accum_T = accum_T + bbm * BLOCK_SIZE + pix_id;
  float4 *local_accum_ar = accum_ar + bbm * BLOCK_SIZE + pix_id;
  uint2 *local_ranges = bucket_ranges + bbm;

  float T = 1.f;
  float3 C = {0.f, 0.f, 0.f};

  float buf_T, ldg_buf_T;
  float3 buf_ar, ldg_buf_ar;
  int offset = 0;
  int contrib = 0;

  buf_T = __ldg(local_T);
  buf_ar = ldg_float3_from_float4(local_ar);
  offset += 1;
  to_do -= 1;
  while (to_do >= 0) {
    const float *ptr_T = local_T + offset * BLOCK_SIZE;
    const float4 *ptr_ar = local_ar + offset * BLOCK_SIZE;

    if (to_do > 0) {
      ldg_buf_T = __ldg(ptr_T);
      ldg_buf_ar = ldg_float3_from_float4(ptr_ar);
    }

    C.x += T * buf_ar.x;
    C.y += T * buf_ar.y;
    C.z += T * buf_ar.z;

    T *= buf_T;

    local_accum_ar->x = C.x;
    local_accum_ar->y = C.y;
    local_accum_ar->z = C.z;
    *local_accum_T = T;

    contrib += local_ranges->y - local_ranges->x;

    local_accum_ar += BLOCK_SIZE;
    local_accum_T += BLOCK_SIZE;
    local_ranges += 1;

    if (T < 0.0001f)
      break;

    buf_T = ldg_buf_T;
    buf_ar = ldg_buf_ar;
    offset += 1;
    to_do -= 1;
  }

  last_contributor[global_pix_id] = contrib;
  Ts_final[global_pix_id] = T;

  // float3* h, w, 3
  // out_color[global_pix_id].x = C.x + T * bg.x;
  // out_color[global_pix_id].y = C.y + T * bg.y;
  // out_color[global_pix_id].z = C.z + T * bg.z;

  // float* 3, h, w
  const int HxW = height * width;
  out_color[0 * HxW + global_pix_id] = C.x + T * bg.x;
  out_color[1 * HxW + global_pix_id] = C.y + T * bg.y;
  out_color[2 * HxW + global_pix_id] = C.z + T * bg.z;

  pixel_color[global_pix_id].x = C.x;
  pixel_color[global_pix_id].y = C.y;
  pixel_color[global_pix_id].z = C.z;

  typedef cub::BlockReduce<int, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                           BLOCK_Y>
      BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  uint32_t max_c =
      BlockReduce(temp_storage).Reduce(contrib, cub::Max());
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    max_contrib[tile_id] = max_c;
  }
}

template <int BLOCK_X, int BLOCK_Y>
uint32_t
render(int P, int num_rendered, int width, int height, float *splat_buffer, 
       uint64_t *gaussian_keys_sorted,
       uint32_t *gaussian_values_sorted, uint2 *ranges, float3 *bg_color,
       float *out_color, int *last_contributor, float *Ts_final, 
       
       std::function<char *(size_t)> img_func,
       std::function<char *(size_t)> spl_func,

       cudaStream_t stream) {
  dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y,
            1);
  cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

  int num_tiles = grid.x * grid.y;

  // Identify start and end of per-tile workloads in sorted list
  identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
      num_rendered, gaussian_keys_sorted, ranges);

  size_t img_chunk_size = required<ImageState>(width * height);
  char *img_chunkptr = img_func(img_chunk_size);
  ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

  perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, ranges,
                                                       imgState.bucket_count);
  cub::DeviceScan::InclusiveSum(
      imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size,
      imgState.bucket_count, imgState.bucket_offsets, num_tiles);
  uint32_t bucket_sum;
  cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1,
             sizeof(unsigned int), cudaMemcpyDeviceToHost);
  // create a state to store. size is number is the total number of buckets *
  // block_size
  size_t sample_chunk_size = required<SampleState>(bucket_sum);
  char *sample_chunkptr = spl_func(sample_chunk_size);
  SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);

  perBucketRange<<<(num_tiles + 255) / 256, 256>>>(
      num_tiles, ranges, imgState.bucket_count, imgState.bucket_offsets,
      sampleState.bucket_ranges, sampleState.bucket_to_tile);

  // renderBucket<16, 16, 2, 4> <<<(bucket_sum + 7) / 8, dim3(8, 4, 8), 0,
  // stream>>>(
  renderBucket<16, 16, 2, 4><<<bucket_sum, dim3(8, 4, 1), 0, stream>>>(
      bucket_sum, width, height,
      render_load_info(gaussian_values_sorted, splat_buffer),
      gaussian_values_sorted, sampleState.bucket_ranges,
      sampleState.bucket_to_tile, sampleState.T, sampleState.ar);

  mergeResults<16, 16><<<grid, dim3(16, 16, 1), 0, stream>>>(
      width, height, imgState.bucket_offsets, sampleState.bucket_ranges,
      sampleState.T, sampleState.ar, sampleState.accum_T, sampleState.accum_ar,
      bg_color, out_color, imgState.pixel_colors, last_contributor, Ts_final,
      imgState.max_contrib);

  return bucket_sum;
}

} // namespace

uint32_t render_16x16(int P, int num_rendered, int width, int height,
                      float *splat_buffer, uint64_t *gaussian_keys_sorted,
                      uint32_t *gaussian_values_sorted, uint2 *ranges,
                      float3 *bg_color, float *out_color,
                      int *last_contributor, float *Ts_final,

                      std::function<char *(size_t)> img_func,
                      std::function<char *(size_t)> smp_func,
                      cudaStream_t stream) {
  uint32_t bucket_sum = render<16, 16>(
      P, num_rendered, width, height, splat_buffer,
      gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color,
      last_contributor, Ts_final, 
      img_func, smp_func, stream);
  return bucket_sum;
}

} // namespace flashgs