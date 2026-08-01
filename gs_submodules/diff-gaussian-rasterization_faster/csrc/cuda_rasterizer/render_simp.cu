#include "../ops.h"
#include "rasterizer_imp.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace faster {
namespace {

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
}

__forceinline__ __device__ void pixel_shader(float &T, int2 pix, float2 xy,
                                             float4 con_o, int curr_idx,
                                             int &idx_max, float &weight_max,
                                             bool &flag, float &weight_sum,
                                             int &weight_count) {
  float2 d = {xy.x - (float)pix.x, xy.y - (float)pix.y};
  
  float power =
      con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
  float alpha;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
  alpha = min(0.99f, alpha);
  if (alpha < ONE_OF_255)
    return;

  const float weight = alpha * T;

  T -= weight;

  weight_sum += weight;
  weight_count += 1;

  if (weight_max < weight) {
    weight_max = weight;
    idx_max = curr_idx;
    flag = true;
  }
}

template <int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ void renderSimpCUDA(const int P, const uint2 *__restrict__ ranges,
                               const uint32_t *__restrict__ point_list,
                               int width, int height, int x_blocks,
                               render_load_info info,

                               float *__restrict__ accum_weights_p,
                               int *__restrict__ accum_weights_count,
                               float *__restrict__ accum_max_count) {
  uint2 range = ranges[blockIdx.y * x_blocks + blockIdx.x];
  int lane = threadIdx.y * blockDim.x + threadIdx.x;
  const void *data = info.data[lane];
  int lg2_scale = info.lg2_scale[lane];

  // uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y +
  // threadIdx.y };
  int2 pix[THREAD_Y][THREAD_X];
#pragma unroll
  for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_X; j++) {
      pix[i][j] = {(int)blockIdx.x * BLOCK_X + (int)threadIdx.x * THREAD_X + j,
                   (int)blockIdx.y * BLOCK_Y + (int)threadIdx.y * THREAD_Y + i};
    }
  }

  int idx_max[THREAD_Y][THREAD_X];
  float weight_max[THREAD_Y][THREAD_X];
  bool flag_update[THREAD_Y][THREAD_X];
#pragma unroll
  for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_X; j++) {
      if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1) {
        continue;
      }
      idx_max[i][j] = 0;
      weight_max[i][j] = 0;
      flag_update[i][j] = false;
    }
  }

  float T[THREAD_Y][THREAD_X];
#pragma unroll
  for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_X; j++) {
      T[i][j] = 1.0f;
    }
  }

  float local_weight_sum = 0.f;
  int local_weight_count = 0;
  int warp_accum_count = 0;
  float warp_accum_sum = 0.f;

  int to_do = range.y - range.x;
  int offset = range.x;
  bool done = false;
  float buf, ldg_buf;
  float2 xy;
  float3 rgb;
  float4 rgbd;
  float4 con_o;
  int point_id, ldg_point_id, curr_id;
  bool load_enable = data != nullptr;

  if (to_do % 2 != 0) {
    point_id = point_list[offset];
    const float *ptr =
        reinterpret_cast<const float *>(reinterpret_cast<const char *>(data) +
                                        ((int64_t)point_id << lg2_scale));
    bool load_enable = data != nullptr;
    if (load_enable)
      buf = __ldg(ptr);
    get_gaussian_features(xy, rgb, con_o, buf, 0);

    local_weight_sum = 0.f;
    local_weight_count = 0;
    warp_accum_count = 0;
    warp_accum_sum = 0.f;

#pragma unroll
    for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_X; j++) {
        if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1) {
          continue;
        }
        pixel_shader(T[i][j], pix[i][j], xy, con_o, point_id,
                     idx_max[i][j], weight_max[i][j], flag_update[i][j],
                     local_weight_sum, local_weight_count);
      }
    }

    warp_accum_sum = warp_sum_float(local_weight_sum);
    warp_accum_count = warp_sum_int(local_weight_count);
    if (lane == 0 && point_id < P) {
      atomicAdd(&accum_weights_p[point_id], warp_accum_sum);
      atomicAdd(&accum_weights_count[point_id], warp_accum_count);
    }

    to_do -= 1;
    offset += 1;
  }
  done = to_do == 0 ? true : false;
  offset = (lane & 4) == 0 ? offset : offset + 1;

  if (to_do > 0) {
    point_id = point_list[offset];
    const float *ptr =
        reinterpret_cast<const float *>(reinterpret_cast<const char *>(data) +
                                        ((int64_t)point_id << lg2_scale));
    if (load_enable)
      buf = __ldg(ptr);
    offset += 2;
    to_do -= 2;
  }

  while (__any_sync(~0, to_do >= 0 && !done)) {
    if (to_do > 0) {
      ldg_point_id = point_list[offset];
      const float *ptr =
          reinterpret_cast<const float *>(reinterpret_cast<const char *>(data) +
                                          ((int64_t)ldg_point_id << lg2_scale));
      if (load_enable)
        ldg_buf = __ldg(ptr);
    }

    get_gaussian_features(xy, rgb, con_o, buf, 0);
    curr_id = __shfl_sync(~0, point_id, 0);

    local_weight_sum = 0.f;
    local_weight_count = 0;
    warp_accum_count = 0;
    warp_accum_sum = 0.f;

#pragma unroll
    for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_X; j++) {
        if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1) {
          continue;
        }
        pixel_shader(T[i][j], pix[i][j], xy, con_o, curr_id, idx_max[i][j],
                     weight_max[i][j], flag_update[i][j], local_weight_sum,
                     local_weight_count);
      }
    }

    warp_accum_sum = warp_sum_float(local_weight_sum);
    warp_accum_count = warp_sum_int(local_weight_count);
    if (lane == 0 && curr_id < P) {
      atomicAdd(&accum_weights_p[curr_id], warp_accum_sum);
      atomicAdd(&accum_weights_count[curr_id], warp_accum_count);
    }

    get_gaussian_features(xy, rgb, con_o, buf, 4);
    curr_id = __shfl_sync(~0, point_id, 4);

    local_weight_sum = 0.f;
    local_weight_count = 0;
    warp_accum_count = 0;
    warp_accum_sum = 0.f;

#pragma unroll
    for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_X; j++) {
        if (pix[i][j].x > width - 1 || pix[i][j].y > height - 1) {
          continue;
        }
        pixel_shader(T[i][j], pix[i][j], xy, con_o, curr_id, idx_max[i][j],
                     weight_max[i][j], flag_update[i][j], local_weight_sum,
                     local_weight_count);
      }
    }

    warp_accum_sum = warp_sum_float(local_weight_sum);
    warp_accum_count = warp_sum_int(local_weight_count);
    if (lane == 0 && curr_id < P) {
      atomicAdd(&accum_weights_p[curr_id], warp_accum_sum);
      atomicAdd(&accum_weights_count[curr_id], warp_accum_count);
    }

    done = true;
#pragma unroll
    for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_X; j++) {
        done = done && T[i][j] < 0.0001f;
      }
    }

    to_do -= 2;
    offset += 2;
    buf = ldg_buf;
    point_id = ldg_point_id;
  }

  const int HxW = height * width;
#pragma unroll
  for (int i = 0; i < THREAD_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_X; j++) {
      if (pix[i][j].x >= width || pix[i][j].y >= height)
        continue;
      if (flag_update[i][j]) {
        atomicAdd(accum_max_count + idx_max[i][j], 1);
      }
    }
  }
}

template <int BLOCK_X, int BLOCK_Y>
void render_simp(int P, int num_rendered, int width, int height,
                 float *splat_buffer, uint64_t *gaussian_keys_sorted,
                 uint32_t *gaussian_values_sorted, uint2 *ranges,
                 float *accum_weights_ptr, int *accum_weights_count,
                 float *accum_max_count, cudaStream_t stream) {
  dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y,
            1);
  cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

  // Identify start and end of per-tile workloads in sorted list
  identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
      num_rendered, gaussian_keys_sorted, ranges);

  renderSimpCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4>
      <<<grid, dim3(8, 4, 1), 0, stream>>>(
          P, ranges, gaussian_values_sorted, width, height, grid.x,
          render_load_info(gaussian_values_sorted, splat_buffer),
          accum_weights_ptr, accum_weights_count, accum_max_count);
  CHECK_CUDA("render_simp");
}

} // namespace

uint32_t render_16x16_simp(int P, int num_rendered, int width, int height,
                           float *splat_buffer, uint64_t *gaussian_keys_sorted,
                           uint32_t *gaussian_values_sorted, uint2 *ranges,
                           float *accum_weights_ptr, int *accum_weights_count,
                           float *accum_max_count, cudaStream_t stream) {
  render_simp<16, 16>(P, num_rendered, width, height, splat_buffer,
                      gaussian_keys_sorted, gaussian_values_sorted, ranges,
                      accum_weights_ptr, accum_weights_count, accum_max_count,
                      stream);
  return 0;
}

} // namespace faster