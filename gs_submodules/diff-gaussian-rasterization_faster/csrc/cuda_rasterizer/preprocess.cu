#include "../ops.h"

// #ifndef CUDA_VERSION
// #define CUDA_VERSION 8000
// #endif

#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"
#include "rasterizer_imp.h"

namespace faster {
namespace {

constexpr float log2e = 1.4426950216293334961f;
constexpr float ln2 = 0.69314718055f;
__device__ int g_tile_cursor = 0;

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

// __forceinline__ __device__ void getRect(const float2 p, int width, int height,
//                                         int2 &rect_min, int2 &rect_max,
//                                         dim3 grid, int block_x, int block_y) {
//   rect_min = {
//       min((int)grid.x, max((int)0, (int)((p.x - width) / (float)block_x))),
//       min((int)grid.y, max((int)0, (int)((p.y - height) / (float)block_y)))};
//   rect_max = {
//       min((int)grid.x, max((int)0, (int)((p.x + width) / (float)block_x) + 1)),
//       min((int)grid.y,
//           max((int)0, (int)((p.y + height) / (float)block_y) + 1))};
// }

// Forward version of 2D covariance matrix computation
__forceinline__ __device__ float3 computeCov2D(const glm::vec3 &position,
                                               float focal_x, float focal_y,
                                               float tan_fovx, float tan_fovy,
                                               float *cov3D,
                                               float *viewmatrix) {
  // The following models the steps outlined by equations 29
  // and 31 in "EWA Splatting" (Zwicker et al., 2002).
  // Additionally considers aspect / scaling of viewport.
  // Transposes used to account for row-/column-major conventions.
  float3 t = transformPoint4x3(position, viewmatrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  glm::mat3 J =
      glm::mat3(focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z), 0.0f,
                focal_y / t.z, -(focal_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3((viewmatrix)[0], (viewmatrix)[4], (viewmatrix)[8],
                          (viewmatrix)[1], (viewmatrix)[5], (viewmatrix)[9],
                          (viewmatrix)[2], (viewmatrix)[6], (viewmatrix)[10]);

  glm::mat3 T = W * J;

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                            cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3f;
  cov[1][1] += 0.3f;
  return {float(cov[0][0]), float(cov[0][1]), float(cov[1][1])};
}

__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs,
                                        const glm::vec3 *means,
                                        glm::vec3 campos, const float *dc,
                                        const float *shs, float *clamped) {
  // The implementation is loosely based on code for
  // "Differentiable Point-Based Radiance Fields for
  // Efficient View Synthesis" by Zhang et al. (2022)
  glm::vec3 pos = means[idx];
  glm::vec3 dir = pos - campos;
  dir = dir / glm::length(dir);

  glm::vec3 *direct_color = ((glm::vec3 *)dc) + idx;
  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
  glm::vec3 result = SH_C0 * direct_color[0];

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;
      result = result + SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] +
               SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
               SH_C2[3] * xz * sh[6] + SH_C2[4] * (xx - yy) * sh[7];

      if (deg > 2) {
        result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
                 SH_C3[1] * xy * z * sh[9] +
                 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
                 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
                 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
                 SH_C3[5] * z * (xx - yy) * sh[13] +
                 SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
      }
    }
  }
  result += 0.5f;

  // RGB colors are clamped to positive values. If values are
  // clamped, we need to keep track of this for the backward pass.
  clamped[0] = (result.x < 0)? 1.f : 0.f;
  clamped[1] = (result.y < 0)? 1.f : 0.f;
  clamped[2] = (result.z < 0)? 1.f : 0.f;
  return glm::max(result, 0.0f);
}

__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b,
                                                          float c, float d,
                                                          float l, float r) {
  float delta = b * b - 4.0f * a * c;
  // return delta >= 0.0f && t1 <= sqrt(delta) && t2 >= -sqrt(delta)
  float t1 = (l - d) * (2.0f * a) + b;
  float t2 = (r - d) * (2.0f * a) + b;
  return delta >= 0.0f && (t1 <= 0.0f || t1 * t1 <= delta) &&
         (t2 >= 0.0f || t2 * t2 <= delta);
}

__forceinline__ __device__ bool
block_intersect_ellipse(int2 pix_min, int2 pix_max, float2 center, float3 conic,
                        float power) {
  float a, b, c, dx, dy;
  float w = 2.0f * power;

  if (center.x * 2.0f < pix_min.x + pix_max.x) {
    dx = center.x - pix_min.x;
  } else {
    dx = center.x - pix_max.x;
  }
  a = conic.z;
  b = -2.0f * conic.y * dx;
  c = conic.x * dx * dx - w;

  if (segment_intersect_ellipse(a, b, c, center.y, pix_min.y, pix_max.y)) {
    return true;
  }

  if (center.y * 2.0f < pix_min.y + pix_max.y) {
    dy = center.y - pix_min.y;
  } else {
    dy = center.y - pix_max.y;
  }
  a = conic.x;
  b = -2.0f * conic.y * dy;
  c = conic.z * dy * dy - w;

  if (segment_intersect_ellipse(a, b, c, center.x, pix_min.x, pix_max.x)) {
    return true;
  }

  return false;
}

__forceinline__ __device__ bool
block_contains_center(int2 pix_min, int2 pix_max, float2 center) {
  return center.x >= pix_min.x && center.x <= pix_max.x &&
         center.y >= pix_min.y && center.y <= pix_max.y;
}

__device__ __forceinline__ uint32_t fast_div_u32_1to32(uint32_t a, uint32_t b) {
  uint64_t res;
  switch (b) {
  case 1:
    return a;
  case 2:
    return a >> 1;
  case 3:
    res = (uint64_t)a * 0x55555556ull;
    return res >> 32;
  case 4:
    return a >> 2;
  case 5:
    res = (uint64_t)a * 0x33333334ull;
    return res >> 32;
  case 6:
    res = (uint64_t)a * 0x2AAAAAABull;
    return res >> 32;
  case 7:
    res = (uint64_t)a * 0x24924925ull;
    return res >> 32;
  case 8:
    return a >> 3;
  case 9:
    res = (uint64_t)a * 0x1C71C71Dull;
    return res >> 32;
  case 10:
    res = (uint64_t)a * 0x1999999Aull;
    return res >> 32;
  case 11:
    res = (uint64_t)a * 0x1745D174ull;
    return res >> 32;
  case 12:
    res = (uint64_t)a * 0x15555556ull;
    return res >> 32;
  case 13:
    res = (uint64_t)a * 0x13B13B14ull;
    return res >> 32;
  case 14:
    res = (uint64_t)a * 0x12492493ull;
    return res >> 32;
  case 15:
    res = (uint64_t)a * 0x11111112ull;
    return res >> 32;
  case 16:
    return a >> 4;
  case 17:
    res = (uint64_t)a * 0x0F0F0F10ull;
    return res >> 32;
  case 18:
    res = (uint64_t)a * 0x0E38E38Full;
    return res >> 32;
  case 19:
    res = (uint64_t)a * 0x0D79435Eull;
    return res >> 32;
  case 20:
    res = (uint64_t)a * 0x0CCCCCCDull;
    return res >> 32;
  case 21:
    res = (uint64_t)a * 0x0C30C30Dull;
    return res >> 32;
  case 22:
    res = (uint64_t)a * 0x0BA2E8BAull;
    return res >> 32;
  case 23:
    res = (uint64_t)a * 0x0B21642Cull;
    return res >> 32;
  case 24:
    res = (uint64_t)a * 0x0AAAAAABull;
    return res >> 32;
  case 25:
    res = (uint64_t)a * 0x0A3D70A4ull;
    return res >> 32;
  case 26:
    res = (uint64_t)a * 0x09D89D8Aull;
    return res >> 32;
  case 27:
    res = (uint64_t)a * 0x097B425Eull;
    return res >> 32;
  case 28:
    res = (uint64_t)a * 0x09249249ull;
    return res >> 32;
  case 29:
    res = (uint64_t)a * 0x08D3DCB1ull;
    return res >> 32;
  case 30:
    res = (uint64_t)a * 0x08888889ull;
    return res >> 32;
  case 31:
    res = (uint64_t)a * 0x08421085ull;
    return res >> 32;
  case 32:
    return a >> 5;
  }
  // If input guaranteed in [1,32], no default case is needed.
  return a / b; // For safety in release mode
}

__constant__ __device__ uint64_t MAGIC_TABLE[33] = {0, // unused, b = 0
                                                    0x100000001ull,
                                                    0x80000001ull,
                                                    0x55555556ull,
                                                    0x40000001ull,
                                                    0x33333334ull,
                                                    0x2AAAAAABull,
                                                    0x24924925ull,
                                                    0x20000001ull,
                                                    0x1C71C71Dull,
                                                    0x1999999Aull,
                                                    0x1745D174ull,
                                                    0x15555556ull,
                                                    0x13B13B14ull,
                                                    0x12492493ull,
                                                    0x11111112ull,
                                                    0x10000001ull,
                                                    0x0F0F0F10ull,
                                                    0x0E38E38Full,
                                                    0x0D79435Eull,
                                                    0x0CCCCCCDll,
                                                    0x0C30C30Dull,
                                                    0x0BA2E8BAull,
                                                    0x0B21642Cull,
                                                    0x0AAAAAABull,
                                                    0x0A3D70A4ull,
                                                    0x09D89D8Aull,
                                                    0x097B425Eull,
                                                    0x09249249ull,
                                                    0x08D3DCB1ull,
                                                    0x08888889ull,
                                                    0x08421085ull,
                                                    0x08000001ull};

__constant__ __device__ uint8_t SHIFT_TABLE[33] = {
    0,  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};

__device__ __forceinline__ uint32_t fast_div_lookup(uint32_t a, uint32_t b) {
  if (b > 32)
    return a / b;
  uint64_t magic = MAGIC_TABLE[b];
  uint8_t shift = SHIFT_TABLE[b];
  return (uint32_t)(((uint64_t)a * magic) >> shift);
}

__forceinline__ __device__ void computeCov3DCUDA(const glm::vec3 &scale,
                                                 float mod,
                                                 const glm::vec4 &rot,
                                                 float *cov3D) {
  // Create scaling matrix
  glm::mat3 S = glm::mat3(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;

  // Normalize quaternion to get valid rotation
  glm::vec4 q = rot; // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  // Compute rotation matrix from quaternion
  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                          2.f * (x * z + r * y), 2.f * (x * y + r * z),
                          1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x),
                          1.f - 2.f * (x * x + y * y));

  glm::mat3 M = S * R;

  // Compute 3D world covariance matrix Sigma
  glm::mat3 Sigma = glm::transpose(M) * M;

  // Covariance is symmetric, only store upper right
  cov3D[0] = Sigma[0][0];
  cov3D[1] = Sigma[0][1];
  cov3D[2] = Sigma[0][2];
  cov3D[3] = Sigma[1][1];
  cov3D[4] = Sigma[1][2];
  cov3D[5] = Sigma[2][2];
}

__global__ void calculateCov3D(int P, const glm::vec3 *__restrict__ scales,
                               const glm::vec4 *__restrict__ rotations,
                               float scale_modifier,
                               float *__restrict__ cov3Ds) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= P)
    return;
  glm::vec3 scale = scales[idx];
  glm::vec4 rot = rotations[idx];
  computeCov3DCUDA(scale, scale_modifier, rot, &cov3Ds[idx * 6]);
}

__global__ void preprocessSmallCUDA(
    int P, int deg, int max_coeffs, const glm::vec3 *__restrict__ positions,
    const float *__restrict__ opacities,
    const float *__restrict__ color_precomp, const float *__restrict__ dc,
    const float *__restrict__ shs, float *viewmatrix, float *projmatrix,
    const glm::vec3 *cam_position, const int W, const int H, const int block_x,
    const int block_y, const float tan_fovx, const float tan_fovy,
    const float focal_x, const float focal_y,
    const glm::vec3 *__restrict__ scales,
    const glm::vec4 *__restrict__ rotations, float scale_modifier,
    float *__restrict__ splat_buffer, int *__restrict__ curr_offset,
    uint64_t *__restrict__ gaussian_keys_unsorted,
    uint32_t *__restrict__ gaussian_values_unsorted,
    uint32_t *__restrict__ gaussian_values_sorted, const dim3 grid, int *radii,
    const float mult, const bool *__restrict__ culling) {
  int lane = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = blockIdx.x * blockDim.z + threadIdx.z;
  int idx_vec = warp_id * FLASHGS_WARP_SIZE + lane;

  // Initialize radius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  bool point_valid = false;
  glm::vec3 p_orig;
  int width = 0;
  int height = 0;
  float3 p_view;
  float2 point_xy;
  float3 conic;
  float opacity;
  float power;
  float log2_opacity;
  int2 rect_min;
  int2 rect_max;
  float my_radius = 0;
  int thread_offset = 0;
  int tile_count = 0;
  if (idx_vec < P && (culling == nullptr || !culling[idx_vec])) {
    do {
      // Perform near culling, quit if outside.
      radii[idx_vec] = 0;
      p_orig = positions[idx_vec];
      p_view = transformPoint4x3(p_orig, viewmatrix);
      if (p_view.z <= 0.2f)
        break;
      opacity = opacities[idx_vec];
      if (255.0f * opacity < 1.0f)
        break;

      // Transform point by projecting
      float4 p_hom = transformPoint4x4(p_orig, projmatrix);
      float p_w = 1.0f / (p_hom.w + 0.0000001f);
      float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

      const int cov3d_idx = COV3D + idx_vec * 32;
      computeCov3DCUDA(scales[idx_vec], scale_modifier, rotations[idx_vec],
                       splat_buffer + cov3d_idx);

      // Compute 2D screen-space covariance matrix
      float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy,
                                splat_buffer + cov3d_idx, viewmatrix);

      // Invert covariance (EWA algorithm)
      float det = (cov.x * cov.z - cov.y * cov.y);
      float det_inv = 1.f / det;
      conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

      float mid = 0.5f * (cov.x + cov.z);
      
      const float opacity_power_threshold = log(opacity * 255);
      const float extent = min(3.33, sqrt(2.0f * opacity_power_threshold));	
	    float lambda = mid + sqrt(max(0.01f, mid * mid - det));
      my_radius = ceil(extent * sqrt(lambda));
      if (my_radius < 0) break;

      log2_opacity = fast_lg2_f32(opacity);
      power = ln2 * 8.0f + ln2 * log2_opacity;
      power *= mult;
      width = (int)(1.414214f * fast_sqrt_f32(cov.x * power) + 1.0f);
      height = (int)(1.414214f * fast_sqrt_f32(cov.z * power) + 1.0f);

      point_xy = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
      getRect(point_xy, width, height, rect_min, rect_max, grid, block_x,
              block_y);
      tile_count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
      point_valid = tile_count > 0;
    } while (false);
  }

  bool vertex_valid = point_valid;
  point_valid = point_valid && tile_count < large_threshold;
  int multi_tiles = __ballot_sync(~0, point_valid);

  int local_idx = 0;
  int warp_tiles = 0;
  int lane_task = 0;
  float2 my_point_xy;
  float3 my_conic;
  int2 my_rect_min;
  int2 my_rect_max;
  int my_tile_count = 0;
  float my_depth = 0;
  float my_power;
  int idx;
  int tile_w;

  int i = __ffs(multi_tiles) - 1;
  float2 local_point_xy = {__shfl_sync(~0, point_xy.x, i),
                           __shfl_sync(~0, point_xy.y, i)};
  float3 local_conic = {
      __shfl_sync(~0, conic.x, i),
      __shfl_sync(~0, conic.y, i),
      __shfl_sync(~0, conic.z, i),
  };
  int2 local_rect_min = {__shfl_sync(~0, rect_min.x, i),
                         __shfl_sync(~0, rect_min.y, i)};
  int2 local_rect_max = {__shfl_sync(~0, rect_max.x, i),
                         __shfl_sync(~0, rect_max.y, i)};
  int local_tile_count = __shfl_sync(~0, tile_count, i);
  float local_depth = __shfl_sync(~0, p_view.z, i);
  float local_power = __shfl_sync(~0, power, i);

  bool tail_flag = false;
  while (multi_tiles) {
    tail_flag = false;

    while (warp_tiles < FLASHGS_WARP_SIZE && multi_tiles) {
      if (warp_tiles > lane && !tail_flag) {
        my_point_xy = local_point_xy;
        my_conic = local_conic;
        my_rect_min = local_rect_min;
        my_rect_max = local_rect_max;
        my_tile_count = local_tile_count;
        my_depth = local_depth;
        my_power = local_power;
        idx = warp_id * FLASHGS_WARP_SIZE + i;
        tile_w = my_rect_max.x - my_rect_min.x;

        lane_task = i;
        tail_flag = true;
        local_idx = my_tile_count - warp_tiles + lane;
      }

      i = __ffs(multi_tiles) - 1;
      multi_tiles &= multi_tiles - 1;

      local_point_xy = {__shfl_sync(~0, point_xy.x, i),
                        __shfl_sync(~0, point_xy.y, i)};
      local_conic = {
          __shfl_sync(~0, conic.x, i),
          __shfl_sync(~0, conic.y, i),
          __shfl_sync(~0, conic.z, i),
      };
      local_rect_min = {__shfl_sync(~0, rect_min.x, i),
                        __shfl_sync(~0, rect_min.y, i)};
      local_rect_max = {__shfl_sync(~0, rect_max.x, i),
                        __shfl_sync(~0, rect_max.y, i)};
      local_tile_count = __shfl_sync(~0, tile_count, i);
      local_depth = __shfl_sync(~0, p_view.z, i);
      local_power = __shfl_sync(~0, power, i);
      warp_tiles += local_tile_count;
    }

    if (warp_tiles > lane && !tail_flag) {
      my_point_xy = local_point_xy;
      my_conic = local_conic;
      my_rect_min = local_rect_min;
      my_rect_max = local_rect_max;
      my_tile_count = local_tile_count;
      my_depth = local_depth;
      my_power = local_power;
      idx = warp_id * FLASHGS_WARP_SIZE + i;
      tile_w = my_rect_max.x - my_rect_min.x;

      lane_task = i;
      tail_flag = true;
      local_idx = my_tile_count - warp_tiles + lane;
    }

    warp_tiles -= FLASHGS_WARP_SIZE;

    int tx = local_idx % tile_w;
    int ty = local_idx / tile_w;

    int x = my_rect_min.x + tx;
    int y = my_rect_min.y + ty;

    bool valid = tail_flag && y < my_rect_max.y && x < my_rect_max.x;
    if (valid) {
      int2 pix_min = {x * block_x, y * block_y};
      int2 pix_max = {pix_min.x + block_x - 1, pix_min.y + block_y - 1};
      valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
              block_intersect_ellipse(pix_min, pix_max, my_point_xy, my_conic,
                                      my_power);
    }
    int mask = __ballot_sync(~0, valid);
    if (mask == 0) {
      continue;
    }
    int my_offset;
    if (lane == 0) {
      my_offset = atomicAdd(curr_offset, __popc(mask));
    }
    // vertex_valid = vertex_valid || lane_task == lane;
    int count = __popc(mask & ((1 << lane) - 1));
    uint64_t key = y * grid.x + x;
    key <<= 32;
    key |= __float_as_uint(my_depth);
    my_offset = __shfl_sync(~0, my_offset, 0);
    if (valid) {
      gaussian_keys_unsorted[my_offset + count] = key;
      gaussian_values_unsorted[my_offset + count] = idx;
    }
  }
  if (tail_flag) {
    if (warp_tiles > 0) {
      local_idx = local_tile_count - warp_tiles;
      warp_tiles = ((warp_tiles + FLASHGS_WARP_SIZE - 1) / FLASHGS_WARP_SIZE) *
                   FLASHGS_WARP_SIZE;

      my_point_xy = local_point_xy;
      my_conic = local_conic;
      my_rect_min = local_rect_min;
      my_rect_max = local_rect_max;
      my_tile_count = local_tile_count;
      my_depth = local_depth;
      my_power = local_power;
      idx = warp_id * FLASHGS_WARP_SIZE + i;
      tile_w = my_rect_max.x - my_rect_min.x;

      // int tx = (local_idx + lane) % tile_w;
      // int ty = (local_idx + lane) / tile_w;

      int ty = fast_div_lookup(local_idx + lane, tile_w);
      int tx = local_idx + lane - tile_w * ty;

      int tile_stride = FLASHGS_WARP_SIZE;
      // int tx_stride = tile_stride % tile_w;
      // int ty_stride = tile_stride / tile_w;

      int ty_stride = fast_div_lookup(tile_stride, tile_w);
      int tx_stride = tile_stride - ty_stride * tile_w;

      for (int t = lane; t < warp_tiles; t += FLASHGS_WARP_SIZE) {
        int x = my_rect_min.x + tx;
        int y = my_rect_min.y + ty;

        bool valid = y < my_rect_max.y && x < my_rect_max.x;

        tx += tx_stride;
        ty += ty_stride;
        if (tx >= tile_w) {
          tx -= tile_w;
          ty += 1;
        }

        if (valid) {
          int2 pix_min = {x * block_x, y * block_y};
          int2 pix_max = {pix_min.x + block_x - 1, pix_min.y + block_y - 1};
          valid = block_contains_center(pix_min, pix_max, my_point_xy) ||
                  block_intersect_ellipse(pix_min, pix_max, my_point_xy,
                                          my_conic, my_power);
        }

        int mask = __ballot_sync(~0, valid);
        if (mask == 0) {
          continue;
        }
        int my_offset;
        if (lane == 0) {
          my_offset = atomicAdd(curr_offset, __popc(mask));
        }
        int count = __popc(mask & ((1 << lane) - 1));
        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= __float_as_uint(my_depth);
        my_offset = __shfl_sync(~0, my_offset, 0);
        if (valid) {
          gaussian_keys_unsorted[my_offset + count] = key;
          gaussian_values_unsorted[my_offset + count] = idx;
        }
      }
    }
  }

  if (vertex_valid) {
    glm::vec3 color;
    if (color_precomp != nullptr) {
      color = ((glm::vec3 *)color_precomp)[idx_vec];
    } else {
      color = computeColorFromSH(idx_vec, deg, max_coeffs,
                                 (const glm::vec3 *)positions, *cam_position,
                                 (const float *)dc, (const float *)shs,
                                 splat_buffer + idx_vec * 32 + CLAMPED);
    }
    float4 rgbd = {color.r, color.g, color.b, p_view.z};
    float4 con_o = {(-0.5f * log2e) * conic.x, -log2e * conic.y,
                    (-0.5f * log2e) * conic.z, log2_opacity};
    float tmp_buffer[32];
    float2 rect_min_float = {__int_as_float(rect_min.x),
                             __int_as_float(rect_min.y)};
    float2 rect_max_float = {__int_as_float(rect_max.x),
                             __int_as_float(rect_max.y)};
    store_to_buffer(tmp_buffer, point_xy, rgbd, con_o, rect_min_float,
                    rect_max_float, conic, p_view, power);
    write_tmp_to_global(tmp_buffer, splat_buffer, idx_vec, 21);
    
    radii[idx_vec] = my_radius;
  }

  if (vertex_valid && !point_valid) {
    int task_buff[3];
    int count =
        (tile_count + large_split_tile_count - 1) / large_split_tile_count;
    int offset = atomicAdd(&g_tile_cursor, count);
    int split_offset = 0;
    task_buff[0] = idx_vec;
    for (int i = 0; i < count - 1; i++) {
      task_buff[1] = split_offset;
      task_buff[2] = large_split_tile_count;
      split_offset += large_split_tile_count;
#pragma unroll
      for (int j = 0; j < 3; j++) {
        gaussian_values_sorted[(offset + i) * 3 + j] = task_buff[j];
      }
    }
    task_buff[1] = split_offset;
    task_buff[2] = tile_count - split_offset;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      gaussian_values_sorted[(offset + count - 1) * 3 + j] = task_buff[j];
    }
  }
}


__device__ void process_unit_large(
    int lane, int task, int off, int len, int block_x, int block_y,
    float *global_buffer, int *__restrict__ curr_offset,
    uint64_t *__restrict__ gaussian_keys_unsorted,
    uint32_t *__restrict__ gaussian_values_unsorted, const dim3 grid) {
  float3 p_view;
  float2 point_xy;
  float3 conic;
  float opacity;
  float power;
  float log2_opacity;
  int2 rect_min;
  int2 rect_max;
  float4 con_o;
  float4 rgbd;
  int tile_count;
  int tile_w;

  // 1
  // float buffer[32];
  // float2 rect_max_float, rect_min_float;
  // read_global_to_tmp(global_buffer, buffer, task, 21);
  // load_from_buffer(buffer, point_xy, rgbd, con_o, rect_min_float,
  // rect_max_float, conic, p_view, power); rect_min =
  // {__float_as_int(rect_min_float.x), __float_as_int(rect_min_float.y)};
  // rect_max = {__float_as_int(rect_max_float.x),
  // __float_as_int(rect_max_float.y)}; tile_count = (rect_max.x - rect_min.x) *
  // (rect_max.y - rect_min.y); tile_w = rect_max.x - rect_min.x;

  // 2
  float buffer[32];
  float2 rect_max_float, rect_min_float;
  if (lane == 0) {
    float buffer[32];
    float2 rect_max_float, rect_min_float;
    read_global_to_tmp(global_buffer, buffer, task, 21);
    load_from_buffer(buffer, point_xy, rgbd, con_o, rect_min_float,
                     rect_max_float, conic, p_view, power);
    rect_min = {__float_as_int(rect_min_float.x),
                __float_as_int(rect_min_float.y)};
    rect_max = {__float_as_int(rect_max_float.x),
                __float_as_int(rect_max_float.y)};
    tile_count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    tile_w = rect_max.x - rect_min.x;
  }

  point_xy = {__shfl_sync(~0, point_xy.x, 0), __shfl_sync(~0, point_xy.y, 0)};
  rgbd = {__shfl_sync(~0, rgbd.x, 0), __shfl_sync(~0, rgbd.y, 0),
          __shfl_sync(~0, rgbd.z, 0), __shfl_sync(~0, rgbd.w, 0)

  };
  con_o = {__shfl_sync(~0, con_o.x, 0), __shfl_sync(~0, con_o.y, 0),
           __shfl_sync(~0, con_o.z, 0), __shfl_sync(~0, con_o.w, 0)};
  rect_min = {__shfl_sync(~0, rect_min.x, 0), __shfl_sync(~0, rect_min.y, 0)};
  rect_max = {__shfl_sync(~0, rect_max.x, 0), __shfl_sync(~0, rect_max.y, 0)};
  conic = {__shfl_sync(~0, conic.x, 0), __shfl_sync(~0, conic.y, 0),
           __shfl_sync(~0, conic.z, 0)};
  p_view = {__shfl_sync(~0, p_view.x, 0), __shfl_sync(~0, p_view.y, 0),
            __shfl_sync(~0, p_view.z, 0)};
  power = __shfl_sync(~0, power, 0);
  tile_count = __shfl_sync(~0, tile_count, 0);
  tile_w = __shfl_sync(~0, tile_w, 0);

  float depth = p_view.z;
  int tx = (off + lane) % tile_w;
  int ty = (off + lane) / tile_w;

  int tile_stride = FLASHGS_WARP_SIZE;
  int tx_stride = tile_stride % tile_w;
  int ty_stride = tile_stride / tile_w;

  len = ((len + FLASHGS_WARP_SIZE - 1) / FLASHGS_WARP_SIZE) * FLASHGS_WARP_SIZE;

  for (int t = lane; t < len; t += tile_stride) {
    int x = rect_min.x + tx;
    int y = rect_min.y + ty;

    bool valid = y < rect_max.y && x < rect_max.x;

    tx += tx_stride;
    ty += ty_stride;
    if (tx >= tile_w) {
      tx -= tile_w;
      ty += 1;
    }

    if (valid) {
      int2 pix_min = {x * block_x, y * block_y};
      int2 pix_max = {pix_min.x + block_x - 1, pix_min.y + block_y - 1};
      valid = block_contains_center(pix_min, pix_max, point_xy) ||
              block_intersect_ellipse(pix_min, pix_max, point_xy, conic, power);
    }

    int mask = __ballot_sync(~0, valid);
    if (mask == 0) {
      continue;
    }
    int my_offset;
    if (lane == 0) {
      my_offset = atomicAdd(curr_offset, __popc(mask));
    }
    int count = __popc(mask & ((1 << lane) - 1));
    uint64_t key = y * grid.x + x;
    key <<= 32;
    key |= __float_as_uint(depth);
    my_offset = __shfl_sync(~0, my_offset, 0);
    if (valid) {
      gaussian_keys_unsorted[my_offset + count] = key;
      gaussian_values_unsorted[my_offset + count] = task;
    }
  }
}

__global__ void run_large_tiles_process_kernel(
    int block_x, int block_y, int num_tiles, float *global_buffer,
    int *__restrict__ curr_offset,
    uint64_t *__restrict__ gaussian_keys_unsorted,
    uint32_t *__restrict__ gaussian_values_unsorted,
    const uint32_t *__restrict__ gaussian_values_sorted, const dim3 grid) {
  int lane = threadIdx.y * blockDim.x + threadIdx.x;
  int task, off, len, my_tile;

  while (true) {
    if (lane == 0) {
      my_tile = atomicAdd(&g_tile_cursor, 1);
    }
    my_tile = __shfl_sync(~0, my_tile, 0);
    if (my_tile >= num_tiles)
      break;

    task = gaussian_values_sorted[my_tile * 3 + 0];
    off = gaussian_values_sorted[my_tile * 3 + 1];
    len = gaussian_values_sorted[my_tile * 3 + 2];
    // if (lane == 0) {
    //     task = gaussian_keys_sorted[my_tile * 3 + 0];
    //     off  = gaussian_keys_sorted[my_tile * 3 + 1];
    //     len  = gaussian_keys_sorted[my_tile * 3 + 2];
    // }
    // task = __shfl_sync(~0, task, 0);
    // off = __shfl_sync(~0, off, 0);
    // len = __shfl_sync(~0, len, 0);

    process_unit_large(lane, task, off, len, block_x, block_y, global_buffer,
                       curr_offset, gaussian_keys_unsorted,
                       gaussian_values_unsorted, grid);
  }
}


glm::mat4 getViewMatrix(glm::vec3 position, glm::mat3 rotation) {
  return glm::mat4(glm::vec4(rotation[0], 0.0f), glm::vec4(rotation[1], 0.0f),
                   glm::vec4(rotation[2], 0.0f),
                   glm::vec4(rotation * -position, 1.0f));
}

glm::mat4 getProjectionMatrix(int width, int height, glm::vec3 position,
                              glm::mat3 rotation, float focal_x, float focal_y,
                              float zFar, float zNear) {
  float top = height / (2.0f * focal_y) * zNear;
  float bottom = -top;
  float right = width / (2.0f * focal_x) * zNear;
  float left = -right;

  glm::mat4 P;
  memset(&P, 0, sizeof P);
  float z_sign = 1.0f;

  P[0][0] = 2.0f * zNear / (right - left);
  P[1][1] = 2.0f * zNear / (top - bottom);
  P[0][2] = (right + left) / (right - left);
  P[1][2] = (top + bottom) / (top - bottom);
  P[3][2] = z_sign;
  P[2][2] = z_sign * zFar / (zFar - zNear);
  P[2][3] = -(zFar * zNear) / (zFar - zNear);
  return glm::transpose(P) * getViewMatrix(position, rotation);
}

} // namespace

void preprocess(int P, int D, int max_coeffs, glm::vec3 *positions,
                float *color_precomp, float *dc, float *shs, float *opacities,
                float *scales, float *rotations, float scale_modifier,
                int width, int height, int block_x, int block_y,
                glm::vec3 *cam_position, float *view_matrix, float *proj_matrix,
                float tan_fovx, float tan_fovy, float zFar, float zNear,
                float *splat_buffer, uint64_t *gaussian_keys_unsorted,
                uint32_t *gaussian_values_unsorted,
                uint32_t *gaussian_values_sorted, int *curr_offset, int *radii,
                float mult, bool *culling, cudaStream_t stream) {
  dim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y,
            1);

  float focal_x = width / (2.0f * tan_fovx);
  float focal_y = height / (2.0f * tan_fovy);

  int zero = 0;
  cudaMemcpyToSymbol(g_tile_cursor, &zero, sizeof(int));
  preprocessSmallCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
      P, D, max_coeffs, positions, opacities, color_precomp, dc, shs,
      view_matrix, proj_matrix, cam_position, width, height, block_x, block_y,
      tan_fovx, tan_fovy, focal_x, focal_y, (glm::vec3 *)scales,
      (glm::vec4 *)rotations, scale_modifier, splat_buffer, curr_offset,
      gaussian_keys_unsorted, gaussian_values_unsorted, gaussian_values_sorted,
      grid, radii, mult, culling);

  int num_tiles = 0;
  cudaMemcpyFromSymbol(&num_tiles, g_tile_cursor, sizeof(int));
  cudaMemcpyToSymbol(g_tile_cursor, &zero, sizeof(int));

  if (num_tiles > 0){
    run_large_tiles_process_kernel<<<num_tiles, dim3(8, 4, 8)>>>(
          block_x, block_y, num_tiles, splat_buffer, curr_offset, gaussian_keys_unsorted,
          gaussian_values_unsorted, gaussian_values_sorted, grid);
  }
  
  // preprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
  //     P, D, max_coeffs, positions, opacities, color_precomp, dc, shs,
  //     view_matrix, proj_matrix, cam_position, width, height, block_x,
  //     block_y, tan_fovx, tan_fovy, focal_x, focal_y, points_xy, (glm::vec3
  //     *)scales, (glm::vec4 *)rotations, scale_modifier, cov3Ds, rgb_depth,
  //     clamped, conic_opacity, curr_offset, gaussian_keys_unsorted,
  //     gaussian_values_unsorted, grid, radii, mult);
}

} // namespace faster