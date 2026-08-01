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

 #include "../ops.h"
 #include "rasterizer_imp.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 
 namespace cg = cooperative_groups;
 
 #define BLOCK_X 16
 #define BLOCK_Y 16
 #define BLOCK_SIZE 256
 #define NUM_CHAFFELS 3
 
 namespace faster {
 namespace {
 
 __forceinline__ __device__ void getRect(const float2 p, int2 ext_rect,
                                         uint2 &rect_min, uint2 &rect_max,
                                         dim3 grid) {
   rect_min = {min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
               min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))};
   rect_max = {
       min(grid.x,
           max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
       min(grid.y,
           max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))};
 }
 
 __forceinline__ __device__ float3 transformPoint4x3(const float3 &p,
                                                     const float *matrix) {
   float3 transformed = {
       matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
       matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
       matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
   };
   return transformed;
 }
 
 __forceinline__ __device__ float4 transformPoint4x4(const float3 &p,
                                                     const float *matrix) {
   float4 transformed = {
       matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
       matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
       matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
       matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
   return transformed;
 }
 
 __forceinline__ __device__ float3 transformVec4x3(const float3 &p,
                                                   const float *matrix) {
   float3 transformed = {
       matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
       matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
       matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
   };
   return transformed;
 }
 
 __forceinline__ __device__ float3
 transformVec4x3Transpose(const float3 &p, const float *matrix) {
   float3 transformed = {
       matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
       matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
       matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
   };
   return transformed;
 }
 
 __forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
   float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
   float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
   float dnormvdz =
       (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
       invsum32;
   return dnormvdz;
 }
 
 __forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
   float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
   float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
 
   float3 dnormvdv;
   dnormvdv.x =
       ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) *
       invsum32;
   dnormvdv.y =
       (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) *
       invsum32;
   dnormvdv.z =
       (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
       invsum32;
   return dnormvdv;
 }
 
 __forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
   float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
   float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
 
   float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
   float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
   float4 dnormvdv;
   dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
   dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
   dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
   dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
   return dnormvdv;
 }
 
 // Backward pass for conversion of spherical harmonics to RGB for
 // each Gaussian.
 __device__ void computeColorFromSH(
     int idx, int deg, int max_coeffs, const glm::vec3 *__restrict__ means,
     glm::vec3 campos, const float *__restrict__ dc,
     const float *__restrict__ shs,
     //  const bool *clamped,
     const float *__restrict__ splat_buffer,
     const glm::vec3 *__restrict__ dL_dcolor, glm::vec3 *__restrict__ dL_dmeans,
     glm::vec3 *__restrict__ dL_ddc, glm::vec3 *__restrict__ dL_dshs) {
   // Compute intermediate values, as it is done during forward
   glm::vec3 pos = means[idx];
   glm::vec3 dir_orig = pos - campos;
   glm::vec3 dir = dir_orig / glm::length(dir_orig);
 
   glm::vec3 *direct_color = ((glm::vec3 *)dc) + idx;
   glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
 
   // Use PyTorch rule for clamping: if clamping was applied,
   // gradient becomes 0.
   glm::vec3 dL_dRGB = dL_dcolor[idx];
   const float* clamped = splat_buffer + idx * 32 + CLAMPED;
   dL_dRGB.x *= (1.f - clamped[0]);
   dL_dRGB.y *= (1.f - clamped[1]);
   dL_dRGB.z *= (1.f - clamped[2]);
   
   glm::vec3 dRGBdx(0, 0, 0);
   glm::vec3 dRGBdy(0, 0, 0);
   glm::vec3 dRGBdz(0, 0, 0);
   float x = dir.x;
   float y = dir.y;
   float z = dir.z;
 
   // Target location for this Gaussian to write SH gradients to
   glm::vec3 *dL_ddirect_color = dL_ddc + idx;
   glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;
 
   // No tricks here, just high school-level calculus.
   float dRGBdsh0 = SH_C0;
   dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
   if (deg > 0) {
     float dRGBdsh1 = -SH_C1 * y;
     float dRGBdsh2 = SH_C1 * z;
     float dRGBdsh3 = -SH_C1 * x;
     dL_dsh[0] = dRGBdsh1 * dL_dRGB;
     dL_dsh[1] = dRGBdsh2 * dL_dRGB;
     dL_dsh[2] = dRGBdsh3 * dL_dRGB;
 
     dRGBdx = -SH_C1 * sh[2];
     dRGBdy = -SH_C1 * sh[0];
     dRGBdz = SH_C1 * sh[1];
 
     if (deg > 1) {
       float xx = x * x, yy = y * y, zz = z * z;
       float xy = x * y, yz = y * z, xz = x * z;
 
       float dRGBdsh4 = SH_C2[0] * xy;
       float dRGBdsh5 = SH_C2[1] * yz;
       float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
       float dRGBdsh7 = SH_C2[3] * xz;
       float dRGBdsh8 = SH_C2[4] * (xx - yy);
       dL_dsh[3] = dRGBdsh4 * dL_dRGB;
       dL_dsh[4] = dRGBdsh5 * dL_dRGB;
       dL_dsh[5] = dRGBdsh6 * dL_dRGB;
       dL_dsh[6] = dRGBdsh7 * dL_dRGB;
       dL_dsh[7] = dRGBdsh8 * dL_dRGB;
 
       dRGBdx += SH_C2[0] * y * sh[3] + SH_C2[2] * 2.f * -x * sh[5] +
                 SH_C2[3] * z * sh[6] + SH_C2[4] * 2.f * x * sh[7];
       dRGBdy += SH_C2[0] * x * sh[3] + SH_C2[1] * z * sh[4] +
                 SH_C2[2] * 2.f * -y * sh[5] + SH_C2[4] * 2.f * -y * sh[7];
       dRGBdz += SH_C2[1] * y * sh[4] + SH_C2[2] * 2.f * 2.f * z * sh[5] +
                 SH_C2[3] * x * sh[6];
 
       if (deg > 2) {
         float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
         float dRGBdsh10 = SH_C3[1] * xy * z;
         float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
         float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
         float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
         float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
         float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
         dL_dsh[8] = dRGBdsh9 * dL_dRGB;
         dL_dsh[9] = dRGBdsh10 * dL_dRGB;
         dL_dsh[10] = dRGBdsh11 * dL_dRGB;
         dL_dsh[11] = dRGBdsh12 * dL_dRGB;
         dL_dsh[12] = dRGBdsh13 * dL_dRGB;
         dL_dsh[13] = dRGBdsh14 * dL_dRGB;
         dL_dsh[14] = dRGBdsh15 * dL_dRGB;
 
         dRGBdx += (SH_C3[0] * sh[8] * 3.f * 2.f * xy + SH_C3[1] * sh[9] * yz +
                    SH_C3[2] * sh[10] * -2.f * xy +
                    SH_C3[3] * sh[11] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[12] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[13] * 2.f * xz +
                    SH_C3[6] * sh[14] * 3.f * (xx - yy));
 
         dRGBdy +=
             (SH_C3[0] * sh[8] * 3.f * (xx - yy) + SH_C3[1] * sh[9] * xz +
              SH_C3[2] * sh[10] * (-3.f * yy + 4.f * zz - xx) +
              SH_C3[3] * sh[11] * -3.f * 2.f * yz +
              SH_C3[4] * sh[12] * -2.f * xy + SH_C3[5] * sh[13] * -2.f * yz +
              SH_C3[6] * sh[14] * -3.f * 2.f * xy);
 
         dRGBdz += (SH_C3[1] * sh[9] * xy + SH_C3[2] * sh[10] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[11] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[12] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[13] * (xx - yy));
       }
     }
   }
 
   // The view direction is an input to the computation. View direction
   // is influenced by the Gaussian's mean, so SHs gradients
   // must propagate back into 3D position.
   glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB),
                     glm::dot(dRGBdz, dL_dRGB));
 
   // Account for normalization of direction
   float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z},
                              float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});
 
   // Gradients of loss w.r.t. Gaussian means, but only the portion
   // that is caused because the mean affects the view-dependent color.
   // Additional mean gradient is accumulated in below methods.
   dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
 }
 
 // Backward version of INVERSE 2D covariance matrix computation
 // (due to length launched as separate kernel before other
 // backward steps contained in preprocess)
 __global__ void computeCov2DCUDA(int P, const float3 *means, const int *radii,
                                  const float *splat_buffer, const float h_x,
                                  float h_y, const float tan_fovx,
                                  float tan_fovy, const float *view_matrix,
                                  const float *dL_dconics, float3 *dL_dmeans,
                                  float *dL_dcov) {
   auto idx = cg::this_grid().thread_rank();
   if (idx >= P || !(radii[idx] > 0))
     return;
 
   // Reading location of 3D covariance for this Gaussian
   // const float *cov3D = cov3Ds + 6 * idx;
   const float *cov3D = splat_buffer + 32 * idx + COV3D;
 
   // Fetch gradients, recompute 2D covariance and relevant
   // intermediate forward results needed in the backward.
   float3 mean = means[idx];
   float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1],
                       dL_dconics[4 * idx + 3]};
   float3 t = transformPoint4x3(mean, view_matrix);
   // printf("idx %d, dL_dconic: %f %f %f\n", idx,  dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3]);
 
 
   const float limx = 1.3f * tan_fovx;
   const float limy = 1.3f * tan_fovy;
   const float txtz = t.x / t.z;
   const float tytz = t.y / t.z;
   t.x = min(limx, max(-limx, txtz)) * t.z;
   t.y = min(limy, max(-limy, tytz)) * t.z;
 
   const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
   const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;
 
   glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z), 0.0f,
                           h_y / t.z, -(h_y * t.y) / (t.z * t.z), 0, 0, 0);
 
   glm::mat3 W = glm::mat3(view_matrix[0], view_matrix[4], view_matrix[8],
                           view_matrix[1], view_matrix[5], view_matrix[9],
                           view_matrix[2], view_matrix[6], view_matrix[10]);
 
   glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                             cov3D[4], cov3D[2], cov3D[4], cov3D[5]);
 
   glm::mat3 T = W * J;
 
   glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
 
   // Use helper variables for 2D covariance entries. More compact.
   float a = cov2D[0][0] += 0.3f;
   float b = cov2D[0][1];
   float c = cov2D[1][1] += 0.3f;
 
   float denom = a * c - b * b;
   float dL_da = 0, dL_db = 0, dL_dc = 0;
   float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
 
   if (denom2inv != 0) {
     // Gradients of loss w.r.t. entries of 2D covariance matrix,
     // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
     // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
     dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y +
                          (denom - a * c) * dL_dconic.z);
     dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y +
                          (denom - a * c) * dL_dconic.x);
     dL_db = denom2inv * 2 *
             (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y +
              a * b * dL_dconic.z);
 
     // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
     // given gradients w.r.t. 2D covariance matrix (diagonal).
     // cov2D = transpose(T) * transpose(Vrk) * T;
     dL_dcov[6 * idx + 0] =
         (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db +
          T[1][0] * T[1][0] * dL_dc);
     dL_dcov[6 * idx + 3] =
         (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db +
          T[1][1] * T[1][1] * dL_dc);
     dL_dcov[6 * idx + 5] =
         (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db +
          T[1][2] * T[1][2] * dL_dc);
 
     // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
     // given gradients w.r.t. 2D covariance matrix (off-diagonal).
     // Off-diagonal elements appear twice --> double the gradient.
     // cov2D = transpose(T) * transpose(Vrk) * T;
     dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da +
                            (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db +
                            2 * T[1][0] * T[1][1] * dL_dc;
     dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da +
                            (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db +
                            2 * T[1][0] * T[1][2] * dL_dc;
     dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da +
                            (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db +
                            2 * T[1][1] * T[1][2] * dL_dc;
   } else {
     for (int i = 0; i < 6; i++)
       dL_dcov[6 * idx + i] = 0;
   }
 
   // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
   // cov2D = transpose(T) * transpose(Vrk) * T;
   float dL_dT00 =
       2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) *
           dL_da +
       (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
   float dL_dT01 =
       2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) *
           dL_da +
       (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
   float dL_dT02 =
       2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) *
           dL_da +
       (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
   float dL_dT10 =
       2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) *
           dL_dc +
       (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
   float dL_dT11 =
       2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) *
           dL_dc +
       (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
   float dL_dT12 =
       2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) *
           dL_dc +
       (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;
 
   // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
   // T = W * J
   float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
   float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
   float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
   float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;
 
   float tz = 1.f / t.z;
   float tz2 = tz * tz;
   float tz3 = tz2 * tz;
 
   // Gradients of loss w.r.t. transformed Gaussian mean t
   float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
   float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
   float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 +
                  (2 * h_x * t.x) * tz3 * dL_dJ02 +
                  (2 * h_y * t.y) * tz3 * dL_dJ12;
 
   // Account for transformation of mean to t
   // t = transformPoint4x3(mean, view_matrix);
   float3 dL_dmean =
       transformVec4x3Transpose({dL_dtx, dL_dty, dL_dtz}, view_matrix);
 
   // Gradients of loss w.r.t. Gaussian means, but only the portion
   // that is caused because the mean affects the covariance matrix.
   // Additional mean gradient is accumulated in BACKWARD::preprocess.
   dL_dmeans[idx] = dL_dmean;
 }
 
 // Backward pass for the conversion of scale and rotation to a
 // 3D covariance matrix for each Gaussian.
 __device__ void computeCov3D(int idx, const glm::vec3 scale, float mod,
                              const glm::vec4 rot, const float *dL_dcov3Ds,
                              glm::vec3 *dL_dscales, glm::vec4 *dL_drots) {
   // Recompute (intermediate) results for the 3D covariance computation.
   glm::vec4 q = rot; // / glm::length(rot);
   float r = q.x;
   float x = q.y;
   float y = q.z;
   float z = q.w;
 
   glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                           2.f * (x * z + r * y), 2.f * (x * y + r * z),
                           1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                           2.f * (x * z - r * y), 2.f * (y * z + r * x),
                           1.f - 2.f * (x * x + y * y));
 
   glm::mat3 S = glm::mat3(1.0f);
 
   glm::vec3 s = mod * scale;
   S[0][0] = s.x;
   S[1][1] = s.y;
   S[2][2] = s.z;
 
   glm::mat3 M = S * R;
 
   const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;
 
   glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
   glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);
 
   // Convert per-element covariance loss gradients to matrix form
   glm::mat3 dL_dSigma =
       glm::mat3(dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
                 0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
                 0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);
 
   // Compute loss gradient w.r.t. matrix M
   // dSigma_dM = 2 * M
   glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
 
   glm::mat3 Rt = glm::transpose(R);
   glm::mat3 dL_dMt = glm::transpose(dL_dM);
 
   // Gradients of loss w.r.t. scale
   glm::vec3 *dL_dscale = dL_dscales + idx;
   dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
   dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
   dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
 
   dL_dMt[0] *= s.x;
   dL_dMt[1] *= s.y;
   dL_dMt[2] *= s.z;
 
   // Gradients of loss w.r.t. normalized quaternion
   glm::vec4 dL_dq;
   dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) +
             2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
             2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
   dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) +
             2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
             2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) -
             4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
   dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) +
             2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
             2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) -
             4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
   dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) +
             2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
             2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) -
             4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
 
   // Gradients of loss w.r.t. unnormalized quaternion
   float4 *dL_drot = (float4 *)(dL_drots + idx);
   *dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z,
                     dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w },
                               // float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
 }
 
 // Backward pass of the preprocessing steps, except
 // for the covariance computation and inversion
 // (those are handled by a previous kernel call)
 template <int C>
 __global__ void preprocessCUDA(
     int P, int D, int M, const float3 *__restrict__ means,
     const int *__restrict__ radii, const float *__restrict__ dc,
     const float *__restrict__ shs, const float *__restrict__ splat_buffer,
     const glm::vec3 *__restrict__ scales,
     const glm::vec4 *__restrict__ rotations, const float scale_modifier,
     const float *__restrict__ proj, const glm::vec3 *__restrict__ campos,
     const float4 *__restrict__ dL_dmean2D, glm::vec3 *__restrict__ dL_dmeans,
     float *__restrict__ dL_dcolor, float *__restrict__ dL_dcov3D,
     float *__restrict__ dL_ddc, float *__restrict__ dL_dsh,
     glm::vec3 *__restrict__ dL_dscale, glm::vec4 *__restrict__ dL_drot) {
   auto idx = cg::this_grid().thread_rank();
   if (idx >= P || !(radii[idx] > 0))
     return;
 
   float3 m = means[idx];
 
   // Taking care of gradients from the screenspace points
   float4 m_hom = transformPoint4x4(m, proj);
   float m_w = 1.0f / (m_hom.w + 0.0000001f);
 
   // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
   // from rendering procedure
   glm::vec3 dL_dmean;
   float mul1 =
       (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
   float mul2 =
       (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
   dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x +
                (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
   dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x +
                (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
   dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x +
                (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;
 
   // That's the second part of the mean gradient. Previous computation
   // of cov2D and following SH conversion also affects it.
   dL_dmeans[idx] += dL_dmean;
 
   // Compute gradient updates due to computing colors from SHs
   if (shs)
     computeColorFromSH(idx, D, M, (glm::vec3 *)means, *campos, dc, shs,
                        splat_buffer, (glm::vec3 *)dL_dcolor,
                        (glm::vec3 *)dL_dmeans, (glm::vec3 *)dL_ddc,
                        (glm::vec3 *)dL_dsh);
 
   // Compute gradient updates due to computing covariance from scale/rotation
   if (scales)
     computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D,
                  dL_dscale, dL_drot);
 }
 
 __device__ __forceinline__ float3 ldg_float3_from_float4(const float4 *p) {
   float3 v;
   v.x = __ldg(&p->x);
   v.y = __ldg(&p->y);
   v.z = __ldg(&p->z);
   return v;
 }
 
 __device__ __forceinline__ void
 pix_backward(const float2 &xy, const int2 &pix, const float4 &con_o,
              const float3 &rgb, const float3 &bg, const float3 &dL_dpixel,
              float &T, float3 &res_ar, float &T_final, const float &ddelx_dx, const float &ddely_dy,
              float3 &dL_dcolor, float4 &dL_dmean2D, float4 &dL_dconic2D,
              float &dL_dopacity) {
   // dx, dy
   const float dx = xy.x - (float)pix.x;
   const float dy = xy.y - (float)pix.y;
 
   // power = con_o.w + con_o.x*dx^2 + con_o.z*dy^2 + con_o.y*dx*dy
   const float dx2 = dx * dx;
   const float dy2 = dy * dy;
   const float dxdy = dx * dy;
   float power = con_o.w + con_o.x * dx2 + con_o.z * dy2 + con_o.y * dxdy;
 
   float alpha;
   asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
   alpha = fminf(0.99f, alpha);
   if (alpha < ONE_OF_255) alpha = 0.f;
   // if (alpha < ONE_OF_255) return;
 
   // 复用 1/(1-alpha)
   const float one_minus_alpha = 1.f - alpha;
   const float inv_one_minus_alpha = 1.f / one_minus_alpha;
 
   // T = min(T / (1-alpha), 1)
   T = fminf(T * inv_one_minus_alpha, 1.f);
   const float dchannel_dcolor = alpha * T;
 
   // pixel = ar + res_ar
   // res_ar = (1 - alpha) * A
   // dpixel_dalpha = dar_dalpha - A = T * c - res_ar / (1 - alpha)
   const float rx = T * rgb.x - res_ar.x * inv_one_minus_alpha;
   const float ry = T * rgb.y - res_ar.y * inv_one_minus_alpha;
   const float rz = T * rgb.z - res_ar.z * inv_one_minus_alpha;
   float dL_dalpha = rx * dL_dpixel.x + ry * dL_dpixel.y + rz * dL_dpixel.z;
   
   // res_ar = pix - ar
   // ar = ar - alpha* T * c
   // res_ar = res_ar + alpha * T * c
   res_ar.x += dchannel_dcolor * rgb.x;
   res_ar.y += dchannel_dcolor * rgb.y;
   res_ar.z += dchannel_dcolor * rgb.z;
 
   
   // dL_dcolor += (alpha*T) * dL_dpixel
   dL_dcolor.x = fmaf(dchannel_dcolor, dL_dpixel.x, dL_dcolor.x);
   dL_dcolor.y = fmaf(dchannel_dcolor, dL_dpixel.y, dL_dcolor.y);
   dL_dcolor.z = fmaf(dchannel_dcolor, dL_dpixel.z, dL_dcolor.z);
 
 
   // background term: dL_dalpha += (-T_final/(1-alpha)) * dot(bg, dL_dpixel)
   const float bg_dot =
       bg.x * dL_dpixel.x + bg.y * dL_dpixel.y + bg.z * dL_dpixel.z;
   dL_dalpha = fmaf((-T_final * inv_one_minus_alpha), bg_dot, dL_dalpha);
 
   // dL_dpower = dL_dalpha * alpha * ln2
   const float dL_dpower = dL_dalpha * (alpha * ln2);
 
   // opacity = 2^(con_o.w)
   float opacity;
   asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(opacity) : "f"(con_o.w));
 
   // mean2D grad
   const float gx = (2.f * con_o.x * dx + con_o.y * dy) * ddelx_dx;
   const float gy = (2.f * con_o.z * dy + con_o.y * dx) * ddely_dy;
   const float tmp_x = dL_dpower * gx;
   const float tmp_y = dL_dpower * gy;
 
   dL_dmean2D.x += tmp_x;
   dL_dmean2D.y += tmp_y;
   dL_dmean2D.z += fabsf(tmp_x);
   dL_dmean2D.w += fabsf(tmp_y);
 
   // conic2D grad
   const float k = (-0.5f * log2e) * dL_dpower;
   dL_dconic2D.x = fmaf(k, dx2, dL_dconic2D.x);
   dL_dconic2D.y = fmaf(k, dxdy, dL_dconic2D.y);
   dL_dconic2D.w = fmaf(k, dy2, dL_dconic2D.w);
 
   // dL_dopacity += dL_dpower / opacity / ln2  == dL_dpower * log2e / opacity
   dL_dopacity = fmaf(dL_dpower, (log2e / opacity), dL_dopacity);
 }
 
 template <int OFFSET, int THREAD_X, int THREAD_Y>
 __device__ __forceinline__ void process_splat_block(
     float buf, int splat_id, int lane, const int2 &pix_min, int local_x0,
     int local_y0, const float3 &bg, const float ddelx_dx, const float ddely_dy,
     const uint32_t contrib_thres,
     // per-thread tile-local arrays
     float (&T)[THREAD_Y][THREAD_X], float3 (&res_ar)[THREAD_Y][THREAD_X],
     float (&T_final)[THREAD_Y][THREAD_X],
     uint32_t (&local_contrib)[THREAD_Y][THREAD_X],
     float3 (&dL_dpixel)[THREAD_Y][THREAD_X],
     // outputs (global)
     float4 *__restrict__ dL_dmean2D, float4 *__restrict__ dL_dconic2D,
     float *__restrict__ dL_dopacity, float *__restrict__ dL_dcolors) {
   float2 xy;
   float3 rgb;
   float4 con_o;
   get_gaussian_features(xy, rgb, con_o, buf, OFFSET);
 
   const int local_splat_id = __shfl_sync(FULL_MASK, splat_id, OFFSET);
 
   float3 dL_dcolor = {0.f, 0.f, 0.f};
   float4 dL_dmean = {0.f, 0.f, 0.f, 0.f};
   float dL_dopa = 0.f;
   float4 dL_dcon = {0.f, 0.f, 0.f, 0.f};
 
 #pragma unroll
   for (int i = 0; i < THREAD_Y; i++) {
 #pragma unroll
     for (int j = 0; j < THREAD_X; j++) {
       if (contrib_thres >= local_contrib[i][j])
         continue;
 
       const int2 pix = {pix_min.x + local_x0 + j, pix_min.y + local_y0 + i};
       pix_backward(xy, pix, con_o, rgb, bg, dL_dpixel[i][j], T[i][j],
                    res_ar[i][j], T_final[i][j], ddelx_dx, ddely_dy, dL_dcolor, dL_dmean, dL_dcon, dL_dopa);
     }
   }
 
   dL_dopa = warp_sum_float(dL_dopa);
   dL_dcolor = warp_sum_float3(dL_dcolor);
   dL_dmean = warp_sum_float4(dL_dmean);
   dL_dcon = warp_sum_float4(dL_dcon);
 
   if (lane == 0) {
     atomicAdd(&(dL_dcolors[local_splat_id * 3 + 0]), dL_dcolor.x);
     atomicAdd(&(dL_dcolors[local_splat_id * 3 + 1]), dL_dcolor.y);
     atomicAdd(&(dL_dcolors[local_splat_id * 3 + 2]), dL_dcolor.z);
 
     atomicAdd(&(dL_dmean2D[local_splat_id].x), dL_dmean.x);
     atomicAdd(&(dL_dmean2D[local_splat_id].y), dL_dmean.y);
     atomicAdd(&(dL_dmean2D[local_splat_id].z), dL_dmean.z);
     atomicAdd(&(dL_dmean2D[local_splat_id].w), dL_dmean.w);
 
     atomicAdd(&(dL_dconic2D[local_splat_id].x), dL_dcon.x);
     atomicAdd(&(dL_dconic2D[local_splat_id].y), dL_dcon.y);
     atomicAdd(&(dL_dconic2D[local_splat_id].w), dL_dcon.w);
 
     atomicAdd(&(dL_dopacity[local_splat_id]), dL_dopa);
   }
 }
 
 template <int THREAD_X, int THREAD_Y>
 __global__ void PerbucketRenderBackwardCUDA(
     uint32_t bucket_sum, const uint2 *__restrict__ bucket_ranges,
     const uint32_t *__restrict__ bucket_to_tile,
     const uint32_t *__restrict__ per_tile_bucket_offset,
 
     const uint32_t *__restrict__ point_list, int width, int height,
     const float3 *__restrict__ pixel_colors,
     const float3 *__restrict__ bg_color,
 
     render_load_info info, const float *__restrict__ final_Ts,
     const uint32_t *__restrict__ n_contrib, const uint32_t *max_contrib,
     const float *__restrict__ sampled_T, const float4 *__restrict__ sampled_ar,
     const float *__restrict__ dL_dpixels, float4 *__restrict__ dL_dmean2D,
     float4 *__restrict__ dL_dconic2D, float *__restrict__ dL_dopacity,
     float *__restrict__ dL_dcolors) {
   const int idx = (int)blockIdx.x;
   if ((uint32_t)idx >= bucket_sum)
     return;
 
   const int lane = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
 
   const uint2 range = bucket_ranges[idx];
   const int tile_id = (int)bucket_to_tile[idx];
   const int bbm = (tile_id == 0) ? 0 : (int)per_tile_bucket_offset[tile_id - 1];
   const int local_id = idx - bbm;
   if ((uint32_t)(local_id * BUCKET_SIZE) >= max_contrib[tile_id])
     return;
 
   const int horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
   const int2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
   const int2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};
 
   int to_do = (int)range.y - (int)range.x;
   if (to_do <= 0)
     return;
 
   const float ddelx_dx = 0.5f * (float)width;
   const float ddely_dy = 0.5f * (float)height;
 
   const void *data = info.data[lane];
   const int lg2_scale = info.lg2_scale[lane];
   const bool load_enable = (data != nullptr);
   const char *base = (const char *)data;
 
   const float *local_T = sampled_T + (size_t)idx * (size_t)BLOCK_SIZE;
   const float4 *local_ar = sampled_ar + (size_t)idx * (size_t)BLOCK_SIZE;
 
   const float3 bg = *bg_color;
 
   const int local_x0 = (int)threadIdx.x * THREAD_X;
   const int local_y0 = (int)threadIdx.y * THREAD_Y;
   const int global_x0 = pix_min.x + local_x0;
   const int global_y0 = pix_min.y + local_y0;
 
   float T[THREAD_Y][THREAD_X];
   float T_final[THREAD_Y][THREAD_X];
   float last_alpha[THREAD_Y][THREAD_X];
   uint32_t local_contrib[THREAD_Y][THREAD_X];
   float3 dL_dpixel[THREAD_Y][THREAD_X];
   float3 res_ar[THREAD_Y][THREAD_X];
 
 #pragma unroll
   for (int i = 0; i < THREAD_Y; i++) {
 #pragma unroll
     for (int j = 0; j < THREAD_X; j++) {
       T[i][j] = 0.f;
       T_final[i][j] = 0.f;
       last_alpha[i][j] = 0.f;
       local_contrib[i][j] = 0u;
       dL_dpixel[i][j] = {0.f, 0.f, 0.f};
       res_ar[i][j] = {0.f, 0.f, 0.f};
     }
   }
 
   const int HxW = height * width;
 #pragma unroll
   for (int i = 0; i < THREAD_Y; i++) {
 #pragma unroll
     for (int j = 0; j < THREAD_X; j++) {
       const int gx = global_x0 + j;
       const int gy = global_y0 + i;
       if (gx >= width || gy >= height)
         continue;
 
       const int lx = local_x0 + j;
       const int ly = local_y0 + i;
       const int local_pix_id = ly * BLOCK_X + lx;
       const int global_pix_id = gx + gy * width;
 
       T[i][j] = __ldg(local_T + local_pix_id);
 
       const float3 pc = pixel_colors[global_pix_id];
       const float4 ar = local_ar[local_pix_id];
       res_ar[i][j].x = pc.x - ar.x;
       res_ar[i][j].y = pc.y - ar.y;
       res_ar[i][j].z = pc.z - ar.z;
 
       local_contrib[i][j] = __ldg(n_contrib + global_pix_id);
 
       // for h, w, 3
       // dL_dpixel[i][j].x   = __ldg(dL_dpixels + 3 * global_pix_id + 0);
       // dL_dpixel[i][j].y   = __ldg(dL_dpixels + 3 * global_pix_id + 1);
       // dL_dpixel[i][j].z   = __ldg(dL_dpixels + 3 * global_pix_id + 2);
 
       // for 3, h, w
       dL_dpixel[i][j].x = __ldg(dL_dpixels + global_pix_id + 0 * HxW);
       dL_dpixel[i][j].y = __ldg(dL_dpixels + global_pix_id + 1 * HxW);
       dL_dpixel[i][j].z = __ldg(dL_dpixels + global_pix_id + 2 * HxW);
       T_final[i][j] = __ldg(final_Ts + global_pix_id);
     }
   }
 
   int offset = (int)range.y - 1;
   const uint32_t contrib_thres = (uint32_t)(local_id * BUCKET_SIZE);
 
   if ((to_do & 1) != 0) {
     const int pid = (int)point_list[offset];
 
     float buf = 0.f;
     if (load_enable) {
       const float *ptr = (const float *)(base + ((int64_t)pid << lg2_scale));
       buf = __ldg(ptr);
     }
 
     process_splat_block<0, THREAD_X, THREAD_Y>(
         buf, pid, lane, pix_min, local_x0, local_y0, bg, ddelx_dx, ddely_dy,
         contrib_thres, T, res_ar, T_final, local_contrib,
         dL_dpixel, dL_dmean2D, dL_dconic2D, dL_dopacity,
         dL_dcolors);
 
     to_do -= 1;
     offset -= 1;
   }
 
   offset = ((lane & 4) == 0) ? offset : (offset - 1);
   const bool done = (to_do == 0);
   int splat_id = 0;
   float buf = 0.f;
   float ldg_buf = 0.f;
 
   if (to_do > 0) {
     const int pid = (int)point_list[offset];
     splat_id = pid;
 
     if (load_enable) {
       const float *ptr = (const float *)(base + ((int64_t)pid << lg2_scale));
       buf = __ldg(ptr);
     }
 
     offset -= 2;
     to_do -= 2;
   }
 
   int ldg_pid = 0;
 
   while (__any_sync(FULL_MASK, (to_do >= 0) && (!done))) {
     if (to_do > 0) {
       ldg_pid = (int)point_list[offset];
       if (load_enable) {
         const float *ptr =
             (const float *)(base + ((int64_t)ldg_pid << lg2_scale));
         ldg_buf = __ldg(ptr);
       }
     }
 
     process_splat_block<0, THREAD_X, THREAD_Y>(
         buf, splat_id, lane, pix_min, local_x0, local_y0, bg, ddelx_dx,
         ddely_dy, contrib_thres, T, res_ar, T_final,
         local_contrib, dL_dpixel, dL_dmean2D, dL_dconic2D,
         dL_dopacity, dL_dcolors);
 
     process_splat_block<4, THREAD_X, THREAD_Y>(
         buf, splat_id, lane, pix_min, local_x0, local_y0, bg, ddelx_dx,
         ddely_dy, contrib_thres, T, res_ar, T_final,
         local_contrib, dL_dpixel, dL_dmean2D, dL_dconic2D,
         dL_dopacity, dL_dcolors);
 
     splat_id = ldg_pid;
     buf = ldg_buf;
     to_do -= 2;
     offset -= 2;
   }
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
 
 __global__ void AddDensificationStatsCUDA(
   int P,
   const int * __restrict__ radii,
   const float4 * __restrict__ dL_dmean2D,
   float * __restrict__ xyz_gradient_accum,
   float * __restrict__ xyz_gradient_accum_abs,
   float * __restrict__ max_radii,
   float * __restrict__ denom
 ){
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0))
    return;

  const float4 grad = __ldg(dL_dmean2D + idx);
  const int radius_val = radii[idx];


  if (denom)
    denom[idx] += 1;
  if (max_radii)
    max_radii[idx] = fast_max_f32(max_radii[idx], (float)radii[idx]);
  if (xyz_gradient_accum)
    xyz_gradient_accum[idx] += fast_sqrt_f32(grad.x * grad.x + grad.y * grad.y);
  if (xyz_gradient_accum_abs)
    xyz_gradient_accum_abs[idx] += fast_sqrt_f32(grad.z * grad.z + grad.w * grad.w);

 }
 
 } // namespace
 
 void preprocess_backward(int P, int D, int M, const float3 *means3D,
                          const int *radii, const float *dc, const float *shs,
                          const float *splat_buffer,
                          // const bool *clamped,
                          const glm::vec3 *scales, const glm::vec4 *rotations,
                          const float scale_modifier,
                          // const float *cov3Ds,
                          const float *viewmatrix, const float *projmatrix,
                          const float focal_x, float focal_y,
                          const float tan_fovx, float tan_fovy,
                          const glm::vec3 *campos, const float4 *dL_dmean2D,
                          const float *dL_dconic, glm::vec3 *dL_dmean3D,
                          float *dL_dcolor, float *dL_dcov3D, float *dL_ddc,
                          float *dL_dsh, glm::vec3 *dL_dscale,
                          glm::vec4 *dL_drot, cudaStream_t stream) {
 
   // Propagate gradients for the path of 2D conic matrix computation.
   // Somewhat long, thus it is its own kernel rather than being part of
   // "preprocess". When done, loss gradient w.r.t. 3D means has been
   // modified and gradient w.r.t. 3D covariance matrix has been computed.
   computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
       P, means3D, radii, splat_buffer, focal_x, focal_y, tan_fovx, tan_fovy,
       viewmatrix, dL_dconic, (float3 *)dL_dmean3D, dL_dcov3D);
 
   // Propagate gradients for remaining steps: finish 3D mean gradients,
   // propagate color gradients to SH (if desireD), propagate 3D covariance
   // matrix gradients to scale and rotation.
   preprocessCUDA<NUM_CHAFFELS><<<(P + 255) / 256, 256, 0, stream>>>(
       P, D, M, (float3 *)means3D, radii, dc, shs, splat_buffer,
       (glm::vec3 *)scales, (glm::vec4 *)rotations, scale_modifier, projmatrix,
       campos, (float4 *)dL_dmean2D, (glm::vec3 *)dL_dmean3D, dL_dcolor,
       dL_dcov3D, dL_ddc, dL_dsh, dL_dscale, dL_drot);
 }
 
 void render_backward(uint32_t bucket_sum, int width, int height,
                      const uint32_t *point_list, const float *bg_color,  
                      const float *splat_buffer, const float *final_Ts,
                      const uint32_t *n_contrib, char *image_buffer,
                      char *sample_buffer, const float *dL_dpixels,
                      float4 *dL_dmean2D, float4 *dL_dconic2D,
                      float *dL_dopacity, float *dL_dcolors,
                      cudaStream_t stream) {
 
   ImageState imgState = ImageState::fromChunk(image_buffer, width * height);
   SampleState sampleState = SampleState::fromChunk(sample_buffer, bucket_sum);
 
   render_load_info info(point_list, splat_buffer);
   PerbucketRenderBackwardCUDA<2, 4><<<bucket_sum, dim3(8, 4, 1), 0, stream>>>(
       bucket_sum, sampleState.bucket_ranges, sampleState.bucket_to_tile,
       imgState.bucket_offsets,
 
       point_list, width, height, imgState.pixel_colors, (float3 *)bg_color,
       info, final_Ts, n_contrib, imgState.max_contrib, sampleState.accum_T,
       sampleState.accum_ar, dL_dpixels, dL_dmean2D, dL_dconic2D, dL_dopacity,
       dL_dcolors);
 }
 
 
 void add_densification_stats(
   int P,
   const int *radii,
   const float4 *dL_dmean2D,
   float *xyz_gradient_accum,
   float *xyz_gradient_accum_abs,
   float *max_radii,
   float *denom,
   cudaStream_t stream
 ){
   AddDensificationStatsCUDA<<<(P + 255)/256, 256, 0, stream >>>(
     P,
     radii,
     dL_dmean2D,
     xyz_gradient_accum,
     xyz_gradient_accum_abs,
     max_radii,
     denom
   );
 }
 
 
 } // namespace faster