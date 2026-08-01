#include "../ops.h"
// #include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rasterizer_imp.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE 256


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


__forceinline__ __device__ void pixel_shader(float3 &C, float &T, int2 pix,
                                             float2 xy, float4 con_o,
                                             float3 rgb, 
                                             
                                             int curr_idx,
                                             int &idx_max, float &weight_max, bool &flag,
                                             float &weight_sum, int &weight_count
                                            ) {
  float2 d = {xy.x - (float)pix.x, xy.y - (float)pix.y};
  // float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y
  // * d.x * d.y;
  float power =
      con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
  float alpha;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
  alpha = min(0.99f, alpha);
  if (alpha < ONE_OF_255) return;

  const float weight = alpha * T;             
  
  C.x += rgb.x * weight;
  C.y += rgb.y * weight;
  C.z += rgb.z * weight;
  T -= weight;

  weight_sum += weight;
  weight_count += 1;

  if (weight_max < weight){
    weight_max = weight;
    idx_max = curr_idx;
    flag = true;
  }

}



template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
render_depthCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float * __restrict__ splat_buffer,
	
	float* __restrict__ out_pts,
	float* __restrict__ out_depth,
	float* __restrict__ accum_alpha,
	int* __restrict__ gidx,
	float* __restrict__ discriminants,

	const float* __restrict__ means3D,
	const glm::vec3* __restrict__ scales,
	const glm::vec4* __restrict__ rotations,

	const float* __restrict__ projmatrix,
	const glm::vec3* __restrict__ cam_pos
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	int contributor = 0;
	int last_contributor = 0;
	// float C[CHANNELS] = { 0 };

	float weight_max=0;
	float depth_max=0;
	float discriminant_max=0;

	int idx_max=0;
	int flag_update=0;

  glm::mat4 matrix = glm::make_mat4x4(projmatrix);
  glm::mat4 matrix_temp = glm::inverse(matrix);
	float *projmatrix_inv= glm::value_ptr(matrix_temp);

	glm::vec3 ray_origin = *cam_pos;
	glm::vec3 point_rec = {0,0,0};





	float3 p_proj_r = { Pix2ndc(pixf.x, W), Pix2ndc(pixf.y, H), 1};

	//inverse process of 'Transform point by projecting'
	float p_hom_x_r = p_proj_r.x*(1.0000001);
	float p_hom_y_r = p_proj_r.y*(1.0000001);
	// self.zfar = 100.0, self.znear = 0.01
	float p_hom_z_r = (100-100*0.01)/(100-0.01);
	float p_hom_w_r = 1;


	glm::vec3 p_hom_r= glm::vec3({p_hom_x_r, p_hom_y_r, p_hom_z_r});
	float4 p_orig_r=transformPoint4x4(p_hom_r, projmatrix_inv);

	glm::vec3 ray_direction={
		p_orig_r.x-ray_origin.x,
		p_orig_r.y-ray_origin.y,
		p_orig_r.z-ray_origin.z,
	};
	glm::vec3 normalized_ray_direction = glm::normalize(ray_direction);




	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			// collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			// collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			collected_xy[block.thread_rank()] = {
				__ldg(splat_buffer + coll_id * 32 + POINT_XY + 0),
				__ldg(splat_buffer + coll_id * 32 + POINT_XY + 1),
			};
			collected_conic_opacity[block.thread_rank()] = {
				__ldg(splat_buffer + coll_id * 32 + CON_O + 0),
				__ldg(splat_buffer + coll_id * 32 + CON_O + 1),
				__ldg(splat_buffer + coll_id * 32 + CON_O + 2),
				__ldg(splat_buffer + coll_id * 32 + CON_O + 3)
			};
		}
		block.sync();

	
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{

	
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			float power =
				con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
			float alpha;
			asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
			alpha = min(0.99f, alpha);
			
			if (alpha < ONE_OF_255)
				continue;

			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}	

			// for (int ch = 0; ch < CHANNELS; ch++)
			// 	C[ch] += splat_buffer[collected_id[j] * 32 + RGBD + ch] * alpha * T;
				
			// compute Gaussian depth
			// Normalize quaternion to get valid rotation
			glm::vec4 q = rotations[collected_id[j]];// / glm::length(rot);
			float rot_r = q.x;
			float rot_x = q.y;
			float rot_y = q.z;
			float rot_z = q.w;


			// Compute rotation matrix from quaternion
			glm::mat3 R = glm::mat3(
				1.f - 2.f * (rot_y * rot_y + rot_z * rot_z), 2.f * (rot_x * rot_y - rot_r * rot_z), 2.f * (rot_x * rot_z + rot_r * rot_y),
				2.f * (rot_x * rot_y + rot_r * rot_z), 1.f - 2.f * (rot_x * rot_x + rot_z * rot_z), 2.f * (rot_y * rot_z - rot_r * rot_x),
				2.f * (rot_x * rot_z - rot_r * rot_y), 2.f * (rot_y * rot_z + rot_r * rot_x), 1.f - 2.f * (rot_x * rot_x + rot_y * rot_y)
			);


			glm::vec3 temp={
				ray_origin.x-means3D[3*collected_id[j]+0],
				ray_origin.y-means3D[3*collected_id[j]+1],
				ray_origin.z-means3D[3*collected_id[j]+2],
			};
			glm::vec3 rotated_ray_origin = R * temp;
			glm::vec3 rotated_ray_direction = R * normalized_ray_direction;


			glm::vec3 a_t= rotated_ray_direction/(scales[collected_id[j]]*3.0f)*rotated_ray_direction/(scales[collected_id[j]]*3.0f);
			float a = a_t.x + a_t.y + a_t.z;

			glm::vec3 b_t= rotated_ray_direction/(scales[collected_id[j]]*3.0f)*rotated_ray_origin/(scales[collected_id[j]]*3.0f);
			float b = 2*(b_t.x + b_t.y + b_t.z);

			glm::vec3 c_t= rotated_ray_origin/(scales[collected_id[j]]*3.0f)*rotated_ray_origin/(scales[collected_id[j]]*3.0f);
			float c = c_t.x + c_t.y + c_t.z-1;


			float discriminant=b*b-4*a*c;	


			float depth = (-b/2/a)/glm::length(ray_direction);
			

			if(depth<0)
				continue;



			if(weight_max<alpha * T)
			{
				weight_max=alpha * T;
				depth_max=depth;
				discriminant_max=discriminant;
				idx_max=collected_id[j];

				point_rec = ray_origin+(-b/2/a)*normalized_ray_direction;			
			}

		
			
			T = test_T;
			last_contributor = contributor;
		}		
			

	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		for (int ch = 0; ch < 3; ch++)
			out_pts[ch * H * W + pix_id] = point_rec[ch];

		out_depth[pix_id] = depth_max;
		accum_alpha[pix_id] = T;
		discriminants[pix_id] = discriminant_max;
		gidx[pix_id]=idx_max;
	}
}



void render_depth(
	int P,
	int num_rendered,
	int width, int height,
	float* splat_buffer,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	uint2* ranges, 
	float* means3D, glm::vec3* scales, glm::vec4* rotations, float* projmatrix, glm::vec3* campos,
	float* out_pts, float* out_depth, float* accum_alpha, int* gidx, float* discriminants,
	cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

    // Identify start and end of per-tile workloads in sorted list
    identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
        num_rendered,
        gaussian_keys_sorted,
        ranges);

	render_depthCUDA<3><<<grid, dim3(16, 16, 1), 0, stream>>>(
		ranges,
		gaussian_values_sorted,
		width, height, 
		splat_buffer,
		
    out_pts, out_depth, accum_alpha, gidx, discriminants,
		
		means3D, scales, rotations, projmatrix, campos
	);
	CHECK_CUDA("render_depth");
}

} // namespace

void render_16x16_depth(
	int P,
	int num_rendered,
	int width, int height,
	float* splat_buffer,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	uint2* ranges, 
	float* means3D, glm::vec3* scales, glm::vec4* rotations, float* projmatrix, glm::vec3* campos,
	float* out_pts, float* out_depth, float* accum_alpha, int* gidx, float* discriminants,
	cudaStream_t stream)
{
	render_depth(P, num_rendered, width, height, splat_buffer,
				gaussian_keys_sorted, gaussian_values_sorted, ranges, 
        
				means3D, scales, rotations, projmatrix, campos,
				out_pts, out_depth, accum_alpha, gidx, discriminants,
				stream);
				
}


} // namespace flashgs