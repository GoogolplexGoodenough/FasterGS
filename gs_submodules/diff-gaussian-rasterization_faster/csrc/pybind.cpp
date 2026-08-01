#include "ops.h"

#include <torch/extension.h>

#include "cuda_rasterizer/adam.h"
#include <ATen/cuda/CUDAContext.h>

#include <fstream>
#include <iostream>
#include <string>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
    auto lambda = [&t](size_t N)
    {
        t.resize_({(long long)N});
        return reinterpret_cast<char *>(t.contiguous().data_ptr());
    };
    return lambda;
}

namespace faster
{
    namespace
    {
        torch::Tensor preprocess_torch(
            int deg,
            torch::Tensor &orig_points, torch::Tensor &color_precomp, torch::Tensor &dc, torch::Tensor &shs, torch::Tensor &opacities,
            torch::Tensor &scales, torch::Tensor &rotations, float scale_modifier,
            int width, int height, int block_x, int block_y,
            torch::Tensor &campos, torch::Tensor &view_matrix, torch::Tensor &proj_matrix,
            float tan_fovx, float tan_fovy, float zFar, float zNear,
            torch::Tensor &gaussian_keys_unsorted, torch::Tensor &gaussian_values_unsorted,
            torch::Tensor &gaussian_values_sorted,
            torch::Tensor &curr_offset, float mult, torch::Tensor &splat_buffer, torch::Tensor &culling)
        {
            int number = opacities.size(0);

            // torch::Tensor splat_buffer = torch::zeros({number, 32}, torch::TensorOptions().device(opacities.device()).dtype(torch::kFloat32));
            // torch::Tensor clamped = torch::zeros({number, 3}, torch::TensorOptions().device(opacities.device()).dtype(torch::kBool));
            torch::Tensor radii = torch::zeros({number}, torch::TensorOptions().device(opacities.device()).dtype(torch::kInt));
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

            int max_coeffs = 0;
            if (shs.size(0) != 0)
            {
                max_coeffs = shs.size(1);
            }

            preprocess(
                number, deg, max_coeffs,
                (glm::vec3 *)orig_points.contiguous().data_ptr<float>(),
                color_precomp.contiguous().data_ptr<float>(),
                dc.contiguous().data_ptr<float>(),
                shs.contiguous().data_ptr<float>(),
                opacities.contiguous().data_ptr<float>(),
                scales.contiguous().data_ptr<float>(),
                rotations.contiguous().data_ptr<float>(),
                scale_modifier,
                width, height, block_x, block_y,
                (glm::vec3 *)campos.contiguous().data_ptr<float>(),
                view_matrix.contiguous().data_ptr<float>(),
                proj_matrix.contiguous().data_ptr<float>(),
                tan_fovx, tan_fovy, zFar, zNear,
                splat_buffer.contiguous().data_ptr<float>(),
                (uint64_t *)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(),
                (uint32_t *)gaussian_values_unsorted.contiguous().data_ptr<int>(),
                (uint32_t *)gaussian_values_sorted.contiguous().data_ptr<int>(),
                curr_offset.data_ptr<int>(),
                radii.contiguous().data_ptr<int>(), mult,
                culling.contiguous().data_ptr<bool>(),
                stream);

            CHECK_CUDA("preprocess");
            return radii;
        }

        void sort_gaussian_torch(int num_rendered,
                                 int width, int height, int block_x, int block_y,
                                 torch::Tensor &list_sorting_space,
                                 torch::Tensor &gaussian_keys_unsorted, torch::Tensor &gaussian_values_unsorted,
                                 torch::Tensor &gaussian_keys_sorted, torch::Tensor &gaussian_values_sorted)
        {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            sort_gaussian(num_rendered,
                          width, height, block_x, block_y,
                          (char *)list_sorting_space.contiguous().data_ptr(), list_sorting_space.size(0),
                          (uint64_t *)gaussian_keys_unsorted.contiguous().data_ptr<int64_t>(), (uint32_t *)gaussian_values_unsorted.contiguous().data_ptr<int>(),
                          (uint64_t *)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(), (uint32_t *)gaussian_values_sorted.contiguous().data_ptr<int>(),
                          stream);

            CHECK_CUDA("preprocess");
        }

        size_t get_sort_buffer_size_torch(int num_rendered)
        {
            return get_sort_buffer_size(num_rendered);
        }

        std::tuple<uint32_t, torch::Tensor, torch::Tensor, torch::Tensor>
        render_16x16_torch(
            int P,
            int num_rendered,
            int width, int height,
            torch::Tensor &splat_buffer,
            torch::Tensor &gaussian_keys_sorted, torch::Tensor &gaussian_values_sorted,
            torch::Tensor &ranges,
            torch::Tensor &bg_color,
            torch::Tensor &last_contributor, torch::Tensor &Ts_final)
        {
            torch::Tensor out_color = torch::zeros({3, height, width}, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor img_buffer = torch::empty({0}, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor smp_buffer = torch::empty({0}, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            std::function<char *(size_t)> img_func = resizeFunctional(img_buffer);
            std::function<char *(size_t)> smp_func = resizeFunctional(smp_buffer);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            uint32_t bucket_sum = render_16x16(
                P, num_rendered,
                width, height,
                (float *)splat_buffer.contiguous().data_ptr<float>(),
                (uint64_t *)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
                (uint32_t *)gaussian_values_sorted.contiguous().data_ptr<int>(),
                (uint2 *)ranges.contiguous().data_ptr<int>(),
                (float3 *)bg_color.contiguous().data_ptr<float>(),
                out_color.contiguous().data_ptr<float>(),
                last_contributor.contiguous().data_ptr<int>(),
                Ts_final.contiguous().data_ptr<float>(),

                img_func, smp_func, stream);

            CHECK_CUDA("render_16x16");
            return std::make_tuple(bucket_sum, out_color, img_buffer, smp_buffer);
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        render_depth_torch(
            int P,
            int num_rendered,
            int width, int height,
            torch::Tensor &splat_buffer,
            torch::Tensor &gaussian_keys_sorted, torch::Tensor &gaussian_values_sorted,
            torch::Tensor &ranges,
            torch::Tensor &means3D, torch::Tensor &scales, torch::Tensor &rotations,
            torch::Tensor &proj_matrix, torch::Tensor &campos)
        {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            torch::Tensor out_pts = torch::zeros({3, height, width}, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor out_depth = torch::full({1, height, width}, 0.0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor accum_alpha = torch::full({1, height, width}, 0.0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor discriminants = torch::full({1, height, width}, 0.0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor gidx = torch::full({1, height, width}, 0.0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kInt));

            render_16x16_depth(
                P, num_rendered,
                width, height,
                (float *)splat_buffer.contiguous().data_ptr<float>(),
                (uint64_t *)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
                (uint32_t *)gaussian_values_sorted.contiguous().data_ptr<int>(),
                (uint2 *)ranges.contiguous().data_ptr<int>(),

                means3D.contiguous().data_ptr<float>(),
                (glm::vec3 *)scales.contiguous().data_ptr<float>(),
                (glm::vec4 *)rotations.contiguous().data_ptr<float>(),
                proj_matrix.contiguous().data_ptr<float>(),
                (glm::vec3 *)campos.contiguous().data_ptr<float>(),

                out_pts.contiguous().data_ptr<float>(),
                out_depth.contiguous().data_ptr<float>(),
                accum_alpha.contiguous().data_ptr<float>(),
                gidx.contiguous().data_ptr<int>(),
                discriminants.contiguous().data_ptr<float>(),
                stream);

            CHECK_CUDA("render_16x16_depth");
            return std::make_tuple(out_pts, out_depth, accum_alpha, gidx, discriminants);
        }

        
        std::tuple<uint32_t, torch::Tensor, torch::Tensor, torch::Tensor>
        render_simp_torch(
            int P,
            int num_rendered,
            int width, int height,
            torch::Tensor &splat_buffer,
            torch::Tensor &gaussian_keys_sorted, torch::Tensor &gaussian_values_sorted,
            torch::Tensor &ranges
        ){
            torch::Tensor accum_weights_ptr = torch::full({P}, 0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));
            torch::Tensor accum_weights_count = torch::full({P}, 0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kInt));
            torch::Tensor accum_max_count = torch::full({P}, 0, torch::TensorOptions().device(splat_buffer.device()).dtype(torch::kFloat));

            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            uint32_t bucket_sum = render_16x16_simp(
                P, num_rendered,
                width, height,
                (float *)splat_buffer.contiguous().data_ptr<float>(),
                (uint64_t *)gaussian_keys_sorted.contiguous().data_ptr<int64_t>(),
                (uint32_t *)gaussian_values_sorted.contiguous().data_ptr<int>(),
                (uint2 *)ranges.contiguous().data_ptr<int>(),
                
                accum_weights_ptr.contiguous().data<float>(),  
                accum_weights_count.contiguous().data<int>(),  
                accum_max_count.contiguous().data<float>(),  
                stream);

            CHECK_CUDA("render_16x16_simp");
            return std::make_tuple(bucket_sum, accum_weights_ptr, accum_weights_count, accum_max_count);
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        backward_torch(
            const int degree,
            const uint32_t bucket_sum,
            const torch::Tensor &means3D,
            const torch::Tensor &radii,
            const torch::Tensor &dc,
            const torch::Tensor &sh,
            const torch::Tensor &opacities,
            const torch::Tensor &scales,
            const torch::Tensor &rotations,
            const float scale_modifier,

            const int width, const int height, const int block_x, const int block_y,

            const float tan_fovx,
            const float tan_fovy,
            const float zFar, const float zNear,

            const torch::Tensor &campos,
            const torch::Tensor &viewmatrix,
            const torch::Tensor &projmatrix,

            const torch::Tensor &img_func,
            const torch::Tensor &smp_func,
            const torch::Tensor &splat_buffer,

            const torch::Tensor &point_list,
            const torch::Tensor &ranges,
            const torch::Tensor &background,

            const torch::Tensor &last_contributor,
            const torch::Tensor &Ts_final,
            const torch::Tensor &dL_dout_color,

            torch::Tensor &xyz_gradient_accum,
            torch::Tensor &xyz_gradient_accum_abs,
            torch::Tensor &max_radii,
            torch::Tensor &denom)
        {
            const int P = means3D.size(0);

            int M = 0;
            if (sh.size(0) != 0)
            {
                M = sh.size(1);
            }

            torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
            torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
            torch::Tensor dL_dcolors = torch::zeros({P, 3}, means3D.options());
            torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
            torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
            torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
            torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
            torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
            torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
            torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

            const float focal_y = height / (2.0f * tan_fovy);
            const float focal_x = width / (2.0f * tan_fovx);

            const dim3 tile_grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);
            const dim3 block(block_x, block_y, 1);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

            float *splat_buffer_ptr = splat_buffer.contiguous().data_ptr<float>();

            if (P != 0)
            {
                render_backward(
                    bucket_sum,
                    width, height,
                    (uint32_t *)point_list.contiguous().data_ptr<int>(),
                    background.contiguous().data_ptr<float>(),
                    splat_buffer_ptr,
                    (float *)Ts_final.contiguous().data_ptr<float>(),
                    (uint32_t *)last_contributor.contiguous().data_ptr<int>(),
                    reinterpret_cast<char *>(img_func.contiguous().data_ptr()),
                    reinterpret_cast<char *>(smp_func.contiguous().data_ptr()),
                    dL_dout_color.contiguous().data_ptr<float>(),
                    (float4 *)dL_dmeans2D.contiguous().data_ptr<float>(),
                    (float4 *)dL_dconic.contiguous().data_ptr<float>(),
                    dL_dopacity.contiguous().data_ptr<float>(),
                    dL_dcolors.contiguous().data_ptr<float>(), stream);

                CHECK_CUDA("render_backward");

                preprocess_backward(
                    P, degree, M,
                    (float3 *)means3D.contiguous().data_ptr<float>(),
                    radii.contiguous().data_ptr<int>(),
                    dc.contiguous().data_ptr<float>(),
                    sh.contiguous().data_ptr<float>(),
                    splat_buffer_ptr,
                    (glm::vec3 *)scales.contiguous().data_ptr<float>(),
                    (glm::vec4 *)rotations.contiguous().data_ptr<float>(),
                    scale_modifier,
                    viewmatrix.contiguous().data_ptr<float>(),
                    projmatrix.contiguous().data_ptr<float>(),
                    focal_x, focal_y, tan_fovx, tan_fovy,
                    (glm::vec3 *)campos.contiguous().data_ptr<float>(),
                    (float4 *)dL_dmeans2D.contiguous().data_ptr<float>(),
                    dL_dconic.contiguous().data_ptr<float>(),
                    (glm::vec3 *)dL_dmeans3D.contiguous().data_ptr<float>(),
                    dL_dcolors.contiguous().data_ptr<float>(),
                    dL_dcov3D.contiguous().data_ptr<float>(),
                    dL_ddc.contiguous().data_ptr<float>(),
                    dL_dsh.contiguous().data_ptr<float>(),
                    (glm::vec3 *)dL_dscales.contiguous().data_ptr<float>(),
                    (glm::vec4 *)dL_drotations.contiguous().data_ptr<float>(), stream);

                CHECK_CUDA("preprocess_backward");

                if (xyz_gradient_accum.size(0) != 0)
                {
                    add_densification_stats(
                        P,
                        radii.contiguous().data_ptr<int>(),
                        (float4 *)dL_dmeans2D.contiguous().data_ptr<float>(),
                        xyz_gradient_accum.contiguous().data_ptr<float>(),
                        xyz_gradient_accum_abs.contiguous().data_ptr<float>(),
                        max_radii.contiguous().data_ptr<float>(),
                        denom.contiguous().data_ptr<float>(),
                        stream);
                }
            }

            return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
        }

        void adamUpdate(
            torch::Tensor &param,
            torch::Tensor &param_grad,
            torch::Tensor &exp_avg,
            torch::Tensor &exp_avg_sq,
            torch::Tensor &visible,
            const float lr,
            const float b1,
            const float b2,
            const float eps,
            const uint32_t N,
            const uint32_t M)
        {
            ADAM::adamUpdate(
                param.contiguous().data<float>(),
                param_grad.contiguous().data<float>(),
                exp_avg.contiguous().data<float>(),
                exp_avg_sq.contiguous().data<float>(),
                visible.contiguous().data<bool>(),
                lr,
                b1,
                b2,
                eps,
                N,
                M);
        }

    } // namespace
} // namespace faster

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto ops = m.def_submodule("ops", "my custom operators");

    ops.def(
        "preprocess",
        &faster::preprocess_torch,
        "preprocess gaussian model data and generate key-value pairs");

    ops.def(
        "sort_gaussian",
        &faster::sort_gaussian_torch,
        "sort gaussian key-value pairs");

    ops.def(
        "get_sort_buffer_size",
        &faster::get_sort_buffer_size_torch,
        "get sort buffer size");

    ops.def(
        "render_16x16",
        &faster::render_16x16_torch,
        "sort key-value pairs and render");

    ops.def(
        "render_simp",
        &faster::render_simp_torch,
        "render_simp"
    );

    ops.def(
        "render_depth",
        &faster::render_depth_torch,
        "render_depth"
    );

    ops.def(
        "backward",
        &faster::backward_torch,
        "backward process");

    ops.def(
        "adamUpdate",
        &faster::adamUpdate,
        "adamUpdate");
}
