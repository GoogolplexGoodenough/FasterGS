import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui, render_simp, render_depth
from gaussian_renderer.render_faster import render, render_imp
import sys
from scene import Scene
from scene.gaussian_model_faster import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
from lpipsPyTorch import lpips
from utils.sh_utils import SH2RGB
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(sh_degree=0)

    scene = Scene(dataset, gaussians, resolution_scales=[1,2])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    optim_start = torch.cuda.Event(enable_timing = True)
    optim_end = torch.cuda.Event(enable_timing = True)
    total_time = 0.0

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras()))
    
    # 初始化视频帧保存相关变量（使用独立的时间追踪，不影响计时器统计）
    # 保存模式: 'images', 'video', 'both'
    save_mode = getattr(args, 'video_save_mode', 'images').lower()
    if save_mode not in ['images', 'video', 'both']:
        print(f"警告: 无效的保存模式 '{save_mode}'，使用默认模式 'images'")
        save_mode = 'images'
    
    # 检查视频合成库是否可用
    if save_mode in ['video', 'both'] and not CV2_AVAILABLE and not IMAGEIO_AVAILABLE:
        print("警告: 未安装 cv2 或 imageio，无法合成视频。将使用 'images' 模式。")
        save_mode = 'images'
    
    save_images = (save_mode == 'images' or save_mode == 'both')
    save_video = (save_mode == 'video' or save_mode == 'both')
    
    # 获取视频FPS参数
    video_fps = getattr(args, 'video_fps', 1.0)
    if video_fps <= 0:
        print(f"警告: 无效的FPS值 {video_fps}，使用默认值 1.0")
        video_fps = 1.0
    
    video_frames_dir = None
    if save_images:
        video_frames_dir = os.path.join(scene.model_path, "video_frames")
        os.makedirs(video_frames_dir, exist_ok=True)
    
    psnr_log_path = os.path.join(scene.model_path, "psnr_log.json")
    psnr_data = []  # 存储 {time: float, iteration: int, psnr: float}
    last_save_total_time = 0.0  # 上次保存时的total_time（用于判断保存间隔）
    video_save_interval = 1.0  # 每隔1秒训练时间保存一张图片
    
    # 初始化视频相机（使用测试集中的第一个相机）
    test_cameras = scene.getTestCameras()
    video_camera = test_cameras[0] if len(test_cameras) > 0 else None
    if video_camera is None:
        # 如果测试集为空，回退到训练集的第一张
        train_cameras = scene.getTrainCameras()
        video_camera = train_cameras[0] if len(train_cameras) > 0 else None
    
    # 视频帧缓冲区（用于合成视频）
    video_frames_buffer = [] if save_video else None

    for iteration in range(first_iter, opt.iterations + 1):   

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)


        if iteration % 1000 == 0 and iteration>args.simp_iteration1:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background, culling=gaussians._culling[:,viewpoint_cam.uid], xyz_grad=iteration<3000, mult=args.mult)

        image, radii = render_pkg["render"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            optim_start.record()
            # # Densification
            if iteration < opt.densify_until_iter:
    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                    grad_div = gaussians.xyz_gradient_accum_abs / gaussians.xyz_gradient_accum 
                    grad_div[grad_div.isnan()] = 0.0
                    
                    mask_blur = grad_div > args.div_thres
                    mask_blur = mask_blur.squeeze(-1)

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune_mask(opt.densify_grad_threshold, 
                                                    0.005, scene.cameras_extent, 
                                                    size_threshold, mask_blur)
                    # mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    # print(gaussians._xyz.shape)
                    
                if iteration == args.depth_reinit_iter:

                    num_depth = gaussians._xyz.shape[0]*args.num_depth_factor

                    # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                    gaussians.interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                    pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background)

                    gaussians.reinitial_pts(pts, rgb)

                    gaussians.training_setup(opt)
                    gaussians.init_culling(len(scene.getTrainCameras()))
                    # mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    # torch.cuda.empty_cache()
                    # print(gaussians._xyz.shape)

                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:
                    gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                    # torch.cuda.empty_cache()
                    # mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    # print(gaussians._xyz.shape)

            if iteration == args.simp_iteration1:
                gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, args, pipe, background)
                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                # torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)
                

            if iteration == args.simp_iteration2:
                gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                # torch.cuda.empty_cache()
                # print(gaussians._xyz.shape)

            if iteration == (args.simp_iteration2+opt.iterations)//2:
                gaussians.init_culling(len(scene.getTrainCameras()))


            # Optimizer step
            if iteration < opt.iterations:
                visible = radii>0
                gaussians.optimizer.step(visible, radii.shape[0])
                # gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # 每隔1秒训练时间保存视频帧（使用total_time判断保存间隔，与训练时间统计保持一致）
            elapsed_since_last_save = total_time - last_save_total_time
            if elapsed_since_last_save >= video_save_interval and video_camera is not None:
                # 渲染图片
                render_pkg_video = render(video_camera, gaussians, pipe, background, mult=args.mult)
                video_image = torch.clamp(render_pkg_video["render"], 0.0, 1.0)
                
                # 计算PSNR
                gt_image_video = torch.clamp(video_camera.original_image.to("cuda"), 0.0, 1.0)
                psnr_value = psnr(video_image, gt_image_video).mean().item()
                
                # 转换为numpy数组
                image_np = (video_image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                frame_filename = None
                # 根据保存模式决定操作
                if save_images:
                    # 保存图片
                    frame_filename = os.path.join(video_frames_dir, f"frame_{len(psnr_data):06d}.png")
                    Image.fromarray(image_np).save(frame_filename)
                
                if save_video:
                    # 添加到视频缓冲区（BGR格式用于cv2，RGB格式用于imageio）
                    if CV2_AVAILABLE:
                        # cv2使用BGR格式
                        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        video_frames_buffer.append(frame_bgr)
                    elif IMAGEIO_AVAILABLE:
                        # imageio使用RGB格式
                        video_frames_buffer.append(image_np)
                
                # 记录PSNR数据（使用total_time而不是墙时钟时间，与训练时间统计保持一致）
                psnr_entry = {
                    "time": total_time,
                    "iteration": iteration,
                    "psnr": psnr_value,
                }
                if save_images and frame_filename:
                    psnr_entry["frame_filename"] = os.path.basename(frame_filename)
                psnr_data.append(psnr_entry)
                
                # 更新上次保存时的total_time
                last_save_total_time = total_time
                
                # 实时保存PSNR日志
                with open(psnr_log_path, 'w') as f:
                    json.dump(psnr_data, f, indent=2)  

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    with open(os.path.join(scene.model_path, "TRAIN_INFO.txt"), "w+") as f:
        f.write("Training Time: {:.2f} seconds, {:.2f} minutes\n".format(total_time, total_time / 60.))
        f.write("GS Number: {}\n".format(gaussians.get_xyz.shape[0]))
    
    # 合成视频（如果选择了视频模式）
    video_path = None
    if save_video and video_frames_buffer is not None and len(video_frames_buffer) > 0:
        video_path = os.path.join(scene.model_path, "training_progress.mp4")
        fps = video_fps  # 使用用户指定的FPS
        
        print(f"\n正在合成视频，共 {len(video_frames_buffer)} 帧，帧率: {fps} FPS...")
        try:
            if CV2_AVAILABLE:
                # 使用cv2合成视频
                height, width = video_frames_buffer[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                for frame in video_frames_buffer:
                    out.write(frame)
                out.release()
                print(f"视频已保存到: {video_path}")
            elif IMAGEIO_AVAILABLE:
                # 使用imageio合成视频
                imageio.mimwrite(video_path, video_frames_buffer, fps=fps, codec='libx264', quality=8)
                print(f"视频已保存到: {video_path}")
        except Exception as e:
            print(f"警告: 视频合成失败: {e}")
            video_path = None
    
    # 绘制并保存PSNR随时间变化的图表
    if len(psnr_data) > 0:
        times = [entry["time"] for entry in psnr_data]
        psnr_values = [entry["psnr"] for entry in psnr_data]
        iterations = [entry["iteration"] for entry in psnr_data]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(times, psnr_values, 'b-', linewidth=2, label='PSNR')
        plt.xlabel('训练时间 (秒)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title('训练过程中PSNR随时间变化', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # 保存图表
        psnr_plot_path = os.path.join(scene.model_path, "psnr_over_time.png")
        plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 最终保存PSNR日志
        with open(psnr_log_path, 'w') as f:
            json.dump(psnr_data, f, indent=2)
        
        # 打印保存信息
        print(f"\n=== 视频/图片保存信息 ===")
        if save_images and video_frames_dir:
            print(f"图片帧已保存到: {video_frames_dir}")
        if save_video and video_path:
            print(f"训练视频已保存到: {video_path} (FPS: {video_fps})")
        print(f"PSNR日志已保存到: {psnr_log_path}")
        print(f"PSNR变化图表已保存到: {psnr_plot_path}")
        print(f"共保存了 {len(psnr_data)} 帧")
        if save_mode == 'video':
            print(f"注意: 已选择仅保存视频模式，中间图片已合成视频，可节省硬盘空间")
    
    return 









def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # print(args.mult, args)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


    # exit()
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)        

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--imp_metric", required=True, type=str, default = None)


    parser.add_argument("--config_path", type=str)

    parser.add_argument("--aggressive_clone_from_iter", type=int, default = 500)
    parser.add_argument("--aggressive_clone_interval", type=int, default = 250)

    parser.add_argument("--div_thres", type=float, default = 12)
    parser.add_argument("--warn_until_iter", type=int, default = 3_000)
    parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
    parser.add_argument("--num_depth_factor", type=float, default=1)

    parser.add_argument("--simp_iteration1", type=int, default = 3_000)
    parser.add_argument("--simp_iteration2", type=int, default = 8_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.6)

    parser.add_argument("--video_save_mode", type=str, default="video", 
                        choices=["images", "video", "both"],
                        help="视频/图片保存模式: 'images' (仅保存图片), 'video' (仅保存视频，节省空间), 'both' (同时保存图片和视频)")
    parser.add_argument("--video_fps", type=float, default=20,
                        help="视频帧率 (FPS)，默认1.0（与实际捕获速率一致）。可设置为更高值（如30）使视频播放更快")
    parser.add_argument("--mult", type=float, default=1.0,  help="mult参数")


    args = parser.parse_args(sys.argv[1:])

    args = read_config(parser)
    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
