import os
from argparse import ArgumentParser
import time

MAX_N_GAUSSIAN = {
    "bicycle": 4877226, 
    "garden": 4137614, 
    "stump": 4408311, 
    "flowers": 2901071, 
    "treehill": 3267573, 
    "room": 1311219, 
    "kitchen": 1597159, 
    "counter": 1083296, 
    "bonsai": 1074610, 
    "drjohnson": 3121872, 
    "playroom": 1840101, 
    "train": 1088608, 
    "truck": 2061586, 
}
MAX_N_GAUSSIAN_FAST = {
    "bicycle": 4033850, 
    "garden": 3480490, 
    "stump": 4246754, 
    "flowers": 2586116, 
    "treehill": 2685730, 
    "room": 1168731, 
    "kitchen": 1740068, 
    "counter": 1059282, 
    "bonsai": 1162322, 
    "drjohnson": 2933140, 
    "playroom": 1703963, 
    "train": 1076535, 
    "truck": 1996811, 
}

mipnerf360_outdoor_scenes = [
    "bicycle", "flowers", "garden", "stump", "treehill"
    ]
mipnerf360_indoor_scenes = [
    "room", "counter", "kitchen", 
    "bonsai"
    ]
tanks_and_temples_scenes = [
    "truck", "train"
    ]
deep_blending_scenes = [
    "drjohnson", "playroom"
    ]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--use_expcomp", action="store_true")
parser.add_argument("--fast", action="store_true")
parser.add_argument("--aa", action="store_true")
parser.add_argument("--gpu", required=True, type=str)
parser.add_argument("--dash", action="store_true")
parser.add_argument("--fastergs", action="store_true")
parser.add_argument("--preset_upperbound", action="store_true")

args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()
if not args.skip_training:
    common_args = " --disable_viewer --eval "
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_depth:
        common_args += " -d depths2/ "

    if args.use_expcomp:
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    file_name = "train.py"
    if args.dash:
        file_name = "train_dash.py"
        common_args += " --densify_mode freq --resolution_mode freq --densify_until_iter 27000 "
        max_n_gs_dict = MAX_N_GAUSSIAN if not args.fast else MAX_N_GAUSSIAN_FAST

    if args.fastergs:
        file_name = "train_dash_faster.py"
        common_args += " --densify_mode freq --resolution_mode freq --densify_until_iter 27000 "
        max_n_gs_dict = MAX_N_GAUSSIAN if not args.fast else MAX_N_GAUSSIAN_FAST

    scene_args = ""
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        if args.dash and args.preset_upperbound:
            scene_args = " --max_n_gaussian {} ".format(max_n_gs_dict[scene])
        os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + f"python {file_name} -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + scene_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        if args.dash and args.preset_upperbound:
            scene_args = " --max_n_gaussian {} ".format(max_n_gs_dict[scene])
        os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + f"python {file_name} -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + scene_args)

    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        if args.dash and args.preset_upperbound:
            scene_args = " --max_n_gaussian {} ".format(max_n_gs_dict[scene])
        os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + f"python {file_name} -s " + source + " -m " + args.output_path + "/" + scene + common_args + scene_args)

    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        if args.dash and args.preset_upperbound:
            scene_args = " --max_n_gaussian {} ".format(max_n_gs_dict[scene])
        os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + f"python {file_name} -s " + source + " -m " + args.output_path + "/" + scene + common_args + scene_args)


if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    
    common_args = " --eval --skip_train"
    
    if args.aa:
        common_args += " --antialiasing "
    if args.use_expcomp:
        common_args += " --train_test_exp "

    for scene, source in zip(all_scenes, all_sources):
        if args.fastergs:
            os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + "python render_faster.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        else:
            os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + "python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("CUDA_VISIBLE_DEVICE={} ".format(args.gpu) + "python metrics.py -m " + scenes_string)
