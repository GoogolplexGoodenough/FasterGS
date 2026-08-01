#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, time
from argparse import ArgumentParser

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

# Scene-specific training parameters from train_base.sh (base mode)
scene_params_base = {
    "bicycle": {"grad_abs_thresh": 0.0012},
    "flowers": {"dense": 0.005, "grad_abs_thresh": 0.0015},
    "garden": {"highfeature_lr": 0.02, "loss_thresh": 0.06, "grad_abs_thresh": 0.0008},
    "stump": {"dense": 0.004, "grad_abs_thresh": 0.0015},
    "treehill": {"dense": 0.01, "grad_abs_thresh": 0.002},
    "room": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0008},
    "counter": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0008},
    "kitchen": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0006},
    "bonsai": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0006},
    "truck": {"highfeature_lr": 0.04, "grad_abs_thresh": 0.0009, "mult": 0.7},
    "train": {"highfeature_lr": 0.042, "grad_abs_thresh": 0.0015, "dense": 0.01, "mult": 0.7},
    "playroom": {"highfeature_lr": 0.0015, "dense": 0.003, "mult": 0.7},
    "drjohnson": {"highfeature_lr": 0.0025, "grad_abs_thresh": 0.0012, "dense": 0.013, "mult": 0.7}
}

# Scene-specific training parameters from train_big.sh (big mode)
scene_params_big = {
    "bicycle": {"grad_abs_thresh": 0.0008},
    "flowers": {"dense": 0.005, "grad_abs_thresh": 0.001},
    "garden": {"highfeature_lr": 0.02, "loss_thresh": 0.06, "grad_abs_thresh": 0.0003},
    "stump": {"dense": 0.004, "grad_abs_thresh": 0.001},
    "treehill": {"dense": 0.01, "grad_abs_thresh": 0.0018},
    "room": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0004},
    "counter": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0004},
    "kitchen": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0002},
    "bonsai": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0002},
    "truck": {"highfeature_lr": 0.04, "grad_abs_thresh": 0.0004, "mult": 0.7},
    "train": {"highfeature_lr": 0.042, "grad_abs_thresh": 0.0004, "dense": 0.015, "mult": 0.7},
    "playroom": {"highfeature_lr": 0.0015, "dense": 0.003, "mult": 0.7, "grad_abs_thresh": 0.0005},
    "drjohnson": {"highfeature_lr": 0.0025, "lowfeature_lr": 0.0005, "grad_abs_thresh": 0.0005, "dense": 0.005, "mult": 0.7}
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--mode", type=str, default="base", choices=["base", "big"])
parser.add_argument("--optimizer_type", type=str, default="default")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--fastergs", action="store_true")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if args.fastergs:
    train_file = "train_faster.py"
    render_file = "render_faster.py"
else:
    train_file = "train.py"
    render_file = "render.py"

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

def run_cmd(CMD, args):
    print(CMD)
    if not args.dry_run:
        os.system(CMD)

def get_scene_params(scene, mode):
    """Get scene-specific parameters as command line string"""
    if mode == "base":
        params = scene_params_base.get(scene, {})
    else:  # big
        params = scene_params_big.get(scene, {})
    param_str = ""
    if "grad_abs_thresh" in params:
        param_str += f" --grad_abs_thresh {params['grad_abs_thresh']}"
    if "dense" in params:
        param_str += f" --dense {params['dense']}"
    if "highfeature_lr" in params:
        param_str += f" --highfeature_lr {params['highfeature_lr']}"
    if "lowfeature_lr" in params:
        param_str += f" --lowfeature_lr {params['lowfeature_lr']}"
    if "loss_thresh" in params:
        param_str += f" --loss_thresh {params['loss_thresh']}"
    if "mult" in params:
        param_str += f" --mult {params['mult']}"
    return param_str

if not args.skip_training:
    common_args = " --eval --test_iterations 30000 "
    common_args += " --optimizer_type {}".format(args.optimizer_type)
    
    if args.mode == "base":
        mode_param = " --densification_interval 500"
        output_suffix = ""
    else:  # big
        mode_param = " --densification_interval 100"
        output_suffix = "_big"
    
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_param = get_scene_params(scene, args.mode)
        CMD = f"python {train_file} -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + output_suffix + common_args + mode_param + scene_param
        run_cmd(CMD, args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        scene_param = get_scene_params(scene, args.mode)
        CMD = f"python {train_file} -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + output_suffix + common_args + mode_param + scene_param
        run_cmd(CMD, args)
    m360_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        scene_param = get_scene_params(scene, args.mode)
        CMD = f"python {train_file} -s " + source + " -m " + args.output_path + "/" + scene + output_suffix + common_args + mode_param + scene_param
        run_cmd(CMD, args)
    tandt_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending  + "/" + scene
        scene_param = get_scene_params(scene, args.mode)
        CMD = f"python {train_file} -s " + source + " -m " + args.output_path + "/" + scene + output_suffix + common_args + mode_param + scene_param
        run_cmd(CMD, args)
    db_timing = (time.time() - start_time)/60.0

# if not args.dry_run:
#     with open(os.path.join(args.output_path, "timing.txt"), 'w') as file:
#         file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

if not args.skip_rendering:
    output_suffix = "_big" if args.mode == "big" else ""
    scene_params_dict = scene_params_big if args.mode == "big" else scene_params_base
    for scene in all_scenes:
        output_path = args.output_path + "/" + scene + output_suffix
        mult_param = ""
        if "mult" in scene_params_dict.get(scene, {}):
            mult_param = f" --mult {scene_params_dict[scene]['mult']}"
        CMD = f"python {render_file} -m {output_path}{mult_param} --eval --skip_train"
        run_cmd(CMD, args)

if not args.skip_metrics:
    output_suffix = "_big" if args.mode == "big" else ""
    for scene in all_scenes:
        output_path = args.output_path + "/" + scene + output_suffix
        CMD = f"python metrics.py -m {output_path}"
        run_cmd(CMD, args)