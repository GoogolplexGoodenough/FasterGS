import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

import os
from argparse import ArgumentParser


def system(cmd):
    print()
    print()
    print()
    print(cmd)
    print()
    sys.stdout.flush()
    os.system(cmd)


mipnerf360_outdoor_scenes = [
    "bicycle", 
    "flowers", "garden", "stump", "treehill"
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
parser.add_argument("--fastergs", action="store_true")
# parser.add_argument("--div_thres", type=float, default = 12)
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/fast")
parser.add_argument("--ip", default="127.0.0.1")
parser.add_argument("--sampling_factor", type=float, default = 0.6)


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
    common_args = f" --eval --test_iterations -1 --config_path {'../config/fast'} --sampling_factor {args.sampling_factor} --ip {args.ip} "
    if args.fastergs:
        for scene in mipnerf360_outdoor_scenes:
            scene_args = " --div_thres {} --mult 1".format(10)
            source = args.mipnerf360 + "/" + scene
            system("python train_faster.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + scene_args + " --imp_metric outdoor")    
        for scene in mipnerf360_indoor_scenes:
            scene_args = " --div_thres {} --mult 1".format(10)
            source = args.mipnerf360 + "/" + scene
            system("python train_faster.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + scene_args + " --imp_metric indoor")

        for scene in tanks_and_temples_scenes:
            scene_args = " --mult 1 --div_thres 8"
            source = args.tanksandtemples + "/" + scene
            system("python train_faster.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + scene_args + " --imp_metric outdoor")
        for scene in deep_blending_scenes:
            scene_args = " --mult 1 --div_thres 8"
            source = args.deepblending + "/" + scene
            system("python train_faster.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + scene_args + " --imp_metric indoor")
    else:
        for scene in mipnerf360_outdoor_scenes:
            source = args.mipnerf360 + "/" + scene
            system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args + " --imp_metric outdoor")    
        for scene in mipnerf360_indoor_scenes:
            source = args.mipnerf360 + "/" + scene
            system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args + " --imp_metric indoor")

        for scene in tanks_and_temples_scenes:
            source = args.tanksandtemples + "/" + scene
            system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + " --imp_metric outdoor")
        for scene in deep_blending_scenes:
            source = args.deepblending + "/" + scene
            system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + " --imp_metric indoor")

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
    if args.fastergs:
        for scene, source in zip(all_scenes, all_sources):
            system("python ../render_faster.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    else:
        for scene, source in zip(all_scenes, all_sources):
            system("python ../render.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    system("python ../metrics.py -m " + scenes_string)