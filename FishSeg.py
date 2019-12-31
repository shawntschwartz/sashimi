#!/usr/bin/env python3

# Created by Shawn Schwartz 11/07/2019
# <shawnschwartz@ucla.edu>

version = "FishSeg"
build = "v1.0 - 20191231"

import os
import sys
import skimage.io
import numpy as np
import datetime
from PIL import Image
from fishseg import fishseg
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models", "fish_segmentation_models")
WEIGHTS_PATH = os.path.join(ROOT_DIR, MODEL_DIR, "fish_segmentation_model_schwartz_v1.h5")
DATASET_DIR = os.path.join(ROOT_DIR, "fishseg")

def welcome_message():
    print("Fish Segmentation [Version 1.0.0] built by Shawn T. Schwartz (2019) at Alfaro Lab, UCLA.")
    print("Contact: shawnschwartz@ucla.edu")
    print("Website: https://shawntylerschwartz.com")
    print("Github: @ShawnTylerSchwartz")

def welcome_fish():
    print("""   
        o                 o
                  o
         o   ______      o
           _/  (   \_
 _       _/  (       \_  O
| \_   _/  (   (    0  \\
|== \_/  (   (          |
|=== _ (   (   (        |
|==_/ \_ (   (          |
|_/     \_ (   (    \__/
          \_ (      _/
            |  |___/
           /__/
    """)

def exit_message():
    print("All processess completed...\n")

def exit_fish():
    print("""
         /¸...¸`:·
 ¸.·´  ¸   `·.¸.·´)
: © ):´;      ¸  {
 `·.¸ `·  ¸.·´\`·¸)
     `\\´´\¸.·´
    """)

def config_setup():
    config = fishseg.FishSegConfig()

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    dataset = fishseg.FishSegDataset()
    dataset.load_custom(DATASET_DIR, "val")
    dataset.prepare()

    return config, dataset

def get_fishes(dir):
    desired_exts = ['jpg', 'jpeg', 'png']
    return [f for f in os.listdir(dir) if any(f.endswith(ext) for ext in desired_exts)]

def get_fish_img(fish_file):
    print("Loading Fish File: ", fish_file)
    return skimage.io.imread(os.path.join(ROOT_DIR, args.input, fish_file))

def detect_mask(model, fish_img):
    return model.detect([fish_img], verbose=0)

def draw_mask(fish_img, mask):
    tmp_result = mask[0]
    tmp_mask = tmp_result['masks']
    tmp_mask = tmp_mask.astype(int)
    for ii in range(tmp_mask.shape[2]):
        tmp_img_mtrx_mask = fish_img
        for jj in range(tmp_img_mtrx_mask.shape[2]):
            tmp_img_mtrx_mask[:,:,jj] = tmp_img_mtrx_mask[:,:,jj] * tmp_mask[:,:,ii]
    return tmp_img_mtrx_mask, tmp_mask

def remove_bg(fish_img, mask, fish_name):
    height, width = fish_img.shape[:2]
    alpha_fish = np.dstack((fish_img, np.zeros((height, width), dtype=np.uint8)+255))
    alpha_fish[:,:,3] = alpha_fish[:,:,3] * mask[:,:,0]
    print("Background successfully removed!")
    save_fish(alpha_fish, fish_name)

def save_fish(fish_img, fish_name):
    bg_removed_img = Image.fromarray(fish_img, 'RGBA')
    bg_removed_img.save((os.path.join(ROOT_DIR, args.output, fish_name+".png")), "PNG")
    print("Successfully saved as ===> "+fish_name+".png  in  /"+args.output+"\n")

def execute_fish_image_processing(model, fish_set):
    if not os.path.exists(os.path.join(ROOT_DIR, "_logs")):
        os.mkdir(os.path.join(ROOT_DIR, "_logs"))
    if not os.path.exists(os.path.join(ROOT_DIR, args.output)):
        os.mkdir(os.path.join(ROOT_DIR, args.output))
    bgremoved_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_removebg.txt'
    bgremoved_log_file = open(os.path.join(ROOT_DIR, "_logs", bgremoved_filename), 'w')
    for counter, fish in enumerate(fish_set, start=1):
        start = datetime.datetime.now()
        loaded_fish = get_fish_img(fish)
        print("Classifing (",counter,"/",len(fish_set),"): ",fish)
        fish_detection_result = detect_mask(model, loaded_fish)
        tmp_masked_image, tmp_mask = draw_mask(loaded_fish, fish_detection_result)
        mod_filename = fish.split('.')
        mod_filename = mod_filename[0]
        remove_bg(tmp_masked_image, tmp_mask, mod_filename)
        end = datetime.datetime.now()
        elapsed_time = end - start
        datestamp = datetime.datetime.utcnow()
        output_path = "/"+args.output
        bgremoved_log_file.write("%s\t%s\t%s\t%s\n" % (datestamp, fish, output_path, elapsed_time))
    bgremoved_log_file.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove backgrounds from fish images and identify taxanomical family.')
    parser.add_argument('--input', required=True, metavar="/input/path/to/fish/images", help='Directory of fish images to remove backgrounds from or to identify')
    parser.add_argument('--output', required=True, metavar="/output/path/for/background-removed/fish/images", help='Directory to place background-removed fish images in')

    args = parser.parse_args()

    assert args.input and args.output,\
        "Arguments --input and --output are required to remove backgrounds of fish images"

    print("Input Directory for Fish Images: ", args.input)
    print("Output Location for Background-Removed Fish Images: ", args.output)

config, dataset = config_setup()

welcome_message()
welcome_fish()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("Loading fish segmentation model: ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Fish segmentation model successfully loaded from: ", WEIGHTS_PATH, "\n")

fish_files = get_fishes(args.input)

execute_fish_image_processing(model, fish_files)

exit_fish()
exit_message()