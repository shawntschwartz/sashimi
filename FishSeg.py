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
import urllib.request
import shutil
from zipfile import ZipFile
from PIL import Image
from fishseg import core
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

MODEL_URL = "https://github.com/ShawnTylerSchwartz/fish-segmentation/releases/download/V1.0/fish_segmentation_model_schwartz_v1.h5"
TRAIN_ZIP_FILE = "train.zip"
VAL_ZIP_FILE = "val.zip"
TRAIN_ZIP_PATH = os.path.join(ROOT_DIR, "fishseg")
VAL_ZIP_PATH = os.path.join(ROOT_DIR, "fishseg")

def welcome_message():
    print("Fish Segmentation [Version 1.0] built by Shawn T. Schwartz (2019) at Alfaro Lab, UCLA.")
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

def exit_message(time):
    print("All processess completed...Total time to complete process: ",time,".\n",sep="")

def exit_fish():
    print("""
         /¸...¸`:·
 ¸.·´  ¸   `·.¸.·´)
: © ):´;      ¸  {
 `·.¸ `·  ¸.·´\`·¸)
     `\\´´\¸.·´
    """)

def config_setup():
    config = core.FishSegConfig()

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    config = InferenceConfig()
    config.display()

    dataset = core.FishSegDataset()
    dataset.load_custom(DATASET_DIR, "val")
    dataset.prepare()

    return config, dataset

def download_model(URL, destination):
    print("Downloading fish segmentation model to",destination,"...")
    with urllib.request.urlopen(URL) as resp, open(destination, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("Fish segmentation model successfully downloaded!")

def unpack_zip(path, file):
    print("Unpacking asset at: ",path,"/",file,sep="")
    zip = ZipFile(os.path.join(path, file))
    zip.extractall(path)
    zip.close()

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
    print("Successfully saved as ===> ",fish_name,".png  in  /",args.output,"\n",sep="")

def execute_fish_image_processing(model, fish_set):
    if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
        os.mkdir(os.path.join(ROOT_DIR, "logs"))
    if not os.path.exists(os.path.join(ROOT_DIR, args.output)):
        os.mkdir(os.path.join(ROOT_DIR, args.output))
    elapsed_times = []
    bgremoved_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_segmentation.csv'
    bgremoved_log_file = open(os.path.join(ROOT_DIR, "logs", bgremoved_filename), 'w')
    for counter, fish in enumerate(fish_set, start=1):
        start = datetime.datetime.now()
        loaded_fish = get_fish_img(fish)
        print("Segmenting (",counter,"/",len(fish_set),"): ",fish,sep="")
        fish_detection_result = detect_mask(model, loaded_fish)
        tmp_masked_image, tmp_mask = draw_mask(loaded_fish, fish_detection_result)
        mod_filename = fish.split('.')
        mod_filename = mod_filename[0]
        remove_bg(tmp_masked_image, tmp_mask, mod_filename)
        end = datetime.datetime.now()
        elapsed_time = end - start
        datestamp = datetime.datetime.utcnow()
        output_path = "/"+args.output
        bgremoved_log_file.write("%s,%s,%s,%s\n" % (datestamp, fish, output_path, elapsed_time))
        elapsed_times.append(elapsed_time)
    bgremoved_log_file.close()
    return sum_time(elapsed_times)

def sum_time(elapsed_times):
    #Adapted from https://stackoverflow.com/questions/2780897/python-summing-up-time/28950926
    sum = datetime.timedelta()
    for time in elapsed_times:
        timestr = str(time)
        (h, m, s) = timestr.split(':')
        d = datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s))
        sum += d
    return sum

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove backgrounds from fish images.')
    parser.add_argument('--input', required=True, metavar="/input/path/to/fish/images", help='Directory of fish images to remove backgrounds from')
    parser.add_argument('--output', required=True, metavar="/output/path/for/background-removed/fish/images", help='Directory to place background-removed fish images in')

    args = parser.parse_args()

    assert args.input and args.output,\
        "Arguments --input and --output are required to remove backgrounds of fish images"

    print("Input Directory for Fish Images: ", args.input)
    print("Output Location for Background-Removed Fish Images: ", args.output)

#unpack
if not os.path.exists(WEIGHTS_PATH):
    download_model(MODEL_URL, WEIGHTS_PATH)
if not os.path.exists(os.path.join(ROOT_DIR, "fishseg", "train")):
    unpack_zip(TRAIN_ZIP_PATH, TRAIN_ZIP_FILE)
if not os.path.exists(os.path.join(ROOT_DIR, "fishseg", "val")):
    unpack_zip(VAL_ZIP_PATH, VAL_ZIP_FILE)

config, dataset = config_setup()

welcome_message()
welcome_fish()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("Loading fish segmentation model: ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Fish segmentation model successfully loaded from: ", WEIGHTS_PATH, "\n")

fish_files = get_fishes(args.input)
if len(fish_files) < 1:
    print("No image files found in specified input directory.")

total_elapsed_time = execute_fish_image_processing(model, fish_files)

exit_fish()
exit_message(total_elapsed_time)