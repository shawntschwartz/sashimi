#!/usr/bin/env python3

# sashimi Version 1.0.0
# Created by Shawn Schwartz 11/07/2019
# <shawnschwartz@ucla.edu>

version = "sashimi"
build = "v1.0.0 - 20191231"

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
DATASET_DIR = os.path.join(ROOT_DIR, "sashimi")

MODEL_URL = "https://github.com/ShawnTylerSchwartz/sashimi/releases/download/V1.0/fish_segmentation_model_schwartz_v1.h5"
TRAIN_ZIP_FILE = "train.zip"
VAL_ZIP_FILE = "val.zip"
TRAIN_ZIP_PATH = os.path.join(ROOT_DIR, "sashimi")
VAL_ZIP_PATH = os.path.join(ROOT_DIR, "sashimi")

def welcome_message():
    print("sashimi [Version 1.0.0] built by Shawn T. Schwartz (2019) at Alfaro Lab, UCLA.")
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
    print("Downloading demo fish segmentation model to",destination,"...")
    with urllib.request.urlopen(URL) as resp, open(destination, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("Demo fish segmentation model successfully downloaded!")

def unpack_zip(path, file):
    print("Unpacking asset at: ",path,"/",file,sep="")
    zip = ZipFile(os.path.join(path, file))
    zip.extractall(path)
    zip.close()

def get_fishes(dir):
    desired_exts = ['jpg', 'jpeg', 'png']
    return [f for f in os.listdir(dir) if any(f.endswith(ext) for ext in desired_exts)]

def get_fish_img(fish_file):
    print("Loading Image File: ", fish_file)
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

def alpha_composite(src, dst, fish_name):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object
    fish_name -- Name of Fish File

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # Modified from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil/9166671#9166671
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    src = np.asarray(src)
    dst = np.asarray(dst)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)    
    out[alpha] *= 255
    np.clip(out,0,255)
    out = out.astype('uint8')
    save_fish(out, fish_name)
    print("Color-filled image successfully generated!")

def fill_bg(fish_img, mask, fish_name, r, g, b):
    red = int(r * 255)
    green = int(g * 255)
    blue = int(b * 255)
    height, width = fish_img.shape[:2]
    alpha_fish = np.dstack((fish_img, np.zeros((height, width), dtype=np.uint8)+255))
    alpha_fish[:,:,3] = alpha_fish[:,:,3] * mask[:,:,0]
    alpha_fish_img = Image.fromarray(alpha_fish, 'RGBA')
    filled_bg_img = Image.new('RGBA', size = alpha_fish_img.size, color = (red, green, blue, 255))
    alpha_composite(alpha_fish_img, filled_bg_img, fish_name)

def fill_bg_only(fish_img, fish_name, r, g, b):
    red = int(r * 255)
    green = int(g * 255)
    blue = int(b * 255)
    fish_img = Image.fromarray(fish_img, 'RGBA')
    filled_bg_img = Image.new('RGBA', size = fish_img.size, color = (red, green, blue, 255))
    alpha_composite(fish_img, filled_bg_img, fish_name)

def make_silhouette(fish_img, mask, fish_name):
    height, width = fish_img.shape[:2]
    alpha_fish = np.dstack((fish_img, np.zeros((height, width), dtype=np.uint8)+255))
    alpha_fish[:,:,3] = alpha_fish[:,:,3] * mask[:,:,0]
    r, g, b, a = np.rollaxis(alpha_fish, axis = -1)
    r[a == 255] = 0
    g[a == 255] = 0
    b[a == 255] = 0
    silhouette_fish = np.dstack([r, g, b, a])
    print("Silhouette image successfully generated!")
    save_fish(silhouette_fish, fish_name)

def save_fish(fish_img, fish_name):
    bg_removed_img = Image.fromarray(fish_img, 'RGBA')
    bg_removed_img.save((os.path.join(ROOT_DIR, args.output, fish_name+".png")), "PNG")
    print("Output image successfully saved as ===> ",fish_name,".png  in  /",args.output,"\n",sep="")

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
        if((args.segmentation is None) or (args.segmentation == 1)):
            loaded_fish = get_fish_img(fish)
            print("Segmenting (",counter,"/",len(fish_set),"): ",fish,sep="")
            fish_detection_result = detect_mask(model, loaded_fish)
            tmp_masked_image, tmp_mask = draw_mask(loaded_fish, fish_detection_result)
        else:
            #loaded_fish = Image.open(fish).convert('RGBA')
            loaded_fish = get_fish_img(fish)
            print("Filling background of (",counter,"/",len(fish_set),"): ",fish,sep="")
        mod_filename = fish.split('.')
        mod_filename = mod_filename[0]
        #check if color args are passed
        if((args.red is not None and args.green is not None and args.blue is not None) and (args.segmentation == 1 or args.segmentation is None)):
            fill_bg(tmp_masked_image, tmp_mask, mod_filename, args.red, args.green, args.blue)
        elif((args.red is not None and args.green is not None and args.blue is not None) and (args.segmentation == 0)):
            fill_bg_only(loaded_fish, mod_filename, args.red, args.green, args.blue)
        elif(args.silhouette == 1):
            make_silhouette(tmp_masked_image, tmp_mask, mod_filename)
        else:
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

    parser = argparse.ArgumentParser(description='Remove backgrounds from organismal images.')
    parser.add_argument('--input', '-i', required=True, metavar="/input/path/to/organismal/images", help='Directory of organismal images to remove backgrounds from')
    parser.add_argument('--output', '-o', required=True, metavar="/output/path/for/background-removed/organismal/images", help='Directory to place background-removed organismal images in')
    parser.add_argument('--red', '-r', required=False, type=float, metavar="0", help='(r)ed intensity values (0 to 1) for colordistance background mask')
    parser.add_argument('--green', '-g', required=False, type=float, metavar="0.4", help='(g)reen intensity values (0 to 1) for colordistance background mask')
    parser.add_argument('--blue', '-b', required=False, type=float, metavar="0", help='(b)lue intensity values (0 to 1) for colordistance background mask')
    parser.add_argument('--silhouette', '-s', required=False, type=int, metavar="1", help='set to 1 (true) to make a silhouette of the segmented organismal image')
    parser.add_argument('--segmentation', '-z', required=False, type=int, metavar="0", help='set to 0 (false) to fill backgrounds of previously segmented organismal images with desired background colors')

    args = parser.parse_args()

    assert args.input and args.output,\
        "Arguments --input and --output are required to remove backgrounds of organismal images"

    print("Input Directory for Organismal Images: ", args.input)
    print("Output Location for Background-Removed Organismal Images: ", args.output)

#unpack
if not os.path.exists(WEIGHTS_PATH):
    download_model(MODEL_URL, WEIGHTS_PATH)
if not os.path.exists(os.path.join(ROOT_DIR, "sashimi", "train")):
    unpack_zip(TRAIN_ZIP_PATH, TRAIN_ZIP_FILE)
if not os.path.exists(os.path.join(ROOT_DIR, "sashimi", "val")):
    unpack_zip(VAL_ZIP_PATH, VAL_ZIP_FILE)

config, dataset = config_setup()

welcome_message()
welcome_fish()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("Loading demo fish segmentation model: ", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Demo fish segmentation model successfully loaded from: ", WEIGHTS_PATH, "\n")

fish_files = get_fishes(args.input)
if len(fish_files) < 1:
    print("No image files found in specified input directory.")

total_elapsed_time = execute_fish_image_processing(model, fish_files)

exit_fish()
exit_message(total_elapsed_time)