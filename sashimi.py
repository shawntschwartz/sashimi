#!/usr/bin/env python3

'''
Sashimi: A toolkit for faciliating high-throughput 
organismal image segmentation using deep learning
=========================================================

Authors: Shawn T. Schwartz & Michael E. Alfaro
Email: shawnschwartz@ucla.edu
Homepage: https://shawnschwartz.com
'''

import os
import sys
import skimage.io
import numpy as np
import datetime
import urllib.request
import shutil
import json
from PIL import Image
from sashimi import core
from sashimi.pycocotools import coco
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import warnings
warnings.filterwarnings("ignore")

# paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "_models")
COCO_WEIGHTS = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
OUTPUT_HOME = os.path.join(ROOT_DIR, "_outputs")

# default fish model url
MODEL_URL = "https://github.com/ShawnTylerSchwartz/sashimi/releases/download/v1.0.0/fish_segmentation_model_schwartz_v1-0-0.h5"
COCO_URL = "https://github.com/ShawnTylerSchwartz/sashimi/releases/download/v1.0.0/mask_rcnn_coco.h5"

# Define functions
def welcome_message():
    print("\n\nSashimi [Version 1.0.0] built by Shawn T. Schwartz at Alfaro Lab, UCLA.")
    print("Contact: shawnschwartz@ucla.edu")
    print("Website: https://shawnschwartz.com")
    print("Github: @ShawnTylerSchwartz")
    print("Twitter: @shawnschwartz_")

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

def download_model(URL, destination):
    print("Downloading required model to", destination, "...")
    with urllib.request.urlopen(URL) as resp, open(destination, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("Model successfully downloaded to models/.")

def get_images(dir):
    desired_exts = ['jpg', 'jpeg', 'png']
    return [f for f in os.listdir(dir) if any(f.endswith(ext) for ext in desired_exts)]

def load_img(path):
    print("Reading organism: ", path)
    return skimage.io.imread(os.path.join(ROOT_DIR, args.input, path))

def detect_mask(model, img):
    return model.detect([img], verbose=0)

def draw_mask(img, mask, organism):
    tmp_result = mask[0]

    if(len(tmp_result['scores']) > 1):
        print("Detected > 1 target... selecting most prominent target.")
        tmp_mask = tmp_result['masks']
        #print(tmp_mask.shape)
    else:
        print("Detected 1 target.")
        tmp_mask = tmp_result['masks']
        #print(tmp_mask.shape)

    tmp_mask = tmp_mask.astype(int)
    #tmp_img_mtrx_mask = img

    #all_recovered_masks = np.array(tmp_img_mtrx_mask)
    all_recovered_masks = []
    print("mask shape [2]: ", tmp_mask.shape[2])
    if(tmp_mask.shape[2] > 0):
        #for ii in range(tmp_img_mtrx_mask.shape[2]):
        for ii in range(tmp_mask.shape[2]):
            #tmp_img_mtrx_mask = img
            #for jj in range(tmp_mask.shape[2]):
            tmp_img_mtrx_mask = load_img(organism)
            for jj in range(tmp_img_mtrx_mask.shape[2]):
                ##tmp_img_mtrx_mask[:,:,ii] = tmp_img_mtrx_mask[:,:,ii] * tmp_mask[:,:,jj]
                tmp_img_mtrx_mask[:,:,jj] = tmp_img_mtrx_mask[:,:,jj] * tmp_mask[:,:,ii]
                #all_recovered_masks = np.append(all_recovered_masks, tmp_img_mtrx_mask)
            all_recovered_masks.append(tmp_img_mtrx_mask)
        print("Length of scores: ", len(tmp_result['scores']))
        all_recovered_masks = np.stack(all_recovered_masks, axis=0)
        print("Number of masks recovered (of np stacked version): ", all_recovered_masks.shape)
    else:
        print("Could not draw reliable mask... skipping.")
        return "NULL", "NULL"
        
    ##return tmp_img_mtrx_mask, tmp_mask
    #all_recovered_masks = np.delete(all_recovered_masks, 0)
    return all_recovered_masks, tmp_mask

def remove_bg(img, mask, path):
    print("---------------------------")
    print("Number of masks to remove and save: ", mask.shape[2])
    print("Number of images imported: ")
    print(img[0].shape)
    if(mask.shape[2] == 1):
        height, width = img[0].shape[:2]
        print(img[0].shape)
        alpha_img = np.dstack((img[0], np.zeros((height, width), dtype=np.uint8)+255))
        alpha_img[:,:,3] = alpha_img[:,:,3] * mask[:,:,0]
        print("Background successfully removed!")
        save_image(alpha_img, path)
    else:
        # for ii in range(mask.shape[2]):
        #     height, width = img[ii].shape[:2]
        #     alpha_img = np.dstack((img[ii], np.zeros((height, width), dtype=np.uint8)+255))
        #     for jj in range(alpha_img.shape[2]):
        #         alpha_img[:,:,jj] = alpha_img[:,:,jj] * mask[:,:,ii]
        #         print("Background successfully removed for image: ", (ii+1))
        #         path_mod = (path + "_" + str(ii+1))
        #         save_image(alpha_img, path_mod)
        # for ii in range(mask.shape[2]):
        #     height, width = img[ii].shape[:2]
        #     alpha_img = np.dstack((img[ii], np.zeros((height, width), dtype=np.uint8)+255))
        #     print("Alpha image shape:", alpha_img.shape)
        #     for jj in range(alpha_img.shape[2]):
        #         alpha_img[:,:,3] = alpha_img[:,:,3] * mask[:,:,ii]
        #         print("Background successfully removed for image: ", (ii+1))
        #         path_mod = (path + "_" + str(ii+1))
        #         save_image(alpha_img, path_mod)
        for ii in range(mask.shape[2]):
            height, width = img[ii].shape[:2]
            alpha_img = np.dstack((img[ii], np.zeros((height, width), dtype=np.uint8)+255))
            alpha_img[:,:,3] = alpha_img[:,:,3] * mask[:,:,ii]
            print("Background successfully removed for image: ", (ii+1))
            path_mod = (path + "_" + str(ii+1))
            save_image(alpha_img, path_mod)

def save_image(img, path):
    bg_removed_img = Image.fromarray(img, 'RGBA')
    bg_removed_img.save((os.path.join(OUTPUT_HOME, args.output, path+".png")), "PNG")
    print("Successfully saved as ===> ",path,".png  in", OUTPUT_HOME, "/",args.output,"\n",sep="")

def alpha_composite(src, dst, path):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object
    path -- Name of Organism File

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
    save_image(out, path)
    print("Color filled organism successfully generated!")

def fill_bg(img, mask, path, r, g, b):
    red = int(r * 255)
    green = int(g * 255)
    blue = int(b * 255)
    height, width = img.shape[:2]
    alpha_img = np.dstack((img, np.zeros((height, width), dtype=np.uint8)+255))
    alpha_img[:,:,3] = alpha_img[:,:,3] * mask[:,:,0]
    alpha_img_img = Image.fromarray(alpha_img, 'RGBA')
    filled_bg_img = Image.new('RGBA', size = alpha_img_img.size, color = (red, green, blue, 255))
    alpha_composite(alpha_img_img, filled_bg_img, path)

def fill_bg_only(img, path, r, g, b):
    red = int(r * 255)
    green = int(g * 255)
    blue = int(b * 255)
    img = Image.fromarray(img, 'RGBA')
    filled_bg_img = Image.new('RGBA', size = img.size, color = (red, green, blue, 255))
    alpha_composite(img, filled_bg_img, path)

def make_silhouette(img, mask, path):
    height, width = img.shape[:2]
    alpha_img = np.dstack((img, np.zeros((height, width), dtype=np.uint8)+255))
    alpha_img[:,:,3] = alpha_img[:,:,3] * mask[:,:,0]
    r, g, b, a = np.rollaxis(alpha_img, axis = -1)
    r[a == 255] = 0
    g[a == 255] = 0
    b[a == 255] = 0
    silhouette = np.dstack([r, g, b, a])
    print("Silhouette of organism successfully generated!")
    save_image(silhouette, path)

def execute_image_processing(model, image_set):
    if not os.path.exists(os.path.join(ROOT_DIR, "_logs")):
        os.mkdir(os.path.join(ROOT_DIR, "_logs"))
    if not os.path.exists(os.path.join(OUTPUT_HOME, args.output)):
        os.mkdir(os.path.join(OUTPUT_HOME, args.output))
    if not os.path.exists(os.path.join(ROOT_DIR, "_logs", "eval_mask_json_"+args.organism)):
        os.mkdir(os.path.join(ROOT_DIR, "_logs", "eval_mask_json_"+args.organism))

    elapsed_times = []
    bgremoved_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_segmentation.csv'
    bgremoved_log_file = open(os.path.join(ROOT_DIR, "_logs", bgremoved_filename), 'w')
    for counter, organism in enumerate(image_set, start=1):
        start = datetime.datetime.now()
        if((args.segmentation is None) or (args.segmentation == 1)):
            loaded_organism = load_img(organism)
            print("Segmenting (",counter,"/",len(image_set),"): ",organism,sep="")
            organism_detection_result = detect_mask(model, loaded_organism)
            tmp_masked_image, tmp_mask = draw_mask(loaded_organism, organism_detection_result, organism)

            if((tmp_masked_image == "NULL") & (tmp_mask == "NULL")):
                continue
        else:
            loaded_organism = load_img(organism)
            print("Filling background of (",counter,"/",len(image_set),"): ",organism,sep="")
        
        mod_filename = os.path.splitext(organism)[0]
        
        # check if color args are passed
        if((args.red is not None and args.green is not None and args.blue is not None) and (args.segmentation == 1 or args.segmentation is None)):
            fill_bg(tmp_masked_image, tmp_mask, mod_filename, args.red, args.green, args.blue)
        elif((args.red is not None and args.green is not None and args.blue is not None) and (args.segmentation == 0)):
            fill_bg_only(loaded_organism, mod_filename, args.red, args.green, args.blue)
        elif(args.silhouette == 1 and (args.segmentation is None or args.segmentation == 1)):
            make_silhouette(tmp_masked_image, tmp_mask, mod_filename)
        else:
            remove_bg(tmp_masked_image, tmp_mask, mod_filename)
            mask_values_file = open(os.path.join(ROOT_DIR, "_logs", "eval_mask_json_"+args.organism, mod_filename + ".json"), 'w')
            tmp_mask_list = tmp_mask.tolist()
            json_str = json.dumps(tmp_mask_list)
            mask_values_file.write(json_str)
            mask_values_file.close()
        end = datetime.datetime.now()
        elapsed_time = end - start
        datestamp = datetime.datetime.utcnow()
        output_path = "/"+args.output
        bgremoved_log_file.write("%s,%s,%s,%s\n" % (datestamp, organism, output_path, elapsed_time))
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

def config_setup(organism, regions, imgsrc):

    config = core.CocoConfig()
    dataset = core.CocoDataset()

    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NAME = organism
        REGIONS = regions

    config = InferenceConfig()
    config.display()

    dataset.load_custom(DATASET_DIR, "val", organism, regions, imgsrc)

    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    return config, dataset

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rapidly remove backgrounds from organimal images with Mask R-CNN.')
    parser.add_argument('--input', '-i', required=True, metavar="/input/path/to/images", help='Directory of images to remove backgrounds from')
    parser.add_argument('--output', '-o', required=True, metavar="/output/path/for/background-removed/images", help='Directory to place background-removed images in')
    parser.add_argument('--red', '-r', required=False, type=float, metavar="0", help='(r)ed intensity values (0 to 1) for background mask')
    parser.add_argument('--green', '-g', required=False, type=float, metavar="0.4", help='(g)reen intensity values (0 to 1) for background mask')
    parser.add_argument('--blue', '-b', required=False, type=float, metavar="0", help='(b)lue intensity values (0 to 1) for background mask')
    parser.add_argument('--silhouette', '-s', required=False, type=int, metavar="1", help='set to 1 (true) to make a silhouette of the segmented image')
    parser.add_argument('--segmentation', '-z', required=False, type=int, metavar="0", help='set to 0 (false) to fill backgrounds of previously segmented images with desired background colors')
    parser.add_argument('--model', '-m', required=False, default=os.path.join(MODEL_DIR, "fish_segmentation_model_schwartz_v1-0-0.h5"), metavar="/path/to/custom/model.h5", help='Path to custom segmentation model to use (defaults to models/fish_segmentation_model_schwartz_v1-0-0.h5)')
    parser.add_argument('--organism', '-n', required=False, default="fish", metavar="name of class label to detect", help="String with name of class label to detect, should match that used in training data. (Defaults to 'fish' for default fish segmentation model.) You can specify a label present in the COCO dataset along with setting --coco/-c to True to utilize the pre-trained weights in COCO.")
    parser.add_argument('--regions', '-y', required=False, default="_fish-segmentation-regions.json", metavar="Name of manual segmentation regions file.", help="Should be placed in both train/ and val/ folders, and should have the same exact filename within each of those two folders. (Defaults to _fish-segmentation-regions.json)")
    parser.add_argument('--imgsrc', '-j', required=False, default="url", metavar="Where are source images located ('local' or 'url')", help="Specify 'local' or 'url' depending on how paths are provided in your regions .JSON file. Important for distribution purposes. Default is URL -- images will be downloaded from source initially, and won't download again unless directory no longer exists. Use 'local' if images are stored directly in the respective train/ and val/ directories.")
    
    args = parser.parse_args()

    assert args.input and args.output,\
        "Arguments --input and --output are required to remove backgrounds of fish images"

    print("Input Directory for Images to Segment: ", args.input)
    print("Output Location for Background-Removed Images: ", args.output)

    DATASET_DIR = os.path.join(ROOT_DIR, "sashimi", "fish")

    if not os.path.exists(OUTPUT_HOME):
        os.mkdir(OUTPUT_HOME)

    if not os.path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)
    if not os.path.exists(os.path.join(DATASET_DIR, "train")):
        os.mkdir(os.path.join(DATASET_DIR, "train"))
    if not os.path.exists(os.path.join(DATASET_DIR, "val")):
        os.mkdir(os.path.join(DATASET_DIR, "val"))

    # download and unpack necessary data
    WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(WEIGHTS_PATH):
        download_model(COCO_URL, WEIGHTS_PATH)

    WEIGHTS_PATH = args.model
    if not os.path.exists(WEIGHTS_PATH):
        download_model(MODEL_URL, WEIGHTS_PATH)     

    config, dataset = config_setup(args.organism, args.regions, args.imgsrc)

    welcome_message()
    welcome_fish()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    print("Loading segmentation model: ", WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH, by_name=True)
    print("Segmentation model successfully loaded from: ", WEIGHTS_PATH, "\n")

    organism_images = get_images(args.input)
    if len(organism_images) < 1:
        print("No image files found in specified input directory.")

    total_elapsed_time = execute_image_processing(model, organism_images)

    exit_fish()
    exit_message(total_elapsed_time)