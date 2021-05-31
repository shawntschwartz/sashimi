#!/usr/bin/env python3

# Created by Shawn Schwartz
# <shawnschwartz@ucla.edu>
#
# sashimi ~> core.py
#
# Adapted from Mask R-CNN by @matterport on Github
# https://github.com/matterport/Mask_RCNN

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import imgaug

from pycocotools.coco import COCO

import shutil
import zipfile
import urllib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "_models")
COCO_WEIGHTS = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
print("Initializing Setup... Please wait...")

COCO_WEIGHTS_PATH = COCO_WEIGHTS # download file from specified URL in sashimi GitHub README.md
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"

    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

DEFAULT_DATASET_YEAR = "2014"

class CocoDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, imgsrc):
        self.add_class(sashimiConfig.NAME, 1, sashimiConfig.NAME)

        print("Detecting class of: ", sashimiConfig.NAME)

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(os.path.join(dataset_dir, sashimiConfig.REGIONS))) #modify JSON file path with your generated file from the sashimi web-interface instructions
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]

        if not os.path.exists(os.path.join(ROOT_DIR, "logs")):
            os.mkdir(os.path.join(ROOT_DIR, "logs"))
        if not os.path.exists(os.path.join(ROOT_DIR, "logs", "gt_mask_json_" + sashimiConfig.NAME)):
            os.mkdir(os.path.join(ROOT_DIR, "logs", "gt_mask_json_" + sashimiConfig.NAME))

        for idx, a in enumerate(annotations):
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            
            if imgsrc == "url":
                image_path = a['filename']
                # download source images
                if not os.path.exists(os.path.join(dataset_dir, os.path.basename(image_path))):
                    with urllib.request.urlopen(image_path) as resp, open(os.path.join(dataset_dir, os.path.basename(image_path)), 'wb') as out:
                        shutil.copyfileobj(resp, out)
                    print("... done downloading " + os.path.basename(image_path))
                else:
                    print(os.path.basename(image_path) + " already exists. Skipping download...")
                image_path = os.path.join(dataset_dir, os.path.basename(image_path))
            elif imgsrc == "local":
                image_path = os.path.join(dataset_dir, a['filename'])

            #print(image_path)
    
            image_path_mod = os.path.splitext(os.path.basename(image_path))[0]
            #print(image_path_mod[0])
            
            #print(ROOT_DIR)
            if not os.path.exists(os.path.join(ROOT_DIR, "logs", "gt_mask_json_" + sashimiConfig.NAME, image_path_mod + ".json")):
                gt_mask_values_file = open(os.path.join(ROOT_DIR, "logs", "gt_mask_json_" + sashimiConfig.NAME, image_path_mod + ".json"), 'w')
            #gt_mask_values_file = open(image_path_mod + ".json")
                for aa in polygons:    
                    tmp_coordinates = list(zip(aa['all_points_x'], aa['all_points_y']))
                    #print(tmp_coordinates)
                    json_str = json.dumps(tmp_coordinates)
                    #print(json_str)
                    gt_mask_values_file.write(json_str)
                gt_mask_values_file.close()
            
                print("Parsing (", imgsrc, ") image data from: ", image_path, " (", idx + 1, "/", len(annotations), ")")
            else:
                print(image_path_mod + ".json already exists. Skipping parsing...")

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(CocoConfig.NAME, image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)

            def load_mask(self, image_id):
                image_info = self.image_info[image_id]
                if image_info["source"] != "fish":
                    return super(self.__class__, self).load_mask(image_id)

                info = self.image_info[image_id]
                mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

                for i, p in enumerate(info["polygons"]):
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                    mask[rr, cc, i] = 1
                
                return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

                def image_reference(self, image_id):
                    info = self.image_info[image_id]
                    if info["source"] == "fish":
                        return info["path"]
                    else:
                        super(self.__class__, self).image_reference(image_id)

            def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
                """Load a subset of the COCO dataset.
                dataset_dir: The root directory of the COCO dataset.
                subset: What to load (train, val, minival, valminusminival)
                year: What dataset year to load (2014, 2017) as a string, not an integer
                class_ids: If provided, only loads images that have the given classes.
                class_map: TODO: Not implemented yet. Supports maping classes from
                    different datasets to the same class ID.
                return_coco: If True, returns the COCO object.
                auto_download: Automatically download and unzip MS-COCO images and annotations
                """

                if auto_download is True:
                    self.auto_download(dataset_dir, subset, year)

                coco = COCO("{}\\annotations\instances_{}{}.json".format(dataset_dir, subset, year))
                if subset == "minival" or subset == "valminusminival":
                    subset = "val"
                image_dir = "{}/{}{}".format(dataset_dir, subset, year)

                # Load all classes or a subset?
                if not class_ids:
                    # All classes
                    class_ids = sorted(coco.getCatIds())

                # All images or a subset?
                if class_ids:
                    image_ids = []
                    for id in class_ids:
                        image_ids.extend(list(coco.getImgIds(catIds=[id])))
                    # Remove duplicates
                    image_ids = list(set(image_ids))
                else:
                    # All images
                    image_ids = list(coco.imgs.keys())

                # Add classes
                for i in class_ids:
                    self.add_class("coco", i, coco.loadCats(i)[0]["name"])

                # Add images
                for i in image_ids:
                    self.add_image(
                        "coco", image_id=i,
                        path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                        width=coco.imgs[i]["width"],
                        height=coco.imgs[i]["height"],
                        annotations=coco.loadAnns(coco.getAnnIds(
                            imgIds=[i], catIds=class_ids, iscrowd=None)))
                if return_coco:
                    return coco

            def auto_download(self, dataDir, dataType, dataYear):
                """Download the COCO dataset/annotations if requested.
                dataDir: The root directory of the COCO dataset.
                dataType: What to load (train, val, minival, valminusminival)
                dataYear: What dataset year to load (2014, 2017) as a string, not an integer
                Note:
                    For 2014, use "train", "val", "minival", or "valminusminival"
                    For 2017, only "train" and "val" annotations are available
                """

                # Setup paths and file names
                if dataType == "minival" or dataType == "valminusminival":
                    imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
                    imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
                    imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
                else:
                    imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
                    imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
                    imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
                # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

                # Create main folder if it doesn't exist yet
                if not os.path.exists(dataDir):
                    os.makedirs(dataDir)

                # Download images if not available locally
                if not os.path.exists(imgDir):
                    os.makedirs(imgDir)
                    print("Downloading images to " + imgZipFile + " ...")
                    with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                        shutil.copyfileobj(resp, out)
                    print("... done downloading.")
                    print("Unzipping " + imgZipFile)
                    with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                        zip_ref.extractall(dataDir)
                    print("... done unzipping")
                print("Will use images in " + imgDir)

                # Setup annotations data paths
                annDir = "{}/annotations".format(dataDir)
                if dataType == "minival":
                    annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
                    annFile = "{}/instances_minival2014.json".format(annDir)
                    annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
                    unZipDir = annDir
                elif dataType == "valminusminival":
                    annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
                    annFile = "{}/instances_valminusminival2014.json".format(annDir)
                    annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
                    unZipDir = annDir
                else:
                    annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
                    annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
                    annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
                    unZipDir = dataDir
                # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

                # Download annotations if not available locally
                if not os.path.exists(annDir):
                    os.makedirs(annDir)
                if not os.path.exists(annFile):
                    if not os.path.exists(annZipFile):
                        print("Downloading zipped annotations to " + annZipFile + " ...")
                        with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                            shutil.copyfileobj(resp, out)
                        print("... done downloading.")
                    print("Unzipping " + annZipFile)
                    with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                        zip_ref.extractall(unZipDir)
                    print("... done unzipping")
                print("Will use annotations in " + annFile)
 
class sashimiConfig(Config):
    NAME = "fish" #update with name of organism you'd like to train a model for
    IMAGES_PER_GPU = 1 #keep at 1 if you have 1 GPU
    NUM_CLASSES = 1 + 1 
    STEPS_PER_EPOCH = 100 #can change to suite your image processing hardware setup
    DETECTION_MIN_CONFIDENCE = 0.9
    REGIONS = "_fish-segmentation-regions.json"

def train(model, imgsrc):
    
    dataset_train = CocoDataset()
    dataset_train.load_custom(args.dataset, "train", imgsrc)
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_custom(args.dataset, "val", imgsrc)
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network heads - STAGE 1:")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up - STAGE 2:")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers - STAGE 3")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)

#TRAINING
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command", metavar="<command>", help="'train'")
    parser.add_argument('--dataset', required=False, metavar="/path/to/custom/dataset", help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command == "train":
        #config = sashimiConfig()
        config = CocoConfig()
    else:
        class InferenceConfig(sashimiConfig):
            GPU_COUNT = 1 #keep at 1 if you have 1 GPU
            IMAGES_PER_GPU = 1 #keep at 1 if you have 1 GPU
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. Use 'train'".format(args.command))