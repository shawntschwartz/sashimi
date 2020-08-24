#!/usr/bin/env python3

# Created by Shawn Schwartz 11/07/2019
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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
print("Initializing Setup...Please wait...")

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") #download file from specified URL in sashimi GitHub README.md
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class sashimiConfig(Config):
    NAME = "fish" #update with name of organism you'd like to train a model for
    IMAGES_PER_GPU = 1 #keep at 1 if you have 1 GPU
    NUM_CLASSES = 1 + 1 
    STEPS_PER_EPOCH = 100 #can change to suite your image processing hardware setup
    DETECTION_MIN_CONFIDENCE = 0.9

class sashimiDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class(sashimiConfig.NAME, 1, sashimiConfig.NAME)

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(os.path.join(dataset_dir, "_fish-segmentation-regions.json"))) #modify JSON file path with your generated file from the sashimi web-interface instructions
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(sashimiConfig.NAME, image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)
        
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != sashimiConfig.NAME:
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == sashimiConfig.NAME:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    dataset_train = sashimiDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = sashimiDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads') #can modify number of epochs if desired

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
        config = sashimiConfig()
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