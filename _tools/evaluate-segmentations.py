#!/usr/bin/env python3

# imports
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# define functions
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

def print_instructions():
    welcome_message()
    welcome_fish()
    print("Instructions: \n(1) Please specify the automatically segmented image from Sashimi that you would like to analyze (--image/-i).")
    print("(2) Please specify the corresponding .JSON evaluation segmentation mask automatically generated for Sashimi for the specified image in (--evalmask/-e).")
    print("\n\nOptional arguments:")
    print("* Please specify the corresponding .JSON ground truth segmentation mask you manually created for the specified image (--gtmask/-gt).")
    print("* Please specify the directory you would like to output the evaluation data to if desired (--output/-o).")
    print("==================================\n\n")

def parse_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return(np.asarray(data))

def get_groundtruth_contours(eval_mask, gt_mask = False):
    # get ground truth segmentation mask
    img_c = np.zeros((eval_mask[:,:,0].shape[0], eval_mask[:,:,0].shape[1]))

    if gt_mask is False:
        contours = "null"
        cont = "null"
        cont_scaled = "null"
    else:
        contours = np.array(gt_mask)
        cont = cv2.drawContours(img_c, [contours.astype(int)], 0, color = (255,255,255), thickness = -1)
        cont_scaled = np.where(cont == 255, 1, cont) # scale contours to binary array

    return(cont_scaled)

def plot_segmentations(img_path, eval_mask, gt_mask = False, saveplots = False, output_dir = "null"):
    # get ground truth segmentation mask
    img_c = np.zeros((eval_mask[:,:,0].shape[0], eval_mask[:,:,0].shape[1]))

    if gt_mask is False:
        contours = "null"
        cont = "null"
        cont_scaled = "null"
    else:
        contours = np.array(gt_mask)
        cont = cv2.drawContours(img_c, [contours.astype(int)], 0, color = (255,255,255), thickness = -1)
        cont_scaled = np.where(cont == 255, 1, cont) # scale contours to binary array

    # extract base file name for saving plots
    base = os.path.basename(img_path)
    fname = os.path.splitext(base)[0] + ".pdf"
    fname = os.path.join(output_dir, fname)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    if gt_mask is False:
        # plot of original source image
        plt.subplot(1,2,1)
        plt.title('Source Image: ' + base, weight="bold")
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # plot of evaluation segmentation mask for source image
        plt.subplot(1,2,2)
        plt.title('Model Predicted Segmentation Mask', weight="bold")
        plt.imshow(eval_mask[:,:,0])
    else:
        # plot of original source image
        plt.subplot(1,4,1)
        plt.title('Source Image: ' + base, weight="bold")
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # plot of evaluation segmentation mask for source image
        plt.subplot(1,4,2)
        plt.title('Model Predicted Segmentation Mask', weight="bold")
        plt.imshow(eval_mask[:,:,0], cmap = "gray")

        # plot of ground truth segmentation image plot
        plt.subplot(1,4,3)
        plt.title('Reference Segmentation Mask', weight="bold")
        plt.imshow(img_c, cmap = "gray")

        # plot of overlay between ground truth and model predicted mask
        plt.subplot(1,4,4)
        plt.title('Model Mask over Reference Mask', weight="bold")
        plt.imshow(img_c, cmap = "gray")
        plt.imshow(eval_mask[:,:,0], cmap = "gray", alpha=0.7)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)

def save_eval_data(eval_mask, gt_mask_scaled, img_path, eval_file, gt_file, saveplots = False, output_dir = "null"):
    # compute metrics
    pxacc = new_pxacc(eval_mask[:,:,0], gt_mask_scaled)
    pxacc_mean = new_mean_pxacc(eval_mask[:,:,0], gt_mask_scaled)
    IU_mean = new_mean_IoU(eval_mask[:,:,0], gt_mask_scaled)
    IU_freq_weighted = new_freq_weighted_IoU(eval_mask[:,:,0], gt_mask_scaled)

    # extract base file name for saving plots
    base = os.path.basename(img_path)
    fname = os.path.splitext(base)[0] + ".csv"
    fname = os.path.join(output_dir, fname)

    with open(os.path.join(output_dir, fname), 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n')
        writer.writerow(["source_image", "eval_image_id", "gt_image_id", "pixel_acc", "mean_acc", "mean_IU", "freq_weighted_IU"])

        row = [img_path, eval_file, gt_file, pxacc, pxacc_mean, IU_mean, IU_freq_weighted]
        writer.writerow(row)

## functions adapted from Martin Keršner's code: https://github.com/martinkersner/py_img_seg_eval
def extract_classes(mask):
    cl = np.unique(mask)
    n_cl = len(cl)
    
    return cl, n_cl

def union_classes(eval_mask, gt_mask):
    eval_cl, _ = extract_classes(eval_mask)
    gt_cl, _ = extract_classes(gt_mask)
    
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    
    return cl, n_cl
    
def get_pixel_area(mask):
    return(mask.shape[0] * mask.shape[1])

def extract_masks(mask, cl, n_cl):
    h, w = mask_size(mask)
    masks = np.zeros((n_cl, h, w))
    
    for i, c in enumerate(cl):
        masks[i, :, :] = mask == c
        
    return masks

def extract_both_masks(eval_mask, gt_mask, cl, n_cl):
    eval_mask_extracted = extract_masks(eval_mask, cl, n_cl)
    gt_mask_extracted = extract_masks(gt_mask, cl, n_cl)
    
    return eval_mask_extracted, gt_mask_extracted

def mask_size(mask):
    try:
        height = mask.shape[0]
        width = mask.shape[1]
    except IndexError:
        raise
    
    return height, width

def check_size(eval_mask, gt_mask):
    h_e, w_e = mask_size(eval_mask)
    h_g, w_g = mask_size(gt_mask)
    
    if(h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matricies!")
        
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

def new_pxacc(eval_mask, gt_mask):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''
    
    check_size(eval_mask, gt_mask)
    
    cl, n_cl = extract_classes(gt_mask)
    
    eval_mask_extracted, gt_mask_extracted = extract_both_masks(eval_mask,
                                                               gt_mask,
                                                               cl, n_cl)
    sum_n_ii = 0
    sum_t_i = 0
    
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask_extracted[i, :, :]
        curr_gt_mask = gt_mask_extracted[i, :, :]
        
        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
        
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i
        
    return pixel_accuracy_

def new_mean_pxacc(eval_mask, gt_mask):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''
    
    check_size(eval_mask, gt_mask)
    
    cl, n_cl = extract_classes(gt_mask)
    eval_mask_extracted, gt_mask_extracted = extract_both_masks(eval_mask,
                                                               gt_mask,
                                                               cl, n_cl)
    
    accuracy = list([0]) * n_cl
    
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask_extracted[i, :, :]
        curr_gt_mask = gt_mask_extracted[i, :, :]
        
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        
        if (t_i != 0):
            accuracy[i] = n_ii / t_i
            
    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def new_mean_IoU(eval_mask, gt_mask):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''
    
    check_size(eval_mask, gt_mask)
    
    cl, n_cl = union_classes(eval_mask, gt_mask)
    _, n_cl_gt = extract_classes(gt_mask)
    eval_mask_extracted, gt_mask_extracted = extract_both_masks(eval_mask,
                                                               gt_mask,
                                                               cl, n_cl)
    
    IU = list([0] * n_cl)
    
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask_extracted[i, :, :]
        curr_gt_mask = gt_mask_extracted[i, :, :]
        
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        
        IU[i] = n_ii / (t_i + n_ij - n_ii)
        
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def new_freq_weighted_IoU(eval_mask, gt_mask):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''
    
    check_size(eval_mask, gt_mask)
    
    cl, n_cl = union_classes(eval_mask, gt_mask)
    eval_mask_extracted, gt_mask_extracted = extract_both_masks(eval_mask,
                                                               gt_mask,
                                                               cl, n_cl)
    
    frequency_weighted_IU_ = list([0]) * n_cl
    
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask_extracted[i, :, :]
        curr_gt_mask = gt_mask_extracted[i, :, :]
        
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
            
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        
        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
        
    sum_k_t_k = get_pixel_area(eval_mask)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

# main
if __name__ == '__main__':
    import argparse

    print_instructions()

    parser = argparse.ArgumentParser(description='Visualize and Evaluate Sashimi Segmentation Masks')
    parser.add_argument('--image', '-i', required=True, metavar='/input/path/to/segmented/image', help='Path of segmented image.')
    parser.add_argument('--evalmask', '-e', required=True, metavar='/input/path/to/evaluation/mask', help='Path of automatically generated evaluation segmentatoin mask from Sashimi JSON.')
    parser.add_argument('--gtmask', '-gt', required=False, default=False, metavar='/input/path/to/ground-truth/mask', help='Path of ground-truth segmentation mask JSON.')
    parser.add_argument('--output', '-o', required=False, metavar='/output/path/to/save/diagnostics', help='Path to save diagnostics if desired.')
    parser.add_argument('--verbose', '-v', default=True, help="Set to False to hide verbose console output and plotting windows.")

    args = parser.parse_args()

    assert args.image and args.evalmask,\
        "Arguments --image/-i and --evalmask/-ei are required at minimum to evaluate Sashimi segmented image mask."

    # make output dir if it doesn't already exist
    if not os.path.exists(os.path.join(ROOT_DIR, args.output)):
        os.mkdir(os.path.join(ROOT_DIR, args.output))

    # parse JSON
    eval_mask = parse_json(args.evalmask)

    gt_mask = args.gtmask
    if gt_mask is False:
        gt_mask_scaled = False
    else:
        gt_mask = parse_json(args.gtmask)
        gt_mask_scaled = get_groundtruth_contours(eval_mask, gt_mask)
    
    # plot data viz
    plot_segmentations(args.image, eval_mask, gt_mask, args.verbose, os.path.join(ROOT_DIR, args.output))

    # save evaluation metrics
    if gt_mask_scaled is not False:
        save_eval_data(eval_mask, gt_mask_scaled, args.image, args.evalmask, args.gtmask, args.verbose, os.path.join(ROOT_DIR, args.output))

    # all done!
    print("Done!")
