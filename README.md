# *Sashimi*: A toolkit for facilitating high-throughput organismal image segmentation using deep learning

Shawn T. Schwartz* and Michael E. Alfaro

Department of Ecology and Evolutionary Biology, University of California, Los Angeles

*Correspondence concerning this article should be addressed to Shawn Schwartz, Terasaki Life Sciences Building 2153, University of California, Los Angeles, CA 90095, USA. Email: shawnschwartz@ucla.edu

**APPLICATION PAPER UNDER REVIEW**

## Abstract
1. One crucial image preprocessing step in any image analysis workflow is image segmentation, or the ability to clearly contrast the foreground target from the background noise in an image. This procedure is typically done manually, imposing a severe bottleneck for the acquisition and extraction of meaningful data from images. Deep learning offers enormous potential for high-throughput phenoscaping in biological studies, and advances in image segmentation meta-algorithms for deep learning provide an opportunity to narrow this data processing bottleneck for studies in biology. However, the most used convolutional neural networks for deep learning image segmentation have been trained on a small fraction of biodiversity, severely limiting their application to image processing problems in ecology and evolutionary biology.
2. We trained a deep learning model to automatically segment target fish images from images with standardized and complex, noisy backgrounds. We then assessed the performance of our deep learning model using qualitative visual inspection and quantitative image segmentation metrics of pixel overlap between reference segmentation masks generated manually by experts and those automatically predicted by the model we trained.
3. Our model removed background pixels with 95% accuracy. Visual inspection revealed that our model segmented fishes with high precision and relatively few artifacts. These results suggest that the current model for fish image segmentation presented here is well-suited for producing high-fidelity segmented specimen images across a variety of background contexts at a rapid pace.
4. We present Sashimi, an easy-to-use command line toolkit to facilitate rapid, automated high-throughput image segmentation of digitized organisms. Sashimi requires no extensive programming nor deep learning implementation experience to use. Out-of-the-box, Sashimi is prepared to automatically segment fish images; however, it can be extended to additional organisms as well. Accordingly, we present a central database for sharing and distributing custom trained segmentation models of other unrepresented organisms, in addition to auxiliary image preprocessing functions for popular downstream color pattern analysis workflows. Lastly, Sashimi includes a simple script to assess qualitatively and quantitatively assess model segmentation performance for images both automatically segmented with deep learning and manually segmented by hand.

## Summary
We present a software module written in *Python* (Version 3.7.3) that is accessible by cloning the GitHub repository [https://github.com/ShawnTylerSchwartz/sashimi](https://github.com/ShawnTylerSchwartz/sashimi). Training and inspiration based on Matterport's open-source Keras and TensorFlow implementation of Mask R-CNN [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

*To view the __sashimi__ open-source organismal model sharing web-interface, please visit: [https://sashimi.shawntylerschwartz.com/](https://sashimi.shawntylerschwartz.com/).*

To get started, clone the repository:
```bash
$ git clone https://github.com/ShawnTylerSchwartz/sashimi.git
```

Specific installation instructions and requirements, as well as instructions to execute our module, are outlined in our GitHub repository’s [requirements.txt](requirements.txt) file. At this release, users should ensure that they have **Python (Version > 3.4)** installed on their system.

## Requirements
Users should have the following requirements installed:
```
// Available in 'requirements.txt'
numpy==1.16.4
scipy==1.2.1
Pillow==6.1.0
Cython==0.29.12
matplotlib==3.1.0
scikit-image==0.15.0
tensorflow==1.15.0
tensorflow-gpu==1.15.0
keras==2.2.5
opencv-python==4.1.1.26
h5py==2.9.0
ipython==7.6.1
imgaug==0.4.0
```

We highly recommend using a virtual environment to run this program. Here is an example:
```bash
# Install virtualenv if not already installed
$ pip install virtualenv
```
```bash
# Create an environment
# (you can create this file in an "envs" directory in your development directory)
$ virtualenv sashimi_env
```
```bash
# Activate the environment
$ source pathtoenvs/sashimi_env/bin/activate
```
```bash
# Install necessary requirements
$ pip install -r requirements.txt
```
```bash
# Deactivate environment when no longer using the sashimi program
$ deactivate
```

## Running the sashimi pipeline
Users can specify the input directory of organismal images via the *--input* flag (input requirement: a local directory in which JPEG, JPG, or PNG images of fish containing their background pixels are stored). The desired output directory can be specified via the *--output* flag when executing the *Python* script from the command line interpreter (if the specified output directory does not exist, the program will first create the directory as specified by the user before performing the automated background removal procedure).

It is recommended to place the input directory of images inside the cloned sashimi repository on your machine.

Here is a default call of the pipeline:
```shell
$ python sashimi.py --input=fish_images --output=fish_images_output
```

There are a variety of arguments that can be specified for more custom use cases beyond coral reef fishes. Below is a table of the various arguments and options:
| Argument      | Expected Input | Description | Required/Optional |
| ----------- | ----------- | ----------- | ----------- |
| *--input, -i*      | *path/to/input*       | Directory of images to remove backgrounds from.      | *Required*       |
| *--output, -o*   | *path/for/output*        | Directory to save background-removed images in.      | *Required*       |
| *--red, -r*   | Floating point value (from 0 to 1).        | (R)ed intensity values (from 0 to 1) for background-mask color. Defaults to alpha layer.      | *Optional*       |
| *--green, -g*   | Floating point value (from 0 to 1).        | (G)reen intensity values (from 0 to 1) for background-mask color. Defaults to alpha layer.      | *Optional*       |
| *--blue, -b*   | Floating point value (from 0 to 1).        | (B)lue intensity values (from 0 to 1) for background-mask color. (Note: program will scale these values to an integer value between 0 and 255.) Defaults to alpha layer.      | *Optional*       |
| *--silhouette, -s* | Boolean value (0 or 1). | Set to 1 (true) to force segmented outputs to be filled in with black pixels (silhouette). | *Optional* |
| *--segmentation, -z*   | Boolean value (0 or 1).        | Set to 0 (false) to skip image segmentation step (if images are already segmented – either manually or from a previous sashimi run) to perform filling background pixels of previously segmented images (note: must be of *“.png”* file format) with desired background-mask color (RGB) values. Defaults to perform segmentation.      | *Optional*       |
| *--model, -m*   | */path/to/custom/model.h5*        | Path to custom segmentation model to use (defaults to models/fish_segmentation_model_schwartz_v1-0-0.h5)      | *Optional*       |
| *--organism, -n* | *name of class label to detect* | String with name of class label to detect, should match that used in training data. (Defaults to 'fish' for default fish segmentation model.) You can specify a label present in the COCO dataset along with setting --coco/-c to True to utilize the pre-trained weights in COCO. | *Optional* |
| *--coco, -c* | *use the COCO pre-trained weights for detection instead of custom model* | Set to True to use the default pre-trained COCO weights for object detection instead of a custom model (defaults to False). | *Optional* |
| *--regions, -y* | *Name of manual segmentation regions file.* | Should be placed in both train/ and val/ folders, and should have the same exact filename within each of those two folders. (Defaults to _fish-segmentation-regions.json) | *Optional* |
| *--imgsrc, -j* | *Where are source images located ('local' or 'url')* | Specify 'local' or 'url' depending on how paths are provided in your regions .JSON file. Important for distribution purposes. Default is URL -- images will be downloaded from source initially, and won't download again unless directory no longer exists. Use 'local' if images are stored directly in the respective train/ and val/ directories. | *Optional* |

## Examples
Below are example calls to the command line for various examples:

*Color background pixels of previously segmented .png file green*
```bash
$ python sashimi.py --input=fish_images --output=fish_images_output --segmentation=0 --red=0.0 --green=1.0 --blue=0.0
```
**Example of output image from previously specified directory:**
![Example Green Fish](_examples/readme-imgs/Chaetodon_falcula_green.png)

(_Chaetodon falcula_)

---

*Color background pixels of previously segmented .png file blue*
```bash
$ python sashimi.py --input=bird_images --output=bird_images_output --segmentation=0 --red=0.0 --green=0.0 --blue=1.0
```
**Example of output image from previously specified directory:**
![Example Blue Bird](_examples/readme-imgs/Tangara_abbas_blue.png)

(_Tangara abbas_)

*Segment images of taxonomic group with custom model*
```bash
$ python sashimi.py --input=fish_images --output=fish_images_output --model=path/to/custom/model.h5
```
**Example of output image from previously specified directory:**
![Example Segmented Fish](_examples/readme-imgs/Epibulus_insidiator.png)

(_Epibulus insidiator_)

## Training a new model
1. Have training data set ready (`train/` and `val/` directories within the `sashimi` Python module directory, each containing the train/val images and `.json` files with training set coordinates generated via instructions on the _sashimi web-interface_, respectively)
2. Run the `core.py` script within the `sashimi` Python module directory using the `train/`, `val/`.
```bash
$ python core.py train --dataset=/path/to/dir/containing/both/train+val_dirs/ --model=/path/to/mask_rcnn_coco.h5
```