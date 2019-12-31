# fish-segmentation

We present a software module written in *Python* (Version 3.6.5) that is accessible by cloning the GitHub repository [https://github.com/ShawnTylerSchwartz/fish-segmentation](https://github.com/ShawnTylerSchwartz/fish-segmentation). Training and inspiration based on Matterport's open-source Keras and TensorFlow implementation of Mask R-CNN [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

To get started, clone the repository:
```
git clone https://github.com/ShawnTylerSchwartz/fish-segmentation.git
```

Specific installation instructions and requirements, as well as instructions to execute our module, are outlined in our GitHub repository’s [requirements.txt](requirements.txt) file. At this release, users should ensure that they have **Python (Version > 3.4)** installed on their system.

## Requirements
We highly recommend using a virtual environment to run this program. Here is an example:
```
# Install virtualenv if not already installed
pip install virtualenv
```
```
# Create an environment
# (you can create this file in a "envs" directory in your development directory)
virtualenv fishseg_env
```
```
# Activate the environment
source pathtoenvs/fishseg_env/bin/activate
```
```
# Install necessary requirements
pip install -r requirements.txt
```
```
# Deactivate environment when no longer using the fish-segmentation program
deactivate
```

## Running Program
Users can specify the input directory of fish images via the *--input* flag (input requirement: a local directory in which JPEG, JPG, or PNG images of fish containing their background pixels are stored). The desired output directory can be specified via the *--output* flag when executing the *Python* script from the command-line interpreter (if the specified output directory does not exist, the program will first create the directory as specified by the user before performing the automated background removal procedure).

It is recommended to place the input directory of images inside the cloned fish-segmentation repository on your machine.
```
python3 FishSeg.py --input=fish_images --output=fish_images_output
```