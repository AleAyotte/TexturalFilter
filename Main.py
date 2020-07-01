"""
    @file:              Main.py
    @Author:            Alexandre Ayotte

    @Creation Date:     01/07/2020
    @Last modification: 01/07/2020

    @Description:       Main program used to test the Filter class and it subclasses.
"""
import argparse
from Filter import LaplacianOfGaussian, Laws, Gabor
import nibabel as nib
import numpy as np
import os, math
import matplotlib.pyplot as plt


def get_input(image_name, repertory="Data"):
    """
    Import the nii image and cast it into numpy nd-array.

    :param image_name: The image name without the extension.
    :param repertory: The image repertory (Options: Data. Result_Martin)
    :return: a numpy nd-array of shape (B, W, H, D) with value between 0 and 1.
    """
    example_filename = os.path.join(
        "C:/Users/moial/OneDrive/Bureau/Maitrise/TexturalFilter", repertory, image_name, '.nii'
    )
    return np.expand_dims(np.array(nib.load(example_filename).dataobj), axis=0) / 255


def main(args):
    print("ok")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_id',
        type=str,
        default='',
        help='Test à effectuer.'
    )

    _args = parser.parse_args()

    main(_args)


