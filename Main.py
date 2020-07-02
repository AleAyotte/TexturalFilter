"""
    @file:              Main.py
    @Author:            Alexandre Ayotte

    @Creation Date:     01/07/2020
    @Last modification: 01/07/2020

    @Description:       Main program used to test the Filter class and it subclasses.
"""
import argparse
from Filter import LaplacianOfGaussian, Laws, Gabor
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os


def get_input(image_name, repertory="Data"):
    """
    Import the nii image and cast it into numpy nd-array.

    :param image_name: The image name without the extension.
    :param repertory: The image repertory (Options: Data. Result_Martin)
    :return: a numpy nd-array of shape (B, W, H, D) with value between 0 and 1.
    """

    example_filename = os.path.join(
        "C:/Users/moial/OneDrive/Bureau/Maitrise/TexturalFilter", repertory, image_name + '.nii'
    )
    return np.expand_dims(np.array(nib.load(example_filename).dataobj), axis=0) / 255


def get_images_and_filter(test_id):
    """
    Get the input and result images and the corresponding filter object according a given test_od.

    :param test_id: The test identificator as string. "Exemple: 4a1"
    :return: Two nd-arrays that represent the input and result images respectively and a filter object.
    """

    VOLEX_LENGTH = 2

    if test_id == "2a":
        phantom_name = "response"
        sigma = 3 / VOLEX_LENGTH
        length = int(2 * 4 * sigma + 1)

        _filter = LaplacianOfGaussian(3, length, sigma=sigma, padding="constant")

    elif test_id == "2b":
        phantom_name = "checkerboard"
        sigma = 5 / VOLEX_LENGTH
        length = int(2 * 4 * sigma + 1)
        _filter = LaplacianOfGaussian(3, length, sigma=sigma, padding="symmetric")

    elif test_id == "3a1":
        phantom_name = "response"
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=False)

    elif test_id == "3a2":
        # 3D rotation invariance, max pooling
        phantom_name = "response"
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=True)

    elif test_id == "3a3":
        # Energy map
        phantom_name = "response"
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=True)

    elif test_id == "3b1":
        phantom_name = "checkerboard"
        _filter = Laws(["E3", "W5", "R5"], padding="symmetric", rot_invariance=False)

    elif test_id == "3b2":
        # 3D rotation invariance, max pooling
        phantom_name = "checkerboard"
        _filter = Laws(["E3", "W5", "R5"], padding="symmetric", rot_invariance=True)

    elif test_id == "3b3":
        # Energy map
        phantom_name = "checkerboard"
        _filter = Laws(["E3", "W5", "R5"], padding="symmetric", rot_invariance=True)

    elif test_id == "4a1":
        phantom_name = "response"
        sigma = 10 / VOLEX_LENGTH
        lamb = 4 / VOLEX_LENGTH
        size = 2*7*sigma+1
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=0.5, theta=-math.pi/3,
                        rot_invariance=False,
                        padding="constant"
                        )

    elif test_id == "4a2":
        # Rotation invariance and orthogonal planes
        phantom_name = "response"
        sigma = 10 / VOLEX_LENGTH
        lamb = 4 / VOLEX_LENGTH
        size = 2*7*sigma+1
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=0.5, theta=-math.pi/4,
                        rot_invariance=True,
                        padding="constant"
                        )

    elif test_id == "4b1":
        phantom_name = "sphere"
        sigma = 20 / VOLEX_LENGTH
        lamb = 8 / VOLEX_LENGTH
        size = 2*7*sigma+1
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=2.5, theta=-5*math.pi/4,
                        rot_invariance=False,
                        padding="symmetric"
                        )

    elif test_id == "4b2":
        # Rotation invariance and orthogonal planes
        phantom_name = "sphere"
        sigma = 20 / VOLEX_LENGTH
        lamb = 8 / VOLEX_LENGTH
        size = 2*7*sigma+1
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=2.5, theta=-math.pi/8,
                        rot_invariance=True,
                        padding="symmetric"
                        )
    else:
        raise NotImplementedError

    _in = get_input(phantom_name, "Data")
    _out = get_input("Phase1_"+test_id, "Result_Martin")

    return _in, _out, _filter


def main(args):
    _in, _out, _filter = get_images_and_filter(test_id=args.test_id)
    print(_in)
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
