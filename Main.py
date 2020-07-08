"""
    @file:              Main.py
    @Author:            Alexandre Ayotte

    @Creation Date:     01/07/2020
    @Last modification: 02/07/2020

    @Description:       Main program used to test the Filter class and it subclasses.
"""
import argparse
from Filter import LaplacianOfGaussian, Laws, Gabor, Wavelet
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import torch


def get_input(image_name, repertory="Data"):
    """
    Import the nii image and cast it into numpy nd-array.

    :param image_name: The image name without the extension.
    :param repertory: The image repertory (Options: Data. Result_Martin)
    :return: a numpy nd-array of shape (B, W, H, D) with value between 0 and 1.
    """

    example_filename = os.path.join(
        repertory, image_name + '.nii'
    )
    return np.expand_dims(np.array(nib.load(example_filename).dataobj), axis=0)


def execute_test(test_id, device="cpu"):
    """
    Get the input and result images and the corresponding filter object according a given test_id.

    :param test_id: The test identificator as string. "Exemple: 4a1"
    :param device: On which device do we need to compute the convolution
    :return: Three nd-arrays that represent the input, the result and the ground truth images respectively.
    """

    VOLEX_LENGTH = 2

    if test_id == "2a":
        _in = get_input("response", "Data")
        sigma = 3 / VOLEX_LENGTH
        length = int(2 * 4 * sigma + 1)

        _filter = LaplacianOfGaussian(3, length, sigma=sigma, padding="constant")
        result = _filter.convolve(_in, device=device)

    elif test_id == "2b":
        _in = get_input("checkerboard", "Data")
        sigma = 5 / VOLEX_LENGTH
        length = int(2 * 4 * sigma + 1)
        _filter = LaplacianOfGaussian(3, length, sigma=sigma, padding="symmetric")
        result = _filter.convolve(_in, device=device)

    elif test_id == "3a1":
        _in = get_input("response", "Data")
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=False)
        result = _filter.convolve(_in, energy_image=False, device=device)

    elif test_id == "3a2":
        _in = get_input("response", "Data")
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=True)
        result = _filter.convolve(_in, energy_image=False, device=device)

    elif test_id == "3a3":
        _in = get_input("response", "Data")
        _filter = Laws(["E5", "L5", "S5"], padding="constant", rot_invariance=True)
        _, result = _filter.convolve(_in, energy_image=True, device=device)

    elif test_id == "3b1":
        _in = get_input("checkerboard", "Data")
        _filter = Laws(["E5", "W5", "R5"], padding="symmetric", rot_invariance=False)
        result = _filter.convolve(_in, energy_image=False, device=device)

    elif test_id == "3b2":
        _in = get_input("checkerboard", "Data")
        _filter = Laws(["E5", "W5", "R5"], padding="symmetric", rot_invariance=True)
        result = _filter.convolve(_in, energy_image=False, device=device)

    elif test_id == "3b3":
        _in = get_input("checkerboard", "Data")
        _filter = Laws(["E5", "W5", "R5"], padding="symmetric", rot_invariance=True)
        _, result = _filter.convolve(_in, energy_image=True, device=device)

    elif test_id == "4a1":
        _in = get_input("response", "Data")
        sigma = 10 / VOLEX_LENGTH
        lamb = 4 / VOLEX_LENGTH
        size = int(2*7*sigma+1)
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=0.5, theta=-math.pi/3,
                        rot_invariance=False,
                        padding="constant"
                        )
        result = _filter.convolve(_in, False, device=device)

    elif test_id == "4a2":
        _in = get_input("response", "Data")
        sigma = 10 / VOLEX_LENGTH
        lamb = 4 / VOLEX_LENGTH
        size = int(2*7*sigma+1)
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=0.5, theta=-math.pi/4,
                        rot_invariance=True,
                        padding="constant"
                        )
        result = _filter.convolve(_in, True, device=device)

    elif test_id == "4b1":
        _in = get_input("sphere", "Data")
        sigma = 20 / VOLEX_LENGTH
        lamb = 8 / VOLEX_LENGTH
        size = int(2*7*sigma+1)
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=2.5, theta=-5*math.pi/4,
                        rot_invariance=False,
                        padding="symmetric"
                        )
        result = _filter.convolve(_in, False, device=device)

    elif test_id == "4b2":
        _in = get_input("sphere", "Data")
        sigma = 20 / VOLEX_LENGTH
        lamb = 8 / VOLEX_LENGTH
        size = int(2*7*sigma+1)
        _filter = Gabor(size=size, sigma=sigma, lamb=lamb,
                        gamma=2.5, theta=-math.pi/8,
                        rot_invariance=True,
                        padding="symmetric"
                        )
        result = _filter.convolve(_in, True, device=device)

    elif test_id == "5a1":
        _in = get_input("response", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="db2",
                          rot_invariance=False,
                          padding="constant"
                          )
        result = _filter.convolve(_in, _filter="LHL", level=1)

    elif test_id == "5a2":
        _in = get_input("response", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="db2",
                          rot_invariance=True,
                          padding="constant"
                          )
        result = _filter.convolve(_in, _filter="LHL", level=1)

    elif test_id == "6a1":
        _in = get_input("sphere", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="coif1",
                          rot_invariance=False,
                          padding="wrap"
                          )
        result = _filter.convolve(_in, _filter="HHL", level=1)

    elif test_id == "6a2":
        _in = get_input("sphere", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="coif1",
                          rot_invariance=True,
                          padding="wrap"
                          )
        result = _filter.convolve(_in, _filter="HHL", level=1)

    elif test_id == "7a1":
        _in = get_input("checkerboard", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="haar",
                          rot_invariance=True,
                          padding="symmetric"
                          )
        result = _filter.convolve(_in, _filter="LLL", level=2)

    elif test_id == "7a2":
        _in = get_input("checkerboard", "Data")
        _filter = Wavelet(ndims=3, wavelet_name="haar",
                          rot_invariance=True,
                          padding="symmetric"
                          )
        result = _filter.convolve(_in, _filter="HHH", level=2)
    else:
        raise NotImplementedError

    ground_truth = get_input("Phase1_"+test_id, "Result_Martin")

    return _in, result / 255, ground_truth / 255


def plot_comparison(result, ground_truth, _slice):
    """
    Plot the coronal, axial and sagittal slices of the ground truth, the result and their squared error

    :param result: The result obtained by the program.
    :param ground_truth: The result obtained by Martin Vallières
    :param _slice: Which slice will be plot along each axis.
    """

    error = (ground_truth - result) ** 2
    mean_square_error = np.mean(error)

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Mean square error: {}'.format(mean_square_error), fontsize=16)

    fig.add_subplot(3, 3, 1, ylabel="Ground truth", title="Coronal")
    plt.imshow(ground_truth[0, :, :, _slice])

    fig.add_subplot(3, 3, 2, title="Axial")
    plt.imshow(ground_truth[0, :, _slice, :])

    fig.add_subplot(3, 3, 3, title="Sagittal")
    plt.imshow(ground_truth[0, _slice, :, :])

    fig.add_subplot(3, 3, 4, ylabel="Result")
    plt.imshow(result[0, :, :, _slice])

    fig.add_subplot(3, 3, 5)
    plt.imshow(result[0, :, _slice, :])

    fig.add_subplot(3, 3, 6)
    plt.imshow(result[0, _slice, :, :])

    fig.add_subplot(3, 3, 7, ylabel="square error")
    plt.imshow(error[0, :, :, _slice])

    fig.add_subplot(3, 3, 8)
    plt.imshow(error[0, :, _slice, :])

    fig.add_subplot(3, 3, 9)
    plt.imshow(error[0, _slice, :, :])

    plt.show()


def main(args):
    torch.set_num_threads(1)
    _in, result, ground_truth = execute_test(test_id=args.test_id, device=args.device)
    plot_comparison(result, ground_truth, _slice=args.slice)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_id',
        type=str,
        default='',
        help='Test to execute.'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="On which device the result will be compute. (Exemple: 'cpu' or 'cuda:0'"
    )

    parser.add_argument(
        '--slice',
        type=int,
        default=31,
        help='Which slice will be plot..'
    )
    _args = parser.parse_args()

    main(_args)
