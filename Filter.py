"""
    @file:              Filter.py
    @Author:            Alexandre Ayotte

    @Creation Date:     06/06/2020
    @Last modification: 07/06/2020

    @Reference: 1)      IBSI Volume 2: Standardised convolutional filtering for Radiomics, Adrien Depeursinge,
                        Vincent Andrearczyk, Philip Whybra, Joost van Griethuysen, Henning Müller, Roger Schaer
                        Martin Vallières, Alex Zwanenburg

    @Description:       This program contain of classes of common filters used in medical imaging. Most of them should
                        allow the users to create a kernel of n dimensions, but the convolution will be limited to the
                        1D, 2D and 3D in most of the case.
"""

import numpy as np
import math
import torch
from torch.nn import functional as F
from abc import ABC, abstractmethod
from itertools import product


class Filter(ABC):
    """
    Class frame of each filters classes like laplacian of gaussian, wavelet,
    """

    def __init__(self, ndims, padding="mirror"):
        """
        Constructor of the abstract class Filter

        :param ndims: Number of dimension
        :param padding: The padding type that will be used to produce the convolution
        """
        super().__init__()
        self.dim = ndims
        self.padding = padding
        self.kernel = None
        
        if self.dim == 1:
            self.conv = F.conv1d
        elif self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d
        else:
            raise NotImplementedError

    def _convolve(self, images, orthogonal_rot=False, device="cpu"):
        """
        Convolve a given n-dimensional array with the kernel to generate a filtered image.

        :param images: A n-dimensional numpy array that represent a batch of images to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image
        """

        in_size = np.shape(images)

        # We only handle 2D or 3D images.
        assert len(in_size) == 3 or len(in_size) == 4, \
            "The tensor should have the followed shape (B, H, W) or (B, D, H, W)"

        if not orthogonal_rot:

            # If we have a 2D kernel but a 3D images, we squeeze the tensor
            if self.dim < len(in_size) - 1:
                images = images.reshape((in_size[0] * in_size[1], in_size[2], in_size[3]))

            # We compute the padding size along each dimension
            padding = [int((self.kernel.shape[-1] - 1) / 2), int((self.kernel.shape[-2] - 1) / 2)]
            pad_list = [1, 2]

            if self.dim == 3:
                padding.extend([int((self.kernel.shape[-3] - 1) / 2)])
                pad_list.extend([3])

            # We pad the images and we add the channel axis.
            padded_imgs = self.pad_imgs(images, padding, pad_list)
            new_imgs = np.expand_dims(padded_imgs, axis=1)

            with torch.no_grad():
                # Convert into torch tensor
                _in = torch.from_numpy(new_imgs).float().to(device)
                _filter = torch.from_numpy(self.kernel).float().to(device)

                # Operate the convolution
                result = self.conv(_in, _filter).to(device)

                # Reshape the data to retrieve the following format: (B, C, D, H, W)
                if self.dim < len(in_size) - 1:
                    result = result.reshape((
                        in_size[0], in_size[1], result.size()[1], in_size[2], in_size[3])
                    ).permute(0, 2, 1, 3, 4)

        # If we want orthogonal rotation
        else:
            coronal_imgs = images
            axial_imgs, sagittal_imgs = np.rot90(images, 1, (1, 2)), np.rot90(images, 1, (1, 3))

            # We stack the images along the batch axis
            new_imgs = np.concatenate(
                (coronal_imgs, axial_imgs, sagittal_imgs),
                axis=0
            )

            result = self._convolve(new_imgs, orthogonal_rot=False, device=device)

            # split and unflip and stack the result on a new axis
            batch = in_size[0]
            result_coronal = result[0:batch, :, :, :, :]
            result_axial = result[batch:2*batch, :, :, :, :].rot90(1, (3, 2))
            result_sagittal = result[2*batch:3*batch, :, :, :, :].rot90(1, (4, 2))
            result = torch.stack([result_coronal, result_axial, result_sagittal])

        return result

    def pad_imgs(self, images, padding, axes):
        """
        Apply padding on a 3d images using a 2D padding pattern.

        :param images: a numpy array that represent the image.
        :param padding: The padding length that will apply on each side of each axe.
        :param axes: A list of axes on which the padding will be done.
        :return: A numpy array that represent the padded image.
        """
        pad_tuple = ()
        j = 1

        for i in range(np.ndim(images)):
            if i in axes:
                pad_tuple += ((padding[-j], padding[-j]),)
                j += 1
            else:
                pad_tuple += ((0, 0),)

        return np.pad(images, pad_tuple, mode=self.padding)

    @abstractmethod
    def create_kernel(self):
        pass


class LaplacianOfGaussian(Filter):
    """
    The Laplacian of gaussian filter class.
    """

    def __init__(self, ndims, size, sigma=0.1, padding="symmetric"):
        """
        The constructor of the laplacian of gaussian (LoG) filter

        :param ndims: Number of dimension of the kernel filter
        :param size: An integer that represent the length along one dimension of the kernel.
        :param sigma: The gaussian standard deviation parameter of the laplacian of gaussian filter
        :param padding: The padding type that will be used to produce the convolution
        """

        assert isinstance(ndims, int) and ndims > 0, "ndims should be a positive integer"
        assert ((size+1)/2).is_integer() and size > 0, "size should be a positive odd number."
        assert sigma > 0, "alpha should be a positive float."

        super().__init__(ndims, padding)

        self.size = int(size)
        self.sigma = sigma
        self.create_kernel()

    def create_kernel(self):
        """
        This method construct the LoG kernel using the parameters specified to the constructor

        :return: The laplacian of gaussian kernel as a numpy multidimensionnal array
        """

        def compute_weight(position):
            distance_2 = np.sum(position**2)
            # $\frac{-1}{\sigma^2} * \frac{1}{\sqrt{2 \pi} \sigma}^D = \frac{-1}{\sqrt{D/2}{2 \pi} * \sigma^{D+2}}$
            first_part = -1/((2*math.pi)**(self.dim/2) * self.sigma**(self.dim+2))

            # $(D - \frac{||k||^2}{\sigma^2}) * e^{\frac{-||k||^2}{2 \sigma^2}}$
            second_part = (self.dim - distance_2/self.sigma**2)*math.e**(-distance_2/(2 * self.sigma**2))

            return first_part * second_part

        # Initialize the kernel as tensor of zeros
        kernel = np.zeros([self.size for _ in range(self.dim)])

        for k in product(range(self.size), repeat=self.dim):
            kernel[k] = compute_weight(np.array(k)-int((self.size-1)/2))

        self.kernel = np.expand_dims(kernel, axis=(0, 1))

    def convolve(self, image, orthogonal_rot=False, device="cpu"):
        """
        Filter a given image using the LoG kernel defined during the construction of this instance.

        :param image: A n-dimensional numpy array that represent the image to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image
        """
        return self._convolve(image, orthogonal_rot, device).cpu().numpy()


class Gabor(Filter):
    """
    The Gabor filter class
    """

    def __init__(self, ndims, size, sigma, lamb, gamma, theta, padding="symmetric"):
        """
        The constructor of the Gabor filter. Highly inspired by Ref 1)

        :param ndims: Number of dimension of the kernel filter
        :param size: An integer that represent the length along one dimension of the kernel.
        :param sigma: A positive float that represent the scale of the Gabor filter
        :param lamb: A positive float that represent the wavelength in the Gabor filter. (mm or pixel?)
        :param gamma: A positive float that represent the spacial aspect ratio
        :param theta: Angle parameter used in the rotation matrix
        :param padding: The padding type that will be used to produce the convolution
        """

        assert isinstance(ndims, int) and ndims > 0, "ndims should be a positive integer"
        assert ((size + 1) / 2).is_integer() and size > 0, "size should be a positive odd number."
        assert sigma > 0, "sigma should be a positive float"
        assert lamb > 0, "lamb represent the wavelength, so it should be a positive float"
        assert gamma > 0, "gamma is the ellipticity of the support of the filter, so it should be a positive float"
        super().__init__(ndims, padding)

        self.size = size
        self.sigma = sigma
        self.lamb = lamb
        self.gamma = gamma
        self.theta = theta

        self.create_kernel()

    def create_kernel(self):
        """
        Create the kernel of the Gabor filter

        :return: A list of numpy 2D-array that contain the kernel of the real part and the imaginary part respectively.
        """

        def compute_weight(position):
            k2 = position[0]*math.cos(self.theta) + position[1] * math.sin(self.theta)
            k1 = position[1]*math.cos(self.theta) - position[0] * math.sin(self.theta)

            common = math.e**(-(k1**2 + (self.gamma*k2)**2)/(2*self.sigma**2))
            real = math.cos(2*math.pi*k1/self.lamb)
            im = math.sin(2*math.pi*k1/self.lamb)
            return common*real, common*im

        # Initialize the kernel as tensor of zeros
        real_kernel = np.zeros([self.size for _ in range(2)])
        im_kernel = np.zeros([self.size for _ in range(2)])

        for k in product(range(self.size), repeat=2):
            real_kernel[k], im_kernel[k] = compute_weight(np.array(k)-int((self.size-1)/2))

        self.kernel = np.expand_dims([real_kernel, im_kernel], axis=1)

    def convolve(self, image, orthogonal_rot=False, device="cpu"):
        """
        Filter a given image using the Gabor kernel defined during the construction of this instance.

        :param image: A n-dimensional numpy array that represent the image to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image as a numpy ndarray
        """

        result = self._convolve(image, orthogonal_rot, device)

        with torch.no_grad():

            if orthogonal_rot:
                # Aggregate the data
                result = torch.norm(result.to(device), dim=2)
                result = torch.mean(result, dim=0)
            else:
                result = torch.norm(result.to(device), dim=1)
            print("shape final", result.size())
        return result.cpu().numpy()


class Laws(Filter):
    """
    The Laws filter class
    """

    def __init__(self, config=None, padding="symmetric"):
        """
        The constructor of the Laws filter

        :param config: A string list of every 1D filter used to create the Laws kernel. Since the outer product is
                       not commutative, we need to use a list to specify the order of the outer product. It is not
                       recommended to use filter of different size to create the Laws kernel.
        :param padding: The padding type that will be used to produce the convolution
        """

        ndims = len(config)

        super().__init__(ndims, padding)

        self.config = config
        self.create_kernel()

    @staticmethod
    def get_filter(name):
        """
        This method create a 1D filter according to the given filter name.

        :param name: The filter name. (Such as L3, L5, E3, E5, S3, S5, W5 or R5)
        :return: A 1D filter that is needed to construct the Laws kernel.
        """

        if name == "L3":
            return 1/math.sqrt(6) * np.array([1, 2, 1])
        elif name == "L5":
            return 1/math.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif name == "E3":
            return 1 / math.sqrt(2) * np.array([-1, 0, 1])
        elif name == "E5":
            return 1 / math.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif name == "S3":
            return 1 / math.sqrt(6) * np.array([-1, 2, -1])
        elif name == "S5":
            return 1 / math.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif name == "W5":
            return 1 / math.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif name == "R5":
            return 1 / math.sqrt(70) * np.array([1, -4, 6, -4, 1])
        else:
            raise Exception("{} is not a valid filter name. "
                            "Choose between, L3, L5, E3, E5, S3, S5, W5, R5".format(name))

    def create_kernel(self):
        """
        Create the Laws by computing the outer product of 1d filter specified in the config attribute.
        Kernel = config[0] X config[1] X ... X config[n]. Where X is the outer product.

        :return: A numpy multi-dimensional arrays that represent the Laws kernel.
        """
        kernel = self.get_filter(self.config[0])
        shape = kernel.shape

        for i in range(1, len(self.config)):
            sub_kernel = self.get_filter(self.config[i])
            shape += np.shape(sub_kernel)

            kernel = np.outer(kernel, sub_kernel).reshape(shape)

        self.kernel = np.expand_dims(kernel, axis=(0, 1))

    def convolve(self, image, orthogonal_rot=False, device="cpu"):
        """
        Filter a given image using the Laws kernel defined during the construction of this instance.

        :param image: A n-dimensional numpy array that represent the image to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image
        """

        return self._convolve(image, orthogonal_rot, device).cpu().numpy()
