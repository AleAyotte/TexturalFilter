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
from itertools import product, permutations


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
            padding = [int((self.kernel.shape[-1] - 1) / 2) for _ in range(self.dim)]
            pad_axis_list = [i for i in range(1, self.dim+1)]

            # We pad the images and we add the channel axis.
            padded_imgs = self._pad_imgs(images, padding, pad_axis_list)
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

    def _pad_imgs(self, images, padding, axes):
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

        :param image: A n-dimensional numpy array that represent the images to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image
        """
        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(image, 1, 3)
        result = torch.squeeze(self._convolve(image, orthogonal_rot, device), dim=1)
        return np.swapaxes(result.cpu().numpy(), 1, 3)


class Gabor(Filter):
    """
    The Gabor filter class
    """

    def __init__(self, size, sigma, lamb, gamma, theta, rot_invariance=False, padding="symmetric"):
        """
        The constructor of the Gabor filter. Highly inspired by Ref 1)

        :param size: An integer that represent the length along one dimension of the kernel.
        :param sigma: A positive float that represent the scale of the Gabor filter
        :param lamb: A positive float that represent the wavelength in the Gabor filter. (mm or pixel?)
        :param gamma: A positive float that represent the spacial aspect ratio
        :param theta: Angle parameter used in the rotation matrix
        :param rot_invariance: If true, rotation invariance will be done on the kernel and the kernel
                               will be rotate 2*pi / theta times.
        :param padding: The padding type that will be used to produce the convolution
        """

        assert ((size + 1) / 2).is_integer() and size > 0, "size should be a positive odd number."
        assert sigma > 0, "sigma should be a positive float"
        assert lamb > 0, "lamb represent the wavelength, so it should be a positive float"
        assert gamma > 0, "gamma is the ellipticity of the support of the filter, so it should be a positive float"
        super().__init__(ndims=2, padding=padding)

        self.size = size
        self.sigma = sigma
        self.lamb = lamb
        self.gamma = gamma
        self.theta = theta
        self.rot = rot_invariance
        self.create_kernel()

    def create_kernel(self):
        """
        Create the kernel of the Gabor filter

        :return: A list of numpy 2D-array that contain the kernel of the real part and the imaginary part respectively.
        """

        def compute_weight(position, theta):
            k2 = position[0]*math.cos(theta) + position[1] * math.sin(theta)
            k1 = position[1]*math.cos(theta) - position[0] * math.sin(theta)

            common = math.e**(-(k1**2 + (self.gamma*k2)**2)/(2*self.sigma**2))
            real = math.cos(2*math.pi*k1/self.lamb)
            im = math.sin(2*math.pi*k1/self.lamb)
            return common*real, common*im

        # Rotation invariance
        nb_rot = round(2*math.pi/abs(self.theta)) if self.rot else 1
        real_list = []
        im_list = []

        for i in range(1, nb_rot+1):
            # Initialize the kernel as tensor of zeros
            real_kernel = np.zeros([self.size for _ in range(2)])
            im_kernel = np.zeros([self.size for _ in range(2)])

            for k in product(range(self.size), repeat=2):
                real_kernel[k], im_kernel[k] = compute_weight(np.array(k)-int((self.size-1)/2), self.theta*i)

            real_list.extend([real_kernel])
            im_list.extend([im_kernel])

        self.kernel = np.expand_dims(
            np.concatenate((real_list, im_list), axis=0),
            axis=1
        )

    def convolve(self, image, orthogonal_rot=False, device="cpu"):
        """
        Filter a given image using the Gabor kernel defined during the construction of this instance.

        :param image: A n-dimensional numpy array that represent the images to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagittal axis
        :param device: On which device do we need to compute the convolution
        :return: The filtered image as a numpy ndarray
        """

        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(image, 1, 3)

        result = self._convolve(image, orthogonal_rot, device)

        with torch.no_grad():
            # Reshape two get real and imaginary response on the first axis.
            _dim = 2 if orthogonal_rot else 1
            nb_rot = int(result.size()[_dim]/2)
            result = torch.stack(torch.split(result, nb_rot, _dim), dim=0)

            # 2D modulus response map
            result = torch.norm(result.to(device), dim=0)

            # Rotation invariance.
            result = torch.mean(result, dim=2) if orthogonal_rot else torch.mean(result, dim=1)

            # Aggregate orthogonal rotation
            result = torch.mean(result, dim=0) if orthogonal_rot else result
        return np.swapaxes(result.cpu().numpy(), 1, 3)


class Laws(Filter):
    """
    The Laws filter class
    """

    def __init__(self, config=None, padding="symmetric", energy_distance=7, rot_invariance=False):
        """
        The constructor of the Laws filter

        :param config: A string list of every 1D filter used to create the Laws kernel. Since the outer product is
                       not commutative, we need to use a list to specify the order of the outer product. It is not
                       recommended to use filter of different size to create the Laws kernel.
        :param padding: The padding type that will be used to produce the convolution
        :param energy_distance: The distance that will be used to create the energy_kernel.
        :param rot_invariance: If true, rotation invariance will be done on the kernel.
        """

        ndims = len(config)

        super().__init__(ndims, padding)

        self.config = config
        self.energy_dist = energy_distance
        self.rot = rot_invariance
        self.energy_kernel = None
        self.create_kernel()
        self.__create_energy_kernel()

    @staticmethod
    def __get_filter(name, pad=False):
        """
        This method create a 1D filter according to the given filter name.

        :param name: The filter name. (Such as L3, L5, E3, E5, S3, S5, W5 or R5)
        :param pad: If true, add zero padding of lenght 1 each side of kernel L3, E3 and S3
        :return: A 1D filter that is needed to construct the Laws kernel.
        """

        if name == "L3":
            ker = np.array([0, 1, 2, 1, 0]) if pad else np.array([1, 2, 1])
            return 1/math.sqrt(6) * ker
        elif name == "L5":
            return 1/math.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif name == "E3":
            ker = np.array([0, -1, 0, 1, 0]) if pad else np.array([-1, 0, 1])
            return 1 / math.sqrt(2) * ker
        elif name == "E5":
            return 1 / math.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif name == "S3":
            ker = np.array([0, -1, 2, -1, 0]) if pad else np.array([-1, 2, -1])
            return 1 / math.sqrt(6) * ker
        elif name == "S5":
            return 1 / math.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif name == "W5":
            return 1 / math.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif name == "R5":
            return 1 / math.sqrt(70) * np.array([1, -4, 6, -4, 1])
        else:
            raise Exception("{} is not a valid filter name. "
                            "Choose between, L3, L5, E3, E5, S3, S5, W5, R5".format(name))

    def __verify_padding_need(self):
        """
        Check if we need to pad the kernels

        :return: A boolean that indicate if a kernel is smaller than at least one other.
        """

        ker_length = np.array([int(name[-1]) for name in self.config])

        return not(ker_length.min == ker_length.max)

    def create_kernel(self):
        """
        Create the Laws by computing the outer product of 1d filter specified in the config attribute.
        Kernel = config[0] X config[1] X ... X config[n]. Where X is the outer product.

        :return: A numpy multi-dimensional arrays that represent the Laws kernel.
        """

        pad = self.__verify_padding_need()
        filter_list = np.array([[self.__get_filter(name, pad) for name in self.config]])

        if self.rot:
            filter_list = np.concatenate((filter_list, np.flip(filter_list, axis=2)), axis=0)
            prod_list = [prod for prod in product(*np.swapaxes(filter_list, 0, 1))]

            perm_list = []
            for i in range(len(prod_list)):
                perm_list.extend([perm for perm in permutations(prod_list[i])])

            perm_list = np.unique(perm_list, axis=0)
            filter_list = perm_list

        kernel_list = []

        for perm in filter_list:
            kernel = perm[0]
            shape = kernel.shape

            for i in range(1, len(perm)):
                sub_kernel = perm[i]
                shape += np.shape(sub_kernel)
                kernel = np.outer(sub_kernel, kernel).reshape(shape)

            kernel_list.extend([np.expand_dims(np.flip(kernel, axis=(-1, 1)), axis=0)])

        self.kernel = np.array(kernel_list)

    def __create_energy_kernel(self):
        """
        Create the kernel that will be used to generate Laws texture energy images

        :return: A numpy multi-dimensional arrays that represent the Laws energy kernel.
        """

        # Initialize the kernel as tensor of zeros
        kernel = np.zeros([self.energy_dist*2+1 for _ in range(self.dim)])

        for k in product(range(self.energy_dist*2 + 1), repeat=self.dim):
            position = np.array(k)-self.energy_dist
            kernel[k] = 1 if np.max(abs(position)) <= self.energy_dist else 0

        self.energy_kernel = np.expand_dims(kernel/np.prod(kernel.shape), axis=(0, 1))

    def __compute_energy_image(self, images, device="cpu"):
        """
        Compute the Laws texture energy images as described in (Ref 1).

        :param images: A n-dimensional numpy array that represent the filtered images
        :param device: On which device do we need to compute the convolution
        :return:
        """

        with torch.no_grad():
            # We convert the kernel and to images into torch tensor
            _in = torch.from_numpy(images).float().to(device)
            _filter = torch.from_numpy(self.energy_kernel).float().to(device)

            # Operate the convolution
            return self.conv(_in.abs_(), _filter).to(device)

    def convolve(self, image, orthogonal_rot=False, energy_image=False, device="cpu"):
        """
        Filter a given image using the Laws kernel defined during the construction of this instance.

        :param image: A n-dimensional numpy array that represent the images to filter
        :param orthogonal_rot: If true, the 3D images will be rotated over coronal, axial and sagital axis
        :param energy_image: If true, return also the Laws Texture Energy Images
        :param device: On which device do we need to compute the convolution
        :return: The filtered image
        """

        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(image, 1, 3)

        if orthogonal_rot:
            raise NotImplementedError

        result = self._convolve(image, orthogonal_rot, device)

        if energy_image:
            # We pad the response map
            padding = [self.energy_dist for _ in range(2 * self.dim)]
            pad_axis_list = [i for i in range(2, self.dim + 2)]

            response = self._pad_imgs(result.cpu().numpy(), padding, pad_axis_list)

            # We compute the energy map and we apply the max pooling on the energy and the response maps
            energy_imgs, _ = torch.max(self.__compute_energy_image(np.swapaxes(response, 0, 1), device), dim=0)
            result, _ = torch.max(result, dim=1)

            return np.swapaxes(result.cpu().numpy(), 1, 3), np.swapaxes(energy_imgs.cpu().numpy(), 1, 3)
        else:
            result, _ = torch.max(result, dim=1)
            return np.swapaxes(result.cpu().numpy(), 1, 3)
