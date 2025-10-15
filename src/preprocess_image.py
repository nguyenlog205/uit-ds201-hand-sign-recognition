import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class ImagePreprocessor():
    def __init__(self):
        """
        Hehe
        """
        self.inane = 0
    
    def sharpen_image(
        self,
        input_image,
        sharpen_factor = 1.5
    ):
        """
        Sharpens an image using a kernel-based filter.

        Args:
            input_image (numpy.ndarray): The input image (can be color or grayscale).
            sharpen_factor (float): The sharpening factor. Higher values lead to a sharper image
                                    but can also introduce noise and artifacts ('halos').
                                    A typical range is 1.0 to 3.0.

        Returns:
            numpy.ndarray: The sharpened image.
        """
        if input_image is None:
            print("Error: Could not read the input image.")
            return None
        
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        # Apply convolution with given kernel

        sharpened_image = cv2.filter2D(input_image, -1, kernel=kernel)
        blurred = cv2.GaussianBlur(input_image, (0, 0), 3)
        sharpened_image = cv2.addWeighted(input_image, 1.0 + sharpen_factor, blurred, -sharpen_factor, 0)
        sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
        return sharpened_image
    
    def filter_noise(
        self,
        input_image,
        sigma = 1
    ):
        """
        Filter noise of an image with Gaussian-based approach.

        Args:
            input_image (numpy.ndarray): The input image (can be color or grayscale).

        Returns:
            numpy.ndarray: The filtered image.
        """
        if input_image is None or not isinstance(input_image, np.ndarray):
            raise ValueError("Invalid input: input_image must be a numpy.ndarray")

        # Apply Gaussian filter
        filtered_image = gaussian_filter(input_image, sigma=sigma)
        return filtered_image


def pipeline():
    return None