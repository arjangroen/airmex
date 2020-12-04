"""Contains function for preprocessing images"""
# pylint: disable=too-many-locals
import math
import cv2
import numpy as np


def preprocess(
        image: np.ndarray,
        min_hw_ratio: int = 1,
        output_width: int = 299,
        output_height: int = 299,
) -> np.ndarray:
    """"
    Function to preprocess images before making predictions
    :parameter
      image (np.ndarray): image to be preprocessed
      min_hw_ratio (int): height-width ratio of the output image
      output_width (int): width of the output image in pixels
      output_height (int): height of the output image in pixels
    :return
      preprocessed_image (np.ndarray): preprocessed image
    """
    # Trim equal rows from top and bottom to get a square image
    rows, cols = image.shape
    r_to_keep = cols * min_hw_ratio
    r_to_delete = rows - r_to_keep
    remove_from_top = int(math.ceil(r_to_delete / 2))
    remove_from_bottom = int(math.floor(r_to_delete / 2))
    image_top_bottom_trimmed = image[remove_from_top: (rows - remove_from_bottom), :]

    # resample to get the desired image size
    image_resampled = cv2.resize(
        image_top_bottom_trimmed,
        dsize=(output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )

    # Normalize pixel values to take the range [0,1]
    image_clean = image_resampled - np.mean(image_resampled)
    image_clean = image_clean / np.std(image_clean)
    image_clean = (image_clean - np.min(image_clean)) / (
            np.max(image_clean) - np.min(image_clean)
    )

    # Stack into three channels
    image_clean_stacked = np.dstack((image_clean, image_clean, image_clean))
    image_clean_stacked = np.moveaxis(image_clean_stacked, -1, 0)

    # Implement ImageNet Standardization
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    prepocessed_image = (image_clean_stacked - imagenet_mean) / imagenet_std

    return prepocessed_image
