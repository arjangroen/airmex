"""Contains function for preprocessing images"""
# pylint: disable=too-many-locals
import math
import cv2
import numpy as np


def preprocess(
        image: np.ndarray,
) -> np.ndarray:
    image_clean = image - np.mean(image)
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
