import os
from utils.preprocess_image import preprocess
import cv2
import torch


def load_images(n_images_per_score: int = 1):
    """
    Loads n images per KL score folder from test and makes them ready for pytorch prediction
    Args:
        n_images_per_score: number of images per KL score, so 5 returns 5x5=25 images

    Returns:
        Tensor of shape (n_samples, 3, 299, 299)

    """
    img_folders = ['data/mendeley/kneeKL299/test/' + str(x) for x in range(5)]
    images = []
    for img_folder in img_folders:
        first_n_files = os.listdir(img_folder)[:n_images_per_score]
        for file in first_n_files:
            img = cv2.imread(os.path.join(img_folder, file), 0).astype("float")
            processed_image = preprocess(img)
            input_image = processed_image.reshape((1,) + processed_image.shape)
            input_image = torch.from_numpy(input_image)
            input_image = input_image.float()

            images.append(input_image)

    images = torch.cat(images, dim=0)
    return images
