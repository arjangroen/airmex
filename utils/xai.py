from typing import Callable
from utils.model import rebuild_kneenet
from utils.load_images import load_images
import torch.nn as nn

import numpy as np


class Airmex(object):

    def __init__(self):
        """
        Args:
            model_build_fn: function that return a deep learning model
            data_load_fn: function that returns data
        """

    def explain(self, dl_model, xai_model, image, baseline, multiply_by_inputs=True):
        prediction_logits = dl_model(image)[0]
        prediction_probas = nn.Softmax()
        target = np.argmax(prediction_probas)

        attr_model = xai_model(dl_model, multiply_by_inputs=True)
        attr = attr_model.attribute(image, target=target).detach().numpy()
        return np.rollaxis(attr, 1, 4)

    def project_deeplift(self, attr, img, alpha=1):
        """
        Integrate DeepLift attribution into the image
        Args:
            attr: attribution from self.explain()
            img: corresponding image
            alpha: scaling parameter of how strong to project the attribution

        Returns:
            img with DeepLift projection
        """
        img = self.normalize(img)
        positive_mask = attr > 0
        negative_mask = attr < 0
        abs_attr = np.abs(attr)
        abs_attr_norm = self.normalize(abs_attr)
        positive_attr = abs_attr_norm * positive_mask
        negative_attr = abs_attr_norm * negative_mask

        img[:, :, 0] = img[:, :, 0] + alpha * positive_attr[:, :, 0]  # Put positive attribution in the red channel
        img[:, :, 1] = img[:, :, 1] + alpha * negative_attr[:, :, 0]  # Put negative attribution in the green channel
        img = self.normalize(img)
        return img

    @staticmethod
    def normalize(img):
        img = img - img.min()
        img = img / img.max()
        return img
