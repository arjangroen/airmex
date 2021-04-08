from typing import Callable
from utils.model import rebuild_kneenet
from utils.load_images import load_images
import torch.nn as nn
import torch
import numpy as np

dl_model = rebuild_kneenet()

def explain(image: torch.Tensor,
            xai_model: Callable,
            baseline,
            multiply_by_inputs=True):
    prediction_logits = dl_model(image)[0]
    softmax = nn.Softmax(dim=0)
    prediction_probas = softmax(prediction_logits)
    explain_label = int(np.argmax(prediction_probas))
    attr_model = xai_model(dl_model, multiply_by_inputs=multiply_by_inputs)
    attr = attr_model.attribute(image, target=explain_label).detach().numpy()
    return np.rollaxis(attr, 1, 4)


def project_redgreen(attr, img, alpha=1):
    """
    Integrate DeepLift attribution into the image
    Args:
        attr: attribution from self.explain()
        img: corresponding image
        alpha: scaling parameter of how strong to project the attribution

    Returns:
        img with DeepLift projection
    """
    img_normalized = normalize(img)
    positive_mask = attr > 0
    negative_mask = attr < 0

    abs_attr = np.abs(attr)
    abs_attr_norm = normalize(abs_attr)
    positive_attr = abs_attr_norm * positive_mask
    negative_attr = abs_attr_norm * negative_mask

    img_normalized[:, :, 0] = img_normalized[:, :, 0] + alpha * positive_attr[:, :, 0]  # Put positive attribution in the red channel
    img_normalized[:, :, 1] = img_normalized[:, :, 1] + alpha * negative_attr[:, :, 0]  # Put negative attribution in the green channel

    img_normalized = normalize(img_normalized)
    return img_normalized


def normalize(ndarray):
    ndarray = ndarray - ndarray.min()
    ndarray = ndarray / ndarray.max()
    return ndarray
