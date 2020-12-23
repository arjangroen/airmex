from captum.attr import DeepLift
from typing import Callable
from utils.model import rebuild_kneenet
from utils.load_images import load_images
import numpy as np
import matplotlib.pyplot as plt


class Airmex(object):

    def __init__(self,
                 attr_model,
                 model_build_fn: Callable = rebuild_kneenet,
                 data_load_fn: Callable = load_images
                 ):
        """

        Args:
            model_build_fn:
            data_load_fn:
        """
        self.attr_model = attr_model
        self.model = model_build_fn()
        self.images = data_load_fn()

    def explain(self, target=4):
        attr_model = self.attr_model(self.model, multiply_by_inputs=True)
        attr = attr_model.attribute(self.images, target=target).detach().numpy()
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


if __name__ == "__main__":
    a = Airmex(attr_model=DeepLift)
    attr = a.explain()[4]
    img = np.rollaxis(a.images[4].detach().numpy(), 0, 3)
    projection = a.project_deeplift(attr, img)
    plt.imshow(projection)
    plt.show()
