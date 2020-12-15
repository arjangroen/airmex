"""tests all functions related to the original KneeNet CNN"""
import os
from pathlib import Path
import torchvision
import cv2
import numpy as np
from utils.model import rebuild_kneenet, predict
from utils.preprocess_image import preprocess


def test_rebuild_model():
    """Tests the function to rebuild the original KneeNet"""
    pretrained_model = rebuild_kneenet()
    assert isinstance(pretrained_model, torchvision.models.densenet.DenseNet)
    assert pretrained_model.classifier.in_features == 14976
    assert pretrained_model.classifier.out_features == 5


def test_preprocess():
    """Tests the preprocessing of images"""
    img_path = os.path.join("data/mendeley/kneeKL299/train/0/9001695L.png")
    img = cv2.imread(img_path, 0).astype("float")
    processed_image = preprocess(img)
    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape == (3, 299, 299)


def test_predict():
    """Tests the predict function by predicting a sample image"""
    img_path = os.path.join("data/mendeley/kneeKL299/train/4/9039627L.png")
    img = cv2.imread(img_path, 0).astype("float")
    processed_image = preprocess(img)
    logits, probabilities = predict(processed_image)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (5,)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (5,)
