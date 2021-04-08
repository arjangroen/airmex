"""Contains functions regarding KneeNet CNN"""
# pylint: disable=ungrouped-imports, no-member
import os
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
import numpy as np


def rebuild_kneenet():
    """
    Function to build the KneeNet model as described by Thomas et al. in
    Thomas KA, Kidziński Ł, Halilaj E, et al. Automated classification of
    radiographic knee osteoarthritis severity using deep neural networks.
    Radiology: Artificial Intelligence. 2020;2(2):e190065.

    For the original repo see: https://github.com/stanfordnmbl/kneenet-docker
    """
    # Initialize densenet architecture
    pretrained_model = torchvision.models.densenet169(pretrained=False)

    for param in pretrained_model.parameters():
        param.requires_grad = False
    pretrained_model.aux_logits = False

    # Modify the output layer of the densenet to fit our number of output classes
    num_features_knee = 14976
    pretrained_model.classifier = nn.Linear(num_features_knee, 5)

    for param in pretrained_model.classifier.parameters():
        param.requires_grad = False

    # Get model's location
    root_dir = Path(__file__).parent.parent
    model_dir = os.path.join(root_dir, "data/models/KneeNet")
    # Load the model's trained weights
    pretrained_model.load_state_dict(
        torch.load(
            os.path.abspath(model_dir), map_location=lambda storage, loc: storage
        )
    )
    pretrained_model.train(False)
    pretrained_model = pretrained_model.float()

    return pretrained_model


def predict(image: np.ndarray):
    """
    Function to predict KL-class of input image and return logits and probablities
    :parameter
      image (np.ndarray): image to predict
    :return
      logits (np.ndarray): logits of all 5 KL-classes
      probabilities (np.ndarray): probabilities of all 5 KL-classes
    """
    #
    input_image = image.reshape((1,) + image.shape)
    input_image = torch.from_numpy(input_image)
    input_image = input_image.float()
    # Rebuild teh model
    model = rebuild_kneenet()
    # Predict classes and retrieve logits
    logits = model(input_image)[0]
    # Get probabilities with softmax function
    smax = nn.Softmax(dim=0)
    probabilities = smax(logits)
    print(f"Predicted KL-class: {np.argmax(probabilities)}")
    return logits.numpy(), probabilities.numpy()