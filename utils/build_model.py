import torch
import torch.nn as nn
import torchvision
import os
from pathlib import Path

def rebuild_kneenet():
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
    model_dir = os.path.join(root_dir, 'data/models/KneeNet')
    # Load the model's trained weights
    pretrained_model.load_state_dict(torch.load(os.path.abspath(model_dir),
                                                map_location=lambda storage, loc: storage))
    pretrained_model.train(False)
    pretrained_model = pretrained_model.float()

    return pretrained_model