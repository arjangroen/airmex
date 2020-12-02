from utils.build_model import rebuild_kneenet
import pytest
import torchvision

# @pytest.mark.filterwarnings('ignore::UserWarning')
def test_rebuild_model():
    pretrained_model = rebuild_kneenet()
    assert type(pretrained_model) == torchvision.models.densenet.DenseNet
    assert pretrained_model.classifier.in_features == 14976
    assert pretrained_model.classifier.out_features == 5