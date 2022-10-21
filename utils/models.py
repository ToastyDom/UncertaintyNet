from re import X
import torch
import torchvision

from torchvision.models import resnet50, ResNet50_Weights

def get_ResNet50(pretrained, trained_weights=None):

    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        weight_file = trained_weights
        model = 0
        print("TO IMPLEMENT")
    return model