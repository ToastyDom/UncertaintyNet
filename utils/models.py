from re import X
import torch
import torchvision

def get_ResNet50(pretrained, trained_weights=None):

    if pretrained:
        model = torchvision.models.resnet50(pretrained=pretrained)
    else:
        weight_file = trained_weights
        model = 0
        print("TO IMPLEMENT")
    return model