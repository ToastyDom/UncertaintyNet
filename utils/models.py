from re import X
import torch
import torchvision

from torchvision.models import resnet50, ResNet50_Weights


# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

def get_ResNet50(pretrained, trained_weights=None):

    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        weight_file = trained_weights
        model = 0
        print("TO IMPLEMENT")




    """ 
    To implement:
    - Freeze backbone?
    - New classifier
    
    
    """

    return model