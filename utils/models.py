from re import X
import torch
import torchvision
import torch.nn as nn
from loguru import logger
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

def get_ResNet50(pretrained, num_classes, freeze="True"):

    if pretrained:
        # Load model with weights
        logger.info("Fetching Model Weights")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    else:
        # Load model without weights
        logger.info("Not pretrained")
        model = resnet50()


    # New classifier
    num_in_feats = model.fc.in_features
    model.fc = nn.Linear(num_in_feats, num_classes)

    if freeze=="True":
        logger.info("Freezing backbone")
        """Freezing Layers"""
        for param in model.parameters():
            param.requires_grad = False
        """Unfreese fc layer"""
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def get_ResNet18(pretrained, num_classes, freeze = True, trained_weights=None):
    pass

    # if pretrained:

    #     # Load model with weights
    #     model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    #     # New classifier
    #     num_in_feats = model.fc.in_features
    #     model.fc = nn.Linear(num_in_feats, num_classes)

    #     if freeze==True:

    #         logger.info("Freezing backbone")

    #         """Freezing Layers"""
    #         for param in model.parameters():
    #             param.requires_grad = False
    #         """Unfreese fc layer"""
    #         for param in model.fc.parameters():
    #             param.requires_grad = True


    # else:
    #     weight_file = trained_weights
    #     model = 0
    #     print("TO IMPLEMENT")


    # return model