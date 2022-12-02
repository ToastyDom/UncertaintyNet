from re import X
import torch
import torchvision
import torch.nn as nn
from loguru import logger
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights, efficientnet_b6, EfficientNet_B6_Weights
import timm


# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch

def get_ResNet50(pretrained, num_classes, freeze="True"):

    if pretrained=="True":
        # Load model with weights
        logger.info("Fetching Model Weights for 50")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    else:
        # Load model without weights
        logger.info("Not pretrained for 50")
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



def get_ResNet101(pretrained, num_classes, freeze="True"):

    if pretrained=="True":
        # Load model with weights
        logger.info("Fetching Model Weights for 101")
        model = resnet101(weights=ResNet101_Weights.DEFAULT)

    else:
        # Load model without weights
        logger.info("Not pretrained for 101")
        model = resnet101()


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




def get_efficientnet(pretrained, num_classes, freeze="True"):

    if pretrained=="True":
        # Load model with weights
        logger.info("Fetching Model Weights for EN")
        model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)

    else:
        # Load model without weights
        logger.info("Not pretrained for EN")
        model = efficientnet_b6()


    # New classifier
    #num_in_feats = model.fc.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                    nn.Linear(2304, num_classes, bias=True))

    if freeze=="True":
        logger.info("Freezing backbone")
        """Freezing Layers"""
        for param in model.parameters():
            param.requires_grad = False
        """Unfreese fc layer"""
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model



def get_vit(pretrained, num_classes, freeze="True"):

    if pretrained=="True":
        # Load model with weights
        logger.info("Fetching Model Weights for ViT")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    else:
        # Load model without weights
        logger.info("Not pretrained for ViT")
        model = timm.create_model('vit_base_patch16_224', num_classes=num_classes)


    if freeze=="True":
        logger.info("Freezing backbone")
        """Freezing Layers"""
        for param in model.parameters():
            param.requires_grad = False
        """Unfreese fc layer"""
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
    


def get_beit(pretrained, num_classes, freeze="True"):

    if pretrained=="True":
        # Load model with weights
        logger.info("Fetching Model Weights for BEiT")
        model = timm.create_model('beit_base_patch16_224', pretrained=True, num_classes=num_classes)

    else:
        # Load model without weights
        logger.info("Not pretrained for BEiT")
        model = timm.create_model('beit_base_patch16_224', num_classes=num_classes)


    if freeze=="True":
        logger.info("Freezing backbone")
        """Freezing Layers"""
        for param in model.parameters():
            param.requires_grad = False
        """Unfreese fc layer"""
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


