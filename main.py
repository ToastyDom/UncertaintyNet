import argparse
#import logging
from loguru import logger
import torch
from datetime import datetime

from utils.datasets import get_cifar_10, get_mnist
from utils.models import get_ResNet50

from training import TrainUncertainty



def parse_args():
    """ arguments for processing pipeline """
    parser = argparse.ArgumentParser(
        description="Training Pipeline"
    )
    parser.add_argument(
        "-s",
        "--setup",
        type=str,
        help="Training or Finetuning?",
        default="training",
        choices=["training", "finetuning"],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Select a model which should be trained",
        default="ResNet50",
        choices=["ResNet50"]
    )    
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Training on which dataset",
        default="Cifar10",
        choices=["cifar10", "mnist"]
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="TRACE",
        choices=["ERROR", "WARN", "INFO", "DEBUG", "TRACE"],
        help="Log level.",
        required=False,
    )
    
    return parser.parse_args()





def main(args):

    setup = args.setup
    dataset = args.dataset
    model = args.model
    epochs = args.epochs

    # Create logging state
    now = datetime.now() # current time
    now_formatted = now.strftime("%d.%m.%y %H:%M:%S")
    settings = {"time": now_formatted,
                "title": "TO DO",
                "setup": setup,
                "model": model,
                "dataset": dataset,
                "batchsize": "TODO"}


    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select Dataset
    logger.info("Loading Dataset")
    if dataset == "cifar10":
        trainset, testset = get_cifar_10()
    elif dataset == "mnist":
        trainset, testset = get_mnist()
    else:
        logger.warning("No dataset selected. Will select Cifar10")
        trainset, testset = get_cifar_10()

    # Select Model
    logger.info("Loading Model")
    if model == "ResNet50":
        torchmodel = get_ResNet50(pretrained=True)
    else:
        logger.warning("No model selected. Will select ResNet50")
        torchmodel = get_ResNet50(pretrained=True)



    # Start Pipeline
    logger.info("Starting Pipeline")
    pipeline = TrainUncertainty(settings=settings,
                                device=device, 
                                model=torchmodel, 
                                trainset=trainset, 
                                testset=testset, 
                                batch_size=32)


    logger.info("Starting Training")
    history = pipeline.train(num_epochs = epochs)


    pass




if __name__ == "__main__":
    args = parse_args()
    main(args)