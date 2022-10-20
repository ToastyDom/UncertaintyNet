import argparse
import logging
import torch

from utils.datasets import get_cifar_10
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
        choices=["Cifar10"]
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


    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select Dataset
    if dataset == "Cifar10":
        trainset, testset = get_cifar_10()
    else:
        logging.warning("No dataset selected. Will select Cifar10")
        trainset, testset = get_cifar_10()

    # Select Model
    if model == "ResNet50":
        torchmodel = get_ResNet50(pretrained=True)
    else:
        logging.warning("No model selected. Will select ResNet50")
        torchmodel = get_ResNet50(pretrained=True)



    # Start Pipeline
    pipeline = TrainUncertainty(device=device, 
                     model=torchmodel, 
                     trainset=trainset, 
                     testset=testset, 
                     batch_size=8)


    history = pipeline.train(num_epochs = epochs)


    pass




if __name__ == "__main__":
    args = parse_args()
    main(args)