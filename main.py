import argparse
import warnings
from datetime import datetime
import os
import numpy as np

import optuna
import torch
#import logging
from loguru import logger

from training import TrainUncertainty
from utils.datasets import get_cifar_10
from utils.models import get_ResNet101, get_ResNet50, get_efficientnet

warnings.filterwarnings("ignore", category=UserWarning) 

def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


# gpu_idx = get_free_gpu_idx()
# print("Using GPU #%s" % gpu_idx)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


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
        choices=["train", "optim", "test", "plot"],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Select a model which should be trained",
        default="resnet50",
        choices=["resnet50", "resnet101", "efficientnet"]
    )

    parser.add_argument(
        "-p",
        "--pretrained",
        type=str,
        help="Should the model use pretrained weights",
        default="True",
        choices=["True", "False"]
    )  

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="Which checkpoint to add",
        required=False
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
        "--learningrate",
        "-lr",
        type=float,
        help="Which learning rate to choose",
    )

    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        help="Which Optimizer to choose",
        choices=["SGD", "ADAM"]
    )

    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        help="Amount of batches",
    )

    parser.add_argument(
        "--freeze",
        "-f",
        type=str,
        default="True",
        choices=["True", "False"],
        required=False
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        required=False
    )

    parser.add_argument(
        "--gpuserver",
        "-g",
        type=str,
        required=False
    )
    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=0,
        required=False
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
    """Main function that starts pip

    Args:
        args (parser): parser arguments
    """

    if args.gpuserver == "True":
        gpu_idx = get_free_gpu_idx()
        print("Using GPU #%s" % gpu_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    setup = args.setup
    dataset = args.dataset
    model = args.model
    pretrained = args.pretrained
    epochs = args.epochs
    batchsize = args.batchsize
    freeze = args.freeze
    checkpoint = args.checkpoint
    title = args.title
    learningrate = args.learningrate
    optimizer = args.optimizer
 
    # Create logging state
    now = datetime.now() # current time
    now_formatted = now.strftime("%d.%m.%y %H:%M:%S")
    settings = {"time": now_formatted,
                "title": title,
                "setup": setup,
                "model": model,
                "pretrained": pretrained,
                "optimizer": optimizer,
                "learningrate": learningrate,
                "dataset": dataset,
                "freeze": freeze,
                "batchsize": batchsize}


    # torch.manual_seed(0)
    # np.random.seed(0)


    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: ", device)
    print(device)

    # Select Dataset
    logger.info("Loading Dataset")
    if dataset == "cifar10":
        trainset, validationset, testset, num_classes = get_cifar_10(setup)
    else:
        logger.warning("No dataset selected. Will select Cifar10")
        trainset, validationset, testset, num_classes = get_cifar_10(setup)

    # Select Model
    logger.info("Loading Model")
    if model == "resnet50":
        torchmodel = get_ResNet50(pretrained=pretrained, freeze=freeze, num_classes=num_classes)
    elif model == "resnet101":
        torchmodel = get_ResNet101(pretrained=pretrained, freeze=freeze, num_classes=num_classes)
    elif model == "efficientnet":
        torchmodel = get_efficientnet(pretrained=pretrained, freeze=freeze, num_classes=num_classes)
    else:
        logger.warning("No model selected. Will select ResNet50")
        torchmodel = get_ResNet50(pretrained=True,freeze=freeze, num_classes=num_classes)




    if setup == "optim":
        logger.info("Starting Optimiation")
        pipeline = TrainUncertainty(settings=settings,
                                    device=device, 
                                    model=torchmodel,
                                    num_classes=num_classes, 
                                    trainset=trainset,
                                    validationset=validationset, 
                                    testset=testset, 
                                    batch_size=None,
                                    optimizer=None,
                                    learningrate=None)
        pipeline.hypersearch = True
            
        history = pipeline.hyper_optimizer(num_trials=epochs)
        pass
    else:
        # Start Pipeline
        logger.info("Starting Pipeline")
        pipeline = TrainUncertainty(settings=settings,
                                    device=device, 
                                    model=torchmodel,
                                    num_classes=num_classes, 
                                    trainset=trainset,
                                    validationset=validationset, 
                                    testset=testset, 
                                    batch_size=batchsize,
                                    optimizer=optimizer,
                                    learningrate=learningrate)



        if checkpoint is not None:
            logger.info("Loading Checkpoint")
            pipeline.load(checkpoint)


        if setup == "train":
            logger.info("Starting Training")
            history = pipeline.train(num_epochs = epochs)

            
        


        elif setup == "test":
            logger.info("Starting Testing")
            history = pipeline.test()

        elif setup == "plot":
            logger.info("Starting Plotting")
            pipeline.plot()



    pass




if __name__ == "__main__":
    args = parse_args()
    main(args)