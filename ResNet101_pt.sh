#!/bin/bash
#SBATCH --time=99:00:00 # maximum allocated time
#SBATCH --job-name=ResNet101_pt # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./ResNet101_pt-%j.out
#SBATCH --error=./ResNet101_pt-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print


##python3 main.py -s optim -m resnet101 -d cifar10 -e 60 -b 64 -f False -p False -g "True" -t 'Optimizing ResNet101 from scratch on Cifar10' 
python3 main.py -s optim -m resnet101 -d cifar10 -e 60 -b 64 -f False -p True -g "True" -t 'Optimizing ResNet101 pretrained on Cifar10'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print