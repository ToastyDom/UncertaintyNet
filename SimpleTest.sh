#!/bin/bash
#SBATCH --time=24:00:00 # maximum allocated time
#SBATCH --job-name=TrainResNet50_Scratch # name of the job
#SBATCH --partition=gpu-12h # which partition the job should be scheduled on
#SBATCH --output=./TrainResNet50_Scratch-%j.out
#SBATCH --error=./TrainResNet50_Scratch-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s train -m resnet50 -d cifar10 -e 25 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training with settings from google'
##python3 main.py -s train -m resnet50 -d cifar10 -e 25 -b 128 -o ADAM -lr 0.01 -f False -p False -t 'Training with settings from google'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print