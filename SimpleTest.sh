#!/bin/bash
#SBATCH --time=24:00:00 # maximum allocated time
#SBATCH --job-name=Trainbatch_021222 # name of the job
#SBATCH --partition=gpu-12h # which partition the job should be scheduled on
#SBATCH --output=./Trainbatch_021222-%j.out
#SBATCH --error=./Trainbatch_021222-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s train -m efficientnet -d cifar10 -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training Efficientnet from scratch on Cifar10'
python3 main.py -s train -m efficientnet -d cifar10 -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training Efficientnet pretrained on Cifar10'

python3 main.py -s train -m vit -d cifar10 -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ViT from scratch on Cifar10'
python3 main.py -s train -m vit -d cifar10 -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ViT pretrained on Cifar10'

##python3 main.py -s train -m resnet50 -d cifar10 -e 25 -b 128 -o ADAM -lr 0.01 -f False -p False -t 'Training with settings from google'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print