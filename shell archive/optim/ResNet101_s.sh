#!/bin/bash
#SBATCH --time=12-23:00:00 # maximum allocated time
#SBATCH --job-name=RN101_s # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./ResNet101_s-%j.out
#SBATCH --error=./ResNet101_s-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s optim -m resnet101 -d ham -e 60 -b 64 -f False -p False -g "True" -t 'Optimizing ResNet101 pretrained on Ham10000'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print