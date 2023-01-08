#!/bin/bash
#SBATCH --time=12-23:00:00 # maximum allocated time
#SBATCH --job-name=RN50_pt_f # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./ResNet50_pt_f-%j.out
#SBATCH --error=./ResNet50_pt_f-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s optim -m resnet50 -d ham -e 60 -b 64 -f True -p True -g "True" -t 'Optimizing ResNet50 pretrained on Ham'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print