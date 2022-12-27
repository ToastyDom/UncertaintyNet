#!/bin/bash
#SBATCH --time=99:00:00 # maximum allocated time
#SBATCH --job-name=EN_s # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./EN_s-%j.out
#SBATCH --error=./EN_s-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s optim -m efficientnet -d cifar10 -e 60 -b 32 -f False -p False -g "True" -t 'Optimizing Efficientnet from scratch on Cifar10'
##python3 main.py -s optim -m efficientnet -d cifar10 -e 60 -b 32 -f False -p True -g "True" -t 'Optimizing Efficientnet pretrained on Cifar10'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print