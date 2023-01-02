#!/bin/bash
#SBATCH --time=12-23:00:00 # maximum allocated time
#SBATCH --job-name=EN_pt # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./EN_pt-%j.out
#SBATCH --error=./EN_pt-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s optim -m efficientnet -d ham -e 60 -b 32 -f False -p True -g "True" -t 'Optimizing Efficientnet pretrained on Ham'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print