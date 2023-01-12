#!/bin/bash
#SBATCH --time=12-23:00:00 # maximum allocated time
#SBATCH --job-name=T_Ham # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./Training_Ham-%j.out
#SBATCH --error=./Training_Ham-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s train -m resnet101 -d ham -e 120  -lr 0.0001861145738540473 -o RMSProp -sch Cosine -b 64 -f False -p False -g "True" -t 'Training ResNet101 from scratch on HAM longer'
python3 main.py -s train -m resnet101 -d ham -e 120  -lr 6.772282588031143e-05 -o ADAM -sch Cosine -b 64 -f False -p True -g "True" -t 'Training ResNet101 pretrained on HAM longer'
python3 main.py -s train -m resnet50 -d ham -e 120  -lr 0.00012726285389791345 -o RMSProp -sch Cosine -b 64 -f False -p False -g "True" -t 'Training ResNet50 from scratch on HAM longer'
python3 main.py -s train -m resnet50 -d ham -e 120  -lr 3.3971212678889315e-05 -o RMSProp -sch Cycle -b 64 -f False -p True -g "True" -t 'Training ResNet50 pretrained on HAM longer'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print