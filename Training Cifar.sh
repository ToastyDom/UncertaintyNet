#!/bin/bash
#SBATCH --time=12-23:00:00 # maximum allocated time
#SBATCH --job-name=T_Cifar # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./Training_Cifar-%j.out
#SBATCH --error=./Training_Cifar-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print

python3 main.py -s train -m resnet50 -d cifar10 -e 76  -lr 0.0002246299309485715 -o RMSProp -sch Cosine -b 64 -f False -p False -g "True" -t 'Training ResNet50 from scratch on CIFAR'
python3 main.py -s train -m resnet50 -d cifar10 -e 79  -lr 2.7482431296973523e-05 -o RMSProp -sch Cosine -b 64 -f False -p True -g "True" -t 'Training ResNet50 pretrained on CIFAR'
python3 main.py -s train -m resnet101 -d cifar10 -e 53  -lr 0.00019227309424128868 -o RMSProp -sch Cycle -b 64 -f False -p False -g "True" -t 'Training ResNet101 from scratch on Cifar'
python3 main.py -s train -m resnet101 -d cifar10 -e 72  -lr 4.328613116663016e-05 -o ADAM -sch Cycle -b 64 -f False -p True -g "True" -t 'Training ResNet101 pretrained on Cifar'
python3 main.py -s train -m efficientnet -d cifar10 -e 78  -lr 0.00166184128354641048 -o RMSProp -sch Cycle -b 64 -f False -p False -g "True" -t 'Training Efficientnet from scratch on Cifar'
python3 main.py -s train -m efficientnet -d cifar10 -e 24  -lr 1.717405379416517e-05 -o ADAM -sch Cosine -b 64 -f False -p True -g "True" -t 'Training Efficientnet pretrained on Cifar'

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print