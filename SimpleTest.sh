#!/bin/bash
#SBATCH --time=24:00:00 # maximum allocated time
#SBATCH --job-name=Trainbatch_121222 # name of the job
#SBATCH --partition=gpu-unlimited # which partition the job should be scheduled on
#SBATCH --output=./Trainbatch_121222-%j.out
#SBATCH --error=./Trainbatch_121222-%j.err
#SBATCH --gres=gpu:1
##SBATCH -w gpu[26-30]  Only define nodes inside that partition

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print


python3 main.py -s optim -m resnet50 -d cifar10 -e 60 -b 128 -f False -p False -g "True" -t 'Optimizing ResNet50 from scratch on Cifar10'
python3 main.py -s optim -m resnet50 -d cifar10 -e 60 -b 128 -f False -p True -g "True" -t 'Optimizing ResNet50 pretrained on Cifar10'

python3 main.py -s optim -m resnet101 -d cifar10 -e 60 -b 64 -f False -p False -g "True" -t 'Optimizing ResNet101 from scratch on Cifar10 - continued' -c "checkpoints/03.12.22 22-02-50/model_40.pt"
python3 main.py -s optim -m resnet101 -d cifar10 -e 60 -b 64 -f False -p True -g "True" -t 'Optimizing ResNet101 pretrained on Cifar10'

python3 main.py -s optim -m efficientnet -d cifar10 -e 60 -b 128 -f False -p False -g "True" -t 'Optimizing Efficientnet from scratch on Cifar10'
python3 main.py -s optim -m efficientnet -d cifar10 -e 60 -b 32 -f False -p True -g "False" -t 'Optimizing Efficientnet pretrained on Cifar10'

python3 main.py -s optim -m vit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Optimizing ViT from scratch on Cifar10 - continued' -c "checkpoints/06.12.22 11-30-35/best_model.pt"
python3 main.py -s optim -m vit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Optimizing ViT pretrained on Cifar10 - continued' -c "checkpoints/07.12.22 01-19-34/best_model.pt"

python3 main.py -s optim -m beit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Optimizing BEiT from scratch on Cifar10'
python3 main.py -s optim -m beit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Optimizing BEiT pretrained on Cifar10'

## python3 main.py -s train -m resnet101 -d cifar10 -e 60 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet50 from scratch on Cifar10'
## python3 main.py -s train -m resnet101 -d cifar10 -e 60 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet50 pretrained on Cifar10'

## python3 main.py -s train -m resnet101 -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet101 from scratch on Cifar10 - continued' -c "checkpoints/03.12.22 22-02-50/model_40.pt"
## python3 main.py -s train -m resnet101 -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet101 pretrained on Cifar10'

## python3 main.py -s train -m efficientnet -d cifar10 -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training Efficientnet from scratch on Cifar10'
## python3 main.py -s train -m efficientnet -d cifar10 -e 80 -b 32 -o ADAM -lr 0.01 -f False -p True -g "False" -t 'Training Efficientnet pretrained on Cifar10'

## python3 main.py -s train -m vit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ViT from scratch on Cifar10 - continued' -c "checkpoints/06.12.22 11-30-35/best_model.pt"
## python3 main.py -s train -m vit -d cifar10 -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ViT pretrained on Cifar10 - continued' -c "checkpoints/07.12.22 01-19-34/best_model.pt"

## python3 main.py -s train -m beit -d cifar10 -e 80 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training BEiT from scratch on Cifar10'
## python3 main.py -s train -m beit -d cifar10 -e 80 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training BEiT pretrained on Cifar10'



## python3 main.py -s train -m resnet101 -d cifar10_ln -e 60 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet50 from scratch on Cifar10_labelnoise'
## python3 main.py -s train -m resnet101 -d cifar10_ln -e 60 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet50 pretrained on Cifar10_labelnoise'
## python3 main.py -s train -m resnet101 -d cifar10_ln -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet101 from scratch on Cifar10_labelnoise'
## python3 main.py -s train -m resnet101 -d cifar10_ln -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet101 pretrained on Cifar10_labelnoise'
## python3 main.py -s train -m efficientnet -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training Efficientnet from scratch on Cifar10_labelnoise'
## python3 main.py -s train -m efficientnet -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training Efficientnet pretrained on Cifar10_labelnoise'

## python3 main.py -s train -m vit -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ViT from scratch on Cifar10_labelnoise'
## python3 main.py -s train -m vit -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ViT pretrained on Cifar10_labelnoise'

## python3 main.py -s train -m beit -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training BEiT from scratch on Cifar10_labelnoise'
## python3 main.py -s train -m beit -d cifar10_ln -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training BEiT pretrained on Cifar10_labelnoise'




## python3 main.py -s train -m resnet101 -d cifar10_im -e 60 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet50 from scratch on Cifar10_imbalanced'
## python3 main.py -s train -m resnet101 -d cifar10_im -e 60 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet50 pretrained on Cifar10_imbalanced'

## python3 main.py -s train -m resnet101 -d cifar10_im -e 60 -b 64 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ResNet101 from scratch on Cifar10_imbalanced'
## python3 main.py -s train -m resnet101 -d cifar10_im -e 60 -b 64 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ResNet101 pretrained on Cifar10_imbalanced'

## python3 main.py -s train -m efficientnet -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training Efficientnet from scratch on Cifar10_imbalanced'
## python3 main.py -s train -m efficientnet -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training Efficientnet pretrained on Cifar10_imbalanced'

## python3 main.py -s train -m vit -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training ViT from scratch on Cifar10_imbalanced'
## python3 main.py -s train -m vit -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training ViT pretrained on Cifar10_imbalanced'

## python3 main.py -s train -m beit -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p False -g "True" -t 'Training BEiT from scratch on Cifar10_imbalanced'
## python3 main.py -s train -m beit -d cifar10_im -e 40 -b 128 -o ADAM -lr 0.01 -f False -p True -g "True" -t 'Training BEiT pretrained on Cifar10_imbalanced'


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt # debugging datetime print