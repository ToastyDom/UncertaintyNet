""" 

Finding out if parameters are set to training:

for param in self.model.parameters():
            print(param.requires_grad)
=> Can also be used to freeze some layers



Optuna:
https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837


Der hat gute Ergebnisse mit CIFAR erreicht
https://github.com/kuangliu/pytorch-cifar


"""