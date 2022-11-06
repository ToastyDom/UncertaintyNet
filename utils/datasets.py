import torch
import torchvision
from torchvision import transforms as T




def get_cifar_10():
    train_transform = T.Compose([T.Resize((224,224)),  #resises the image so it can be perfect for our model.
                                 T.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                 T.RandomRotation(10),     #Rotates the image to a specified angel
                                 T.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                 T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                 T.ToTensor(), # comvert the image to tensor so that it can work with torch
                                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                                 ])

    transform = T.Compose([T.ToTensor(),
                           T.Resize((224,224)),
                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

    # Split training set into sets for training and validation.
    trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes


