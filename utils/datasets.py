import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from loguru import logger
from glob import glob
import os
import pandas as pd
from PIL import Image

# torch.manual_seed(0)
# np.random.seed(0)

class HAM10000(Dataset):
    """Pytroch dataloader for the HAM Dataset"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def get_ham10000(setup):
    logger.info("Gettng Ham10000 Dataset!")
    train_transform = T.Compose([T.Resize((224,224)),  #resises the image so it can be perfect for our model.
                                 T.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                 T.RandomRotation(10),     #Rotates the image to a specified angel
                                 T.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                 T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                 T.ToTensor(), # comvert the image to tensor so that it can work with torch
                                 T.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)) #Normalize all the images
                                 ])

    transform = T.Compose([T.ToTensor(),
                           T.Resize((224,224)),
                           T.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])


    # Retrieve all images
    data_dir = "Ham10000"
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    # Get data .csv and add new info to dataframe
    df_original = pd.read_csv(os.path.join(data_dir, 'Ham_data.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['benign_malignant']).codes

    # Get train val and testset
    df_train = df_original[(df_original["set"]=="train") & (df_original["fold"]==0.0)]
    df_test = df_original[(df_original["set"]=="test")]
    df_val = df_original[(df_original["set"]=="val") & (df_original["fold"]==0.0)]

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()

    trainset = HAM10000(df_train, transform=train_transform)
    validationset = HAM10000(df_val, transform=train_transform)
    testset = HAM10000(df_test, transform=transform)

    num_classes = 2

    return trainset, validationset, testset, num_classes


    





def get_cifar_10(setup):
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

    # Split training set into sets for training and validation for optimizing processes
    if setup == "optim":
        logger.info("Splitting trainset into train and validation")
        trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])
    else:
        validationset = None

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes




def count_data(dataset):
    # Count current data
    classes = list(dataset.classes)
    class_counter = np.zeros(len(classes))
    for data, label in dataset:
        class_counter[label] +=1
    print(class_counter)
    logger.info("Class counter: ", class_counter)




def make_imbalanced(dataset, percentages): 

    # Count current data
    classes = list(dataset.classes)
    class_counter = np.zeros(len(classes))
    for data, label in dataset:
        class_counter[label] +=1

    # Percentages we want per class
    percentage = percentages
    # Create a list of the number of images for each class
    num_images = class_counter * percentage   # [5000. 5000. 5000. 4000. 2500. 1000. 1000. 1000.  500.  500.]

    # Create a list of indices for the total number of images
    indices = list(range(len(dataset)))

    """ We want to go through all the data. If the data matches the current class, put it in a 
    tmp list. And then just save a subsection of it to our dataset"""
    class_indices = []
    for i in range(len(classes)):  # i ist jeweils eine Klasse
        data_from_this_class = []
        for j in range(len(dataset)):   # j ist jeweils ein data label paar aus dem Datensatz
            if dataset[j][1] == i:  # Wenn die aktuelle Klasse mit der Datei übereinstimmt
                data_from_this_class.append(indices[j]) # wir wissen dass element j zu dieser Klasse dazu gehört
        
        class_indices.append(data_from_this_class[:num_images[i]])  # Nimm aber nur so viele wie oben angegeben


    # Flatten the list
    class_indices = [item for sublist in class_indices for item in sublist]

    # Create a new dataset with the new indices
    imbalanced_dataset = torch.utils.data.Subset(dataset, class_indices)

    count_data(imbalanced_dataset)


    return imbalanced_dataset


def get_cifar_10_imbalanced(setup):
    logger.info("CIFAR10 with imbalancy")
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

    # Make dataset imbalanced
    percentages = [1, 1, 1, 0.8, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1]
    trainset = make_imbalanced(trainset, percentages)

    # Split training set into sets for training and validation for optimizing processes
    if setup == "optim":
        logger.info("Splitting trainset into train and validation")
        trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])
    else:
        validationset = None

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Make dataet imbalances
    testset = make_imbalanced(testset, percentages)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes



def get_cifar_10_scarcity(setup):
    logger.info("CIFAR10 with scarcity")
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

    # Make dataset imbalanced
    percentages = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    trainset = make_imbalanced(trainset, percentages)

    # Split training set into sets for training and validation for optimizing processes
    if setup == "optim":
        logger.info("Splitting trainset into train and validation")
        trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])
    else:
        validationset = None

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Make dataet imbalances
    testset = make_imbalanced(testset, percentages)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes



def get_cifar_10_scarcity_ib(setup):
    logger.info("CIFAR10 with scarcity and imbalancy")
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

    # Make dataset imbalanced
    percentages = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    trainset = make_imbalanced(trainset, percentages)

    # Split training set into sets for training and validation for optimizing processes
    if setup == "optim":
        logger.info("Splitting trainset into train and validation")
        trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])
    else:
        validationset = None

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Make dataet imbalances
    testset = make_imbalanced(testset, percentages)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes





def get_cifar_10_label_noise(setup):
    logger.info("CIFAR10 with label Noise")
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



    # Implement label noise
    for data in trainset:
        if random.random() < 0.2:
            # Randomly select a label not equal to the correct one
            incorrect_label = random.choice([x for x in range(10) if x != data[1]])
            data = (data[0], incorrect_label)


    # Split training set into sets for training and validation for optimizing processes
    if setup == "optim":
        logger.info("Splitting trainset into train and validation")
        trainset, validationset = torch.utils.data.random_split(trainset, [49000, 1000])
    else:
        validationset = None

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Implement label noise
    for data in testset:
        if random.random() < 0.2:
            # Randomly select a label not equal to the correct one
            incorrect_label = random.choice([x for x in range(10) if x != data[1]])
            data = (data[0], incorrect_label)
    
    num_classes = 10

    return trainset, validationset, testset, num_classes