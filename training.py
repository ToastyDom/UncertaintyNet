import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import shutil
import torch.nn.functional as F
from utils.evaluation import ece_score, auroc_score, sensitivity_specificity, balanced_acc_score, calibration_error, top_calibration_error, brier_multi #, sensitivity_score
import matplotlib.pyplot as plt
import optuna
import torch.optim as optim
from loguru import logger
from utils.models import get_ResNet18, get_ResNet50


""" 
Parameter die ich setzen können möchte:

Welches Netzwerk ich wählen möchte
Ob dieses Netzwerk pretrained ist
Welches Dataset zum traineren
Welcher Optimizer, welcher Criterion

Später ob fine-tuning oder nicht
Vlt noch settings.json einfügen für leanring rate oder so

"""


def ensure_directory(path):
    """Method to ensure directory exists

    Args:
        path (str): path to folder to create
    """
    if not os.path.exists(path):
        os.mkdir(path)



def update_logs(settings, best=False, optim_data={}):
    """Function to update logs of the training configuration

    Args:
        settings (dict): dict with the current settings
        best (bool, optional): If this is the best model. Defaults to False.
    """

    with open("logs/training_logs.json") as json_file:
        log_json = json.load(json_file)

        log_time = settings["time"]

        # If current time does not exist yet in the log
        if log_time not in log_json:
            log_json[log_time] = {}

        log_json[log_time]["title"] = settings["title"]
        log_json[log_time]["setup"] = settings["setup"]
        log_json[log_time]["model"] = settings["model"]
        log_json[log_time]["freeze"] = settings["freeze"]
        log_json[log_time]["dataset"] = settings["dataset"]
        log_json[log_time]["batchsize"] = settings["batchsize"]
        log_json[log_time]["optimizer"] = settings["optimizer"]
        log_json[log_time]["scheduler"] = settings["scheduler"]
        log_json[log_time]["current_epoch"] = settings["current_epoch"]
        log_json[log_time]["current_checkpoint"] = settings["current_checkpoint"]
        log_json[log_time]["best_checkpoint"] = settings["best_checkpoint"]
        log_json[log_time]["optim_data"] = optim_data
        log_json[log_time]["history"] = settings["history"]


        # If this is the best model. save that in the config
        if "best_config" not in log_json[log_time]:
            log_json[log_time]["best_config"] = {}

        if best:
            log_json[log_time]["best_config"]["this_epoch"] = settings["current_epoch"]
            log_json[log_time]["best_config"]["this_train_accuracy"] = settings["history"]["training_accuracy"][-1]
            log_json[log_time]["best_config"]["this_test_accuracy"] = settings["history"]["testing_accuracy"][-1]
            log_json[log_time]["best_config"]["this_balanced_test_accuracy"] = settings["history"]["balanced_accuracy"][-1]
            log_json[log_time]["best_config"]["this_ece"] = settings["history"]["ece"][-1]
            log_json[log_time]["best_config"]["this_nll"] = settings["history"]["nll"][-1]
            log_json[log_time]["best_config"]["this_brier"] = settings["history"]["brier"][-1]
            log_json[log_time]["best_config"]["this_calib_error"] = settings["history"]["calib_error"][-1]
            log_json[log_time]["best_config"]["this_top_calib_error"] = settings["history"]["top_calib_error"][-1]
            log_json[log_time]["best_config"]["this_auroc"] = settings["history"]["auroc"][-1]
            log_json[log_time]["best_config"]["this_sensitivty"] = settings["history"]["sensitivity"][-1]
            log_json[log_time]["best_config"]["this_specificity"] = settings["history"]["specificity"][-1]

    # save changes
    with open("logs/training_logs.json", "w") as json_file:
        json_file.seek(0) # set pointer to file beginning
        json.dump(log_json, json_file, indent=4)
        json_file.truncate()

        print("")
        print("### Updated Logs ###")
        print("")



def backup_logs(settings):
    """Method to create backup of the log after each epoch so we dont loose data

    Args:
        settings (dict): dict of current settings
    """

    # make sure backup folder exists
    ensure_directory("./logs/backups")

    # make a backup
    log_time = settings["time"]
    log_time = log_time.replace(":","-")
    shutil.copyfile("logs/training_logs.json", f"logs/backups/{log_time}.json")



class TrainUncertainty:

    def __init__(self, settings, device, model, num_classes, trainset, validationset, testset, batch_size, optimizer, learningrate):
        """Training class to train the models

        Args:
            settings (dict): dict with current settings
            device (str): either cpu or gpu
            model (torch): torch model to train
            trainset (torch dataset): Trainingset
            testset (torch dataset): Testingset
            batch_size (int, optional): Batches for training. Defaults to 64.
        """

        self.device = device
        self.model = model
        self.num_classes = num_classes
        self.trainset = trainset
        self.validationset = validationset
        self.testset = testset
        self.batch_size = batch_size
        self.str_optimizer = optimizer
        self.learningrate = learningrate
        self.settings = settings
        self.this_epoch = 0
        self.best_accuracy = 0.0
        self.hypersearch = False

        
        # Select Optimizer
        if self.str_optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learningrate, momentum=0.9, weight_decay=5e-4)
        elif self.str_optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningrate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        else:
            logger.warning("No Optimizer selected! And therefore no learningrate")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


        # Update Settings
        self.settings["optimizer"] = str(self.optimizer)


        # Select Criterion
        self.criterion = nn.CrossEntropyLoss()


        self.nll_loss = nn.NLLLoss()


        # Select Scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.learningrate, epochs=epochs, steps_per_epoch=len(train_loader))


        # Initiate Dataloades
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Model to Device
        self.model.to(self.device)

        # Some bookkeeping:
        self.iter = 0
        self.history = {
            'training_loss': [],
            'training_accuracy': [],
            'testing_loss': [],
            'ece': [],
            'nll': [],
            'brier': [],
            'calib_error': [],
            'top_calib_error': [],
            'auroc': [],
            'sensitivity': [],
            'specificity': [],
            'testing_accuracy': [],
            'balanced_accuracy': []
        }

        ensure_directory("checkpoints")


    def objective(self, trial):

        
        pretrained = self.settings["pretrained"]
        freeze = self.settings["freeze"]
        num_classes = self.num_classes
        model = self.settings["model"]

        if model == "resnet50":
            logger.info("Loading ResNet50 new")
            self.model = get_ResNet50(pretrained=pretrained, freeze=freeze, num_classes=num_classes)
        else:
            logger.error("No model available")

        # Model to Device
        self.model.to(self.device)

        self.optim_params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1,log=True),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                'batchsize': trial.suggest_categorical("batchsize", [32,64,128,256,512]),
                'momentum': trial.suggest_float("momentum", 0.0, 1.0, step=0.1),
                'weight_dacay': trial.suggest_float('weight_dacay', 1e-6, 1e-1, log=True)
                }
    
        
        self.trial = trial
        accuracy = self.train(num_epochs=10)
    

        return accuracy


    def load(self, checkpoint):
        """
        Load stored checkpoint to resume training.

        Args:
            checkpoint (str): Path to stored training state.

        """
        self.model.to(self.device)

        state = torch.load(checkpoint)

        # Copy trained parameters into models.
        self.model.load_state_dict(state['model'])

        # Copy optimizer states.
        self.optimizer.load_state_dict(state['optimizer'])

        # Copy history.
        self.history = state['history']

        # Copy epoch and iteration trackers.
        self.epoch = state['epoch']

        # Load best accuracy
        self.best_accuracy = state['best_acc']


    def save(self, best=False):
        """Function to save current state of training

        Args:
            best (bool, optional): If this is the best model. Defaults to False.
        """

        # make sure backup folder exists
        ensure_directory("./checkpoints")

        state = {

            # Save network parameters.
            'model': self.model.state_dict(),

            # Save optimizer state.
            'optimizer': self.optimizer.state_dict(),

            # Save epochs and iterations.
            'epoch': self.this_epoch,

            # Save previous training history.
            'history': self.history,

            # Save settings for logging
            'settings': self.settings,

            # Best accuracy
            'best_acc': self.best_accuracy

        }

        if best:
            torch.save(state, self.model_path)
            torch.save(state, self.best_path)
        else:
            torch.save(state, self.model_path)


    def test(self, validation=False):
        """Testing / Validation part of the framework. 
        Loads batches of the data and tests the output

        Returns:
            running_validation_loss: Validaiton loss
            accuracy: Accuracy
            ece: Expected Calibration Error
        """
        self.model.eval()

        # Set progress bar format.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'

        # logs
        running_validation_loss = 0.0
        running_nll_loss = 0.0
        num_correct = 0
        num_examples = 0
        all_predicitons = np.array([],dtype=int)
        all_labels = np.array([],dtype=int)
        all_labels_tf = []
        all_outputs = []

        if self.hypersearch == True:
            this_dataloader = self.valloader
            logger.info("Using Validationset")
        else:
            this_dataloader = self.testloader
            logger.info("Using Testingset")

        

        with torch.no_grad():
            for iteration, (inputs, labels) in tqdm(enumerate(this_dataloader), bar_format=bar_format, total=len(this_dataloader)):

                    # Inputs and labels to device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Feed network the images
                    outputs = self.model(inputs)

                    # Calculate loss
                    loss = self.criterion(outputs,labels)
                    nll_loss = self.nll_loss(outputs,labels)

                    # Update loss
                    running_validation_loss += loss.data.item() * inputs.size(0)
                    running_nll_loss += nll_loss.data.item() * inputs.size(0)

                    # Predicted Labels
                    prob_output = F.softmax(outputs, dim=1).cpu().data.numpy()  # prediction probability, Eigentlich auch confidence
                    pred_labels = torch.max(F.softmax(outputs, dim = 1), dim = 1)[1]

                    # Count correct ones
                    correct = torch.eq(pred_labels, labels).view(-1)
                    num_correct += torch.sum(correct).item()
                    num_examples += correct.shape[0]

                    # Collect all outputs
                    pred_labels = pred_labels.cpu()
                    outputs = outputs.cpu()
                    labels = labels.cpu()
                    all_predicitons = np.concatenate((all_predicitons,pred_labels))
                    all_labels = np.concatenate((all_labels,labels))

                    
                    if all_outputs == []:
                        all_outputs = outputs
                    else:
                        all_outputs = torch.cat([all_outputs, outputs], dim=0)

                    if all_labels_tf == []:
                        all_labels_tf = labels
                    else:
                        all_labels_tf = torch.cat([all_labels_tf, labels], dim=0)

                    
     
        
        


        # Calculate Errors 
        all_preds = F.softmax(all_outputs, dim=1).cpu().data.numpy()
        ece = ece_score(all_preds, np.array(all_labels))
        calib_error = calibration_error(all_preds, np.array(all_labels))
        top_calib_error = top_calibration_error(all_preds, np.array(all_labels))
        brier = brier_multi(all_preds, np.array(all_labels), self.num_classes)
        running_nll_loss /= len(this_dataloader.dataset)
        
        # Validation loss
        running_validation_loss /= len(this_dataloader.dataset)

        # Accuracy
        accuracy = num_correct / num_examples

        # Balanced Accuracy
        balanced_acc = balanced_acc_score(all_predicitons, all_labels)

        # AUROC
        auroc = auroc_score(all_outputs, all_labels_tf, self.num_classes)

        # Specificity & Sensitivity
        sensitivity, specificity = sensitivity_specificity(all_outputs, all_labels, self.num_classes)


        
        print(f'testing_loss: {running_validation_loss}')
        print(f'ece: {ece}')
        print(f"nll: {running_nll_loss}")
        print(f'brier: {brier}')
        print(f'calib_error: {calib_error}')
        print(f'top_calib_error: {top_calib_error}')
        print(f'specificity: {specificity}')
        print(f'sensitivity: {sensitivity}')
        print(f'auroc: {auroc}')
        print(f'accuracy: {accuracy}')
        print(f'balanced_accuracy: {balanced_acc}')
        print("")

        return running_validation_loss, ece, running_nll_loss, brier, calib_error, top_calib_error, specificity, sensitivity, auroc, accuracy, balanced_acc


    def plot(self):
        """Function to plot examples of the dataset along with confidence and prediciton scores
        """

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for iteration, (inputs, labels) in enumerate(self.testloader):
                
                # Inputs and labels to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Run input through model
                outputs = self.model(inputs)  

                # Calculate relevant data from output
                pred_probs = F.softmax(outputs, dim=1).cpu().data.numpy()  # prediction probability, Eigentlich auch confidence
                values, indices = max_outputs = torch.max(outputs, 1)  # indice of classifier
                indices = np.array(indices)
                labels = np.array(labels)

                # Which are correctly labled?
                classified = [indices == labels]  
                correct = sum(sum(classified)) # True is equal to 1 
                accuracy = correct/len(labels)
                print(accuracy)

                # Plot image
                fig, axs = plt.subplots(int(np.ceil(len(labels)/4)), 4, figsize=(20, 5))
                
                counter = 0
                row = 0
                column = 0
                for i, item in enumerate(range(len(labels))):

                    # Current axis
                    ax = axs[row][column]

                    # Find index of prediction
                    predicted_label_index = indices[i]

                    # Set title with relevant info
                    ax.set_title("Logits: {conf:.3f}, Confidence: {prob:.0f}%, {lbreak} Predicted Label: {index}, Actual Label: {label}".format(conf =outputs[i][predicted_label_index ],
                                                                                                                                                      prob = pred_probs[i][predicted_label_index ],
                                                                                                                                                      lbreak = "\n",
                                                                                                                                                      index = indices[i],
                                                                                                                                                      label = labels[i]))
                    
                    # Transpose the image from dataloader format into the format that is required by plt
                    img = np.transpose(inputs[i].cpu().detach().numpy(), (1, 2, 0)) 
                    img =(img+1)/2  # datatype is float, so we need to change the range

                    # Add image to plot
                    ax.imshow(img)

                    # Depending on amount of images go to next row
                    counter += 1
                    column += 1
                    if counter == 4:
                        row += 1
                        counter = 0
                        column = 0

                fig.subplots_adjust(hspace=1.4)
                plt.rcParams.update({'font.size': 6})
                plt.savefig("test.png")
                break
   

    def set_hyperparameter(self):

        #self.load("checkpoints/03.11.22 22-33-55/best_model.pt")

        # Initiate optimizer
        optimizer_name = self.optim_params['optimizer']
        if optimizer_name == "Adam":
            logger.info("Loading Adam")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optim_params['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=self.optim_params['weight_dacay'], amsgrad=False)
        elif optimizer_name == "SGD":
            logger.info("Loading SGD")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.optim_params['learning_rate'], momentum=self.optim_params['momentum'])
        elif optimizer_name == "RMSprop":
            logger.info("Loading RMSprop")
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.optim_params['learning_rate'])
        else:
            logger.error("No optimizer found")
            self.optimizer = getattr(optim, self.optim_params['optimizer'])(self.model.parameters(), lr= self.optim_params['learning_rate'])
        
        # Initiate Dataloader
        self.batch_size = self.optim_params['batchsize']
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.optim_params['batchsize'], shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(self.validationset, batch_size=self.optim_params['batchsize'], shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.optim_params['batchsize'], shuffle=True, num_workers=2)
        
        # Initiate Schedular
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def train(self, num_epochs):
        """Function to train the model

        Args:
            num_epochs (int): amount of epochs to train
        Returns:
            self.history: training history
        """

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.learningrate, epochs=num_epochs, steps_per_epoch=len(self.trainloader))
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.settings["scheduler"] = str(self.scheduler)


        if self.hypersearch == True:
            logger.info("Changing Dataset for Hyperparamter serach")
            self.set_hyperparameter()
            

        # Set progress bar format.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
        
        # For each epoch
        for epoch in range(num_epochs):
            running_training_loss = 0.0
            running_corrects = 0
            running_validation_loss = 0.0


            ############
            # Training #
            ############

            self.model.train()
            for iteration, (inputs, labels) in tqdm(enumerate(self.trainloader), bar_format=bar_format, total=len(self.trainloader)):

                # Inputs and labels to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Feed network the images
                outputs = self.model(inputs)

                # Get classifications
                _, preds = torch.max(outputs, 1)

                # Calculate loss
                loss = self.criterion(outputs,labels)

                # Backprop with that loss
                loss.backward()

                # Optimiter step TODO: What does that mean?
                self.optimizer.step()

                # Store loss
                running_training_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                #print("Learning Rate Batch:", self.optimizer.param_groups[0]["lr"])
                #self.scheduler.step()
            

            # Print LR
            print("Learning Rate Epoch:", self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()

            # Safe Training loss
            running_training_loss /= len(self.trainloader.dataset)
            training_accuracy = np.float64(running_corrects.double() / len(self.trainloader.dataset))

            self.history['training_loss'].append(running_training_loss)
            self.history['training_accuracy'].append(training_accuracy)
            print(f'epoch: {epoch + 1}, training_loss: {running_training_loss}, training_acc: {training_accuracy}')



            ##############
            # Testing #
            ##############

            running_validation_loss, ece, nll, brier, calib_error, top_calib_error, specificity, sensitivity, auroc, test_accuracy, balanced_acc = self.test()


            ##############
            # Save state #
            ##############
           
            # Save state in history
            self.history['testing_loss'].append(running_validation_loss)
            self.history['ece'].append(ece)
            self.history['nll'].append(nll)
            self.history['brier'].append(brier)
            self.history['calib_error'].append(calib_error)
            self.history['top_calib_error'].append(top_calib_error)
            self.history['specificity'].append(specificity)
            self.history['sensitivity'].append(sensitivity)
            self.history['auroc'].append(auroc)
            self.history['testing_accuracy'].append(test_accuracy)
            self.history['balanced_accuracy'].append(balanced_acc)
            
            
            
            

            # Save training state
            log_time = self.settings["time"]
            log_time = log_time.replace(":", "-")
            self.this_epoch += 1
            ensure_directory(f"checkpoints/{log_time}")
            self.model_path = f'checkpoints/{log_time}/model_{str(self.this_epoch)}.pt'
            self.best_path = f'checkpoints/{log_time}/best_model.pt'

            # if this is currently the best model, save that accordingly
            if balanced_acc > self.best_accuracy:
                self.best = True
                self.best_accuracy = balanced_acc
                self.save(best=True)
            else:
                self.best = False
                self.save()


            # Update Logging
            self.settings["current_epoch"] = epoch
            self.settings["current_checkpoint"] = self.model_path
            self.settings["best_checkpoint"] = self.best_path
            self.settings["history"] = self.history
            update_logs(self.settings, self.best)


        backup_logs(self.settings)

        # self.scheduler.step()

        # If we use optuna, prune if bad
        if self.hypersearch == True:
            logger.info("Check for pruning")
            self.trial.report(balanced_acc, epoch)

            # Handle pruning based on the intermediate value.
            if self.trial.should_prune():
                raise optuna.TrialPruned()


        return balanced_acc


    def hyper_optimizer(self, num_trials):
        
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
        study.optimize(self.objective, n_trials=num_trials)
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        # Delete history so it does not spam

        self.history = {
            'training_loss': [],
            'training_accuracy': [],
            'validation_loss': [],
            'ece': [],
            'auroc': [],
            'sensitivity': [],
            'specificity': [],
            'validation_accuracy': []
        }

        self.settings['history'] = self.history
 
        update_logs(self.settings, self.best, optim_data=best_trial.params)


        



