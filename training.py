import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import shutil
import torch.nn.functional as F

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
    if not os.path.exists(path):
        os.mkdir(path)



def update_logs(settings, best=False):
    # TODO: Vielleicht noch original Log markieren

    with open("logs/training_logs.json") as json_file:
        log_json = json.load(json_file)

        log_time = settings["time"]

        # If current time does not exist yet in the log
        if log_time not in log_json:
            log_json[log_time] = {}

        log_json[log_time]["title"] = settings["title"]
        log_json[log_time]["setup"] = settings["setup"]
        log_json[log_time]["model"] = settings["model"]
        log_json[log_time]["dataset"] = settings["dataset"]
        log_json[log_time]["batchsize"] = settings["batchsize"]
        log_json[log_time]["current_epoch"] = settings["current_epoch"]
        log_json[log_time]["current_checkpoint"] = settings["current_checkpoint"]
        log_json[log_time]["best_checkpoint"] = settings["best_checkpoint"]
        log_json[log_time]["ECE_score"] = settings["ECE_score"]
        log_json[log_time]["history"] = settings["history"]


        if "best_config" not in log_json[log_time]:
            log_json[log_time]["best_config"] = {}

        if best:
            log_json[log_time]["best_config"]["this_epoch"] = settings["current_epoch"][-1]
            log_json[log_time]["best_config"]["this_accuracy"] = settings["history"]["accuracy"][-1]
            log_json[log_time]["best_config"]["this_ECE_score"] = settings["ECE_score"][-1]
        

    # save changes
    with open("logs/training_logs.json", "w") as json_file:
        json_file.seek(0) # set pointer to file beginning
        json.dump(log_json, json_file, indent=4)
        json_file.truncate()



def backup_logs(settings):

    # make sure backup folder exists
    ensure_directory("./logs/backups")

    # make a backup
    log_time = settings["time"]
    log_time = log_time.replace(":","-")
    shutil.copyfile("logs/training_logs.json", f"logs/backups/{log_time}.json")









class TrainUncertainty:

    def __init__(self, settings, device, model, trainset, testset, batch_size = 64):
        """
        Args:
            device (_type_): _description_
            network (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 64.
        """

        self.device = device
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size
        self.settings = settings
        self.this_epoch = 0
        self.best_accuracy = 0.0


        # For now hardcoded optimiziers
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001,betas=(0.0, 0.99))
        self.criterion = nn.CrossEntropyLoss()

        # Some bookkeeping:
        self.iter = 0
        self.history = {
            'training_loss': [],
            'validation_loss': [],
            'accuracy': []
        }


    def load(self, checkpoint):
        """
        Load stored checkpoint to resume training.

        Parameters:
            - checkpoint (str): Path to stored training state.

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


    def save(self, best=False):

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
            'history': self.history

        }

        if best:
            torch.save(state, self.model_path)
            torch.save(state, self.best_path)
        else:
            torch.save(state, self.model_path)



    def train(self, num_epochs):

        self.model.to(self.device)

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Set progress bar format.
        bar_format = '{l_bar}{bar:20}{r_bar}{bar:-10b}'
        
        # For each epoch
        for epoch in range(num_epochs):
            running_training_loss = 0.0
            running_validation_loss = 0.0


            ############
            # Training #
            ############

            self.model.train()
            for iteration, (inputs, labels) in tqdm(enumerate(trainloader), bar_format=bar_format, total=len(trainloader)):

                # Inputs and labels to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients TODO: What does that mean?
                self.optimizer.zero_grad()

                # Feed network the images
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs,labels)

                # Backprop with that loss
                loss.backward()

                # Optimiter step TODO: What does that mean?
                self.optimizer.step()

                # Store loss
                running_loss = loss.item()

                # print statistics
                if iteration % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {iteration + 1:5d}] training_loss: {running_loss / 2000:.3f}')
                

            # Safe Training loss
            running_training_loss /= len(trainloader.dataset)

            self.history['training_loss'].append(running_training_loss)
            print(f'[{epoch + 1} finished, {iteration + 1:5d}] training_loss: {running_training_loss}')



            ##############
            # Validation #
            ##############


            self.model.eval()
            num_correct = 0
            num_examples = 0
            for iteration, (inputs, labels) in tqdm(enumerate(testloader), bar_format=bar_format, total=len(testloader)):

                # Inputs and labels to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Feed network the images
                outputs = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs,labels)

                # Update loss
                running_validation_loss += loss.data.item() * inputs.size(0)

                # Count correct ones
                correct = torch.eq(torch.max(F.softmax(outputs, dim = 1), dim = 1)[1], labels).view(-1)  # TODO What does this mean?
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]


            # Safe Testing loss and accuracy
            running_validation_loss /= len(testloader.dataset)
            accuracy = num_correct / num_examples
           
            self.history['validation_loss'].append(running_training_loss)
            self.history['accuracy'].append(accuracy)
            print(f'[{epoch + 1} finished, {iteration + 1:5d}] validation_loss: {running_validation_loss}, accuracy: {accuracy}')

                    

            # Save training state
            self.this_epoch = epoch + 1
            self.model_path = f'checkpoints/best_{str(self.this_epoch)}.pt'
            self.best_path = f'checkpoints/best_{str(self.this_epoch)}.pt'

            if accuracy > self.best_accuracy:
                self.best = True
                self.best_accuracy = accuracy
                self.save(best=True)
            else:
                self.best = False
                self.save()


            

            # Update Logging
            self.settings["current_epoch"] = epoch
            self.settings["current_checkpoint"] = self.model_path
            self.settings["best_checkpoint"] = self.best_path
            self.settings["ECE_score"] = "TODO"
            self.settings["history"] = self.history
            update_logs(self.settings, self.best)


        backup_logs(self.settings)
        return self.history



