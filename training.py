import os
import torch


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




class TrainUncertainty:

    def __init__(self, device, model, trainset, testset, batch_size = 64):
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


        # For now hardcoded optimiziers
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=0.0001,betas=(0.0, 0.99))


        # Some bookkeeping:
        self.epoch = 0
        self.iter = 0
        self.history = {
            'training_loss' : []
        }


    def save(self):

        state = {

            # Save network parameters.
            'network': self.network.state_dict(),
            'discriminator': self.discriminator.state_dict(),

            # Save optimizer state.
            'network_optimizer': self.optimizer.state_dict(),

            # Save epochs and iterations.
            'epoch': self.epoch,
            'iter': self.iter,
            
            # Save previous training history.
            'history': self.history

        }

        torch.save(state, f'{self.save_path}/{str(self.iter).zfill(10)}.pt')



    def train(self):
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True, num_workers=2)


        print(self.model)



