import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm



# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        # store the inputs and outputs
        self.X = df.values[:, :-2]
        self.y = df.values[:, -2]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    


# model definition
MLP = nn.Sequential(OrderedDict([
          ('dense1', nn.Linear(34,128)),
          ('relu1', nn.ReLU()),
          ('batchnorm1', nn.BatchNorm1d(128)),
          ('dense2', nn.Linear(128,128)),
          ('relu2', nn.ReLU()),
          ('batchnorm2', nn.BatchNorm1d(128)),
          ('dense3', nn.Linear(128,128)),
          ('relu3', nn.ReLU()),
          ('batchnorm3', nn.BatchNorm1d(128)),
          ('dense4', nn.Linear(128,128)),
          ('relu4', nn.ReLU()),
          ('batchnorm4', nn.BatchNorm1d(128)),
          ('dense5', nn.Linear(128,5)),
          ('act1', nn.Softmax())  
        ]))

# prepare the dataset
def prepare_data(train_path,valid_path):  #,test_path
    # load the dataset
    train = CSVDataset(train_path)
    valid = CSVDataset(valid_path)
    #test = CSVDataset(test_path)
    
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=128, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=128, shuffle=True)
    #test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, valid_dl #,test_dl

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Number of epochs to train the model
num_epochs = 20
# train the model
def train_model(train_dl, model,valid_dl):
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # enumerate epochs
    loss_per_iter = []
    loss_per_epoch = []
    valid_loss_per_epoch = []
    min_valid_loss = np.inf
    for epoch in range(num_epochs):
        # enumerate mini batches
        training_loss = 0.0
        loop = tqdm(enumerate(train_dl),total=len(train_dl), leave=False)
        for i, (inputs, targets) in loop:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.float())
            # calculate loss
            loss = criterion(yhat, targets.long())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            
            # save the loss to plot
            training_loss += loss.item()
            loss_per_iter.append(loss.item())
            
            # update progressbar
            loop.set_description(f"Epoch[{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
        loss_per_epoch.append(training_loss/(i+1))
        
        valid_loss = 0.0
        model.eval()
        for j, (v_data,v_labels) in enumerate(valid_dl):
            v_data = v_data.to(device)
            v_labels = v_labels.to(device)
            val_yhat = model(v_data.float())
            v_loss = criterion(val_yhat, v_labels.long())
            valid_loss += v_loss.item()
        
        print(f"Epoch {epoch+1} \t Training Loss: {training_loss/len(train_dl)} \t Validation Loss: {valid_loss/len(valid_dl)}")
        valid_loss_per_epoch.append(valid_loss/(j+1))
        
        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased: ({min_valid_loss:.5f}---{valid_loss/len(valid_dl):.5f})\t Saving the model..")
            min_valid_loss = valid_loss/len(valid_dl)
            
            torch.save(model.state_dict(),'saved_weights.pth')
            
        
    # Plot training loss curve
    plt.figure(figsize=(10,6))
#     plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per Mini-batch")
    plt.plot(np.arange(len(loss_per_epoch)), loss_per_epoch, ".-", label="Training Loss per Epoch")
    plt.plot(np.arange(len(valid_loss_per_epoch)), valid_loss_per_epoch, ".-", label="Validation Loss per Epoch")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
           


# prepare the data
train_path = 'REBA_Train_dataset.csv'
valid_path = 'REBA_Val_dataset.csv'
train_dl, valid_dl = prepare_data(train_path,valid_path)
# define the network
model = MLP.to(device)
print("Model: \n",model)
# train the model
train_model(train_dl, model, valid_dl)