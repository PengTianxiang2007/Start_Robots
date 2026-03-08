import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def split_and_preprocessing(path, test_size):
    raw_data = pd.read_csv(path)
    X = raw_data.iloc[:, :5]
    Y = raw_data.iloc[:,5: ]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=test_size, random_state=42)

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_valid = X_scaler.transform(X_valid)
    X_test = X_scaler.transform(X_test)
    Y_train = Y_scaler.fit_transform(Y_train)
    Y_valid = Y_scaler.transform(Y_valid)
    Y_test = Y_scaler.transform(Y_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32) 
    Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

    return X_train_tensor, X_valid_tensor, X_test_tensor, Y_train_tensor, Y_valid_tensor, Y_test_tensor, X_scaler, Y_scaler

def get_dataloader(X, Y, batchsize, shuffle=True):
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset = dataset, batch_size=batchsize, shuffle=shuffle)
    return dataloader

def train_part(train_dataloader, valid_dataloader, epochs, net, loss, optimizer, device):
    loss_list = []
    epoch_list = []
    valid_epoch_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        net.train()
        for X, Y in train_dataloader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), Y)
            l.backward()
            optimizer.step()
            epoch_loss += l.item()
        avg_epoch_loss  = epoch_loss / len(train_dataloader)
        loss_list.append(avg_epoch_loss)
        epoch_list.append(epoch+1)
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                net.eval()
                valid_epoch_loss = 0
                for X, Y in valid_dataloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    valid_loss = loss(net(X), Y)
                    valid_epoch_loss += valid_loss.item()
            avg_valid_loss = valid_epoch_loss / len(valid_dataloader)
            valid_epoch_list.append(epoch+1)
            valid_loss_list.append(avg_valid_loss)
            print(f"epoch:{epoch+1}   loss:{avg_epoch_loss}   valid_loss:{avg_valid_loss}")
    return epoch_list, loss_list, valid_epoch_list, valid_loss_list

def training(path, test_size, epochs, net, loss, lr, batch_size, device):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, X_scaler, Y_scaler = split_and_preprocessing(path, test_size)
    train_loader = get_dataloader(X_train, Y_train, batch_size, shuffle=True)
    valid_loader = get_dataloader(X_valid, Y_valid, batch_size, shuffle=False)
    train_epoch_list, train_loss_list, valid_epoch_list, valid_loss_list = train_part(
        train_loader, valid_loader, epochs, net, loss, optimizer, device
    )
    return train_epoch_list, train_loss_list, valid_epoch_list, valid_loss_list, X_scaler, Y_scaler

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"use  : {device}")
    return device

def get_net(device):
    net = nn.Sequential(
    nn.Linear(5, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.05),

    nn.Linear(512, 512), 
    nn.BatchNorm1d(512),
    nn.ReLU(),

    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    
    nn.Linear(256, 64), 
    nn.BatchNorm1d(64),
    nn.ReLU(),
    
    nn.Linear(64, 12)
)
    net = net.to(device)
    return net

def evaluate(net,path,X_scaler, Y_scaler, device):
    net.eval()
    input_num = input("Please input the angles")
    input_num = input_num.split()
    input_array = np.array([list(map(float, input_num))], dtype=np.float32)
    input_array = X_scaler.transform(input_array)
    input_tensor = torch.tensor(input_array)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = net(input_tensor)
    output = Y_scaler.inverse_transform(output.cpu().numpy())
    return output

def return_matrix(output):
    row1 = output[0,:4].tolist()
    row2 = output[0,4:8].tolist()
    row3 = output[0,8:].tolist()
    row4 = [0.0, 0.0, 0.0, 1.0]
    output_matrix = [row1, row2, row3, row4]
    return output_matrix
