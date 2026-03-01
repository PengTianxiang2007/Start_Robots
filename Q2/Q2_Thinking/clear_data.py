import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
"""标准步骤
1.用pandas读取数据,切分特征和标签
2.用train_test_split切分数据
3.用StandardScaler进行标准化
4.转换为tensor张量
6.封装为TensorDataset,再封装为dataloader
"""

raw_data = pd.read_csv("fk_dataset_matrix.csv")
X = raw_data.iloc[:, :6]
Y = raw_data.iloc[:,6: ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)  #每一行代表一个样本，每一列代表一个特征(输入特征)
X_test = X_scaler.transform(X_test)
X_valid = X_scaler.transform(X_valid)
Y_train = Y_scaler.fit_transform(Y_train)
Y_valid = Y_scaler.transform(Y_valid)
Y_test = Y_scaler.transform(Y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

BATCH_SIZE=64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train(train_dataloader, valid_dataloader, net, optimizer, loss, epochs):
    epoch_list = []
    loss_list = []    
    for epoch in range(epochs):
        net.train()
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs=net(batch_X)
            n = loss(outputs, batch_Y)
            n.backward()
            optimizer.step()
            
        epoch_list.append(epoch)
        loss_list.append(n.item())
        if epoch%2000 == 0:
            net.eval()
            for valid_X, valid_Y in valid_dataloader:
                real_loss = loss(net(valid_X), valid_Y)
            print(f"epoch:{epoch} :        loss : {n}            real_loss : {real_loss}")
    return epoch_list, loss_list

net = nn.Sequential(nn.Flatten(), nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 12))
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss = nn.MSELoss()
epochs = 10000
epoch_list, loss_list = train(train_loader, valid_loader, net, optimizer, loss, epochs)
plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.show()