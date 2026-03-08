from functions import training, get_net, choose_device
import torch
from torch import nn
import matplotlib as plt
import joblib


device = choose_device()
path = "fk_dataset_matrix.csv"
test_size = 0.1
net = get_net(device)
loss = nn.MSELoss()
epochs = 200
lr = 0.0001
batch_size = 256

train_epoch_list, train_loss_list, valid_epoch_list, valid_loss_list, X_scaler, Y_scaler= training(path, test_size, 
                                                                        epochs, net, loss, lr, batch_size, device)
plt.plot(train_epoch_list, train_loss_list)
plt.show()

save_path = ("1000000data.pth")
torch.save(net.state_dict(), save_path)
joblib.dump((X_scaler, Y_scaler), 'Y_scaler.pkl')