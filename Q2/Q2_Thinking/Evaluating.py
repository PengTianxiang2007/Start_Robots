import torch
from torch import nn
from functions import choose_device, get_net, evaluate, return_matrix
import joblib

x_scaler, Y_scaler= joblib.load('Y_scaler.pkl')
device = choose_device()
net = get_net(device)
output = evaluate(net, Y_scaler, device)
matrix = return_matrix(output)