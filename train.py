from model.unet_model import UNet
import matplotlib.pyplot as plt

# Some basic or classic models used to test the performance, temporarily not provided on the GitHub
'''
from model.Unet3Plus.UNet_3Plus import UNet_3Plus
from model.ResUnet.resunet_d6_causal_mtskcolor_ddist import *
from model.ResUnet.resunet_d7_causal_mtskcolor_ddist import *
from mxnet import nd
from model.ResUnet2.res_model import *
from model.ResUnet2.res_model2 import *
from model.Alexnet.Alexnet import *
from model.Netcollection import Netcollection
from model.Segnet import Segnet
from model.FCN3.FCN import *
from model.UnetPlusPlus.UnetPlusPlus import UnetPlusPlus
from model.ResUnetPlus.ResUnetPlus import 
'''
from model.ResUnetPlus.ResUnetPlusAtt import ResUnetPlusAtt

from dataset import Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import numpy as np



def train_net(net, device, data_path, epochs=50, batch_size=4, lr=8e-6):
    # load dataset
    dataset = Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

 
    # define RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.99)

    # define Loss
    weights = torch.tensor([1.0, 2.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_loss = float('inf')
    torch.cuda.empty_cache()

    epoches_collection = []
    loss_collection = []
    for epoch in range(epochs):
        epoches_collection.append(epoch)
        
        net.train()
        with tqdm(total=len(dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for image, label in train_loader:
                temp = 0
                optimizer.zero_grad()

                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
              
                label = label.squeeze() 

                pred = net(image)
                loss = criterion(pred, label)

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                
                temp += loss.item()
                optimizer.step()
                loss.backward()
                optimizer.step()

            loss_collection.append(temp)
            print('Loss/train', loss.item())
    print(best_loss)

    epoches_collection = np.array(epoches_collection)
    loss_collection = np.array(loss_collection)
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.plot(epoches_collection, loss_collection, "ob-", label='Original Loss')

    sim_loss = np.polyfit(epoches_collection, loss_collection, 2)
    print("Polynomial coefficients:", sim_loss)

    p = np.poly1d(sim_loss)
    fitted_losses = p(epoches_collection)
    plt.plot(epoches_collection, fitted_losses, "r-", label='Fitted Curve')

    plt.legend()
    plt.show()
    plt.savefig("xxx") # fig save dir




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Net1
    #net = UNet(n_channels=1, n_classes=2)

    #Net2
    #net = UNet_3Plus(in_channels=1, n_classes=2, feature_scale=2)

    #Net3
    #net = ResNet(block = BasicBlock, blocks_num = [3, 4, 6, 3], num_classes=2, include_top = True)  # Make sure to use the correct class name

    #Net4
    net = ResUnetPlusAtt()

    net.to(device=device)

    # training dir
    data_path = "xxx"
    train_net(net, device, data_path)