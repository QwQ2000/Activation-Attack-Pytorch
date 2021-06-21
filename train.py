import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from vit_pytorch import ViT
from torchvision.transforms import ToTensor
import numpy as np 


trainset = CIFAR10(root = './CIFAR-10',download = True,train = True,transform = ToTensor())
testset = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = ToTensor())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch = 100

model = ViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 10,
            depth = 3,
            heads = 4,
            mlp_dim = 32
).to(device)

def train():
    loader = DataLoader(trainset,batch_size = 32,shuffle = True)
    optimizer = Adam(model.parameters(),lr = 1e-3)
    criterion = CrossEntropyLoss().to(device)

    for epoch in range(n_epoch):
        losses = []
        for idx,(x,y) in enumerate(loader):
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {}:train_loss:{}".format(epoch,np.mean(losses)))

if __name__ == '__main__':
    train()