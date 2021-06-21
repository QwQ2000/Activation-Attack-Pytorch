import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam,SGD
from torch.nn import CrossEntropyLoss
from vit_pytorch import ViT
from torchvision.transforms import ToTensor
import numpy as np 
from sklearn.metrics import accuracy_score
from torchsummary import summary
from tqdm import tqdm

trainset = CIFAR10(root = './CIFAR-10',download = True,train = True,transform = ToTensor())
testset = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = ToTensor())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch = 1000
milestones = set([i for i in range(0,n_epoch,20)])

model = ViT(image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 16,
            depth = 4,
            heads = 2,
            mlp_dim = 32
).to(device)

def train():
    loader = DataLoader(trainset,batch_size = 128,shuffle = True)
    optimizer = SGD(model.parameters(),lr = 1e-3,momentum = 0.5)
    criterion = CrossEntropyLoss().to(device)

    for epoch in tqdm(range(n_epoch)):
        losses = []
        for idx,(x,y) in enumerate(loader):
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in milestones:
            print('Epoch {}:train_loss:{} test_acc:{}'.format(epoch,np.mean(losses),test()))
            model.train()
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }
            torch.save(state,'ckpts/{}.pth'.format(epoch))

def test():
    model.eval()
    loader = DataLoader(testset,batch_size = 1,shuffle = True)
    y,pred = [],[]
    for idx,(x0,y0) in enumerate(loader):
        x0,y0 = torch.Tensor(x0).to(device),y0.numpy()
        y.append(y0[0])
        pred0 = model(x0).cpu().detach().numpy()
        pred.append(np.where(pred0[0] == np.max(pred0[0]))[0][0])
    y,pred = np.array(y),np.array(pred)
    return accuracy_score(y,pred)

if __name__ == '__main__':
    if device != torch.device('cpu'):
        summary(model,input_size = (3,32,32),batch_size = -1)
    train()