import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import AdamW,SGD,Adam
from torch.nn import *
from pytorch_pretrained_vit import ViT
from torchvision.transforms import ToTensor,Resize,Compose
from torchvision.models import resnet18,squeezenet1_1,densenet121
import numpy as np 
from sklearn.metrics import accuracy_score
from torchsummary import summary
from tqdm import tqdm

transform = Compose([
        #Scale(384),
        Resize(224),
        ToTensor(),
])

trainset = CIFAR10(root = './CIFAR-10',download = True,train = True,transform = transform)
testset = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = transform)
_,valset = torch.utils.data.random_split(testset, [int(0.95 * len(testset)),len(testset) - int(0.95 * len(testset))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epoch = 21
milestones = set([i for i in range(0,n_epoch,1)])

torch.backends.cudnn.benchmark = True

#model = resnet18(pretrained = True)
#model.fc = Linear(in_features = 512,out_features = 10)

#model = ViT('B_16_imagenet1k',pretrained = True)
#model.fc = Linear(in_features = 768,out_features = 10)

#model = squeezenet1_1(pretrained = True)
#model.classifier = Sequential(
#    Dropout(p = 0.5, inplace = False),
#    Conv2d(512, 10, kernel_size = (1, 1), stride = (1, 1)),
#    ReLU(inplace = True),
#    AdaptiveAvgPool2d(output_size = (1, 1))
#)
model = densenet121(pretrained = True)
model.classifier = Linear(1024,10)
model = model.to(device)

def train():
    loader = DataLoader(trainset,batch_size = 16,shuffle = True,num_workers = 4,pin_memory = True)
    optimizer = Adam(model.parameters(),lr = 1e-5)
    criterion = CrossEntropyLoss().to(device)

    for epoch in tqdm(range(n_epoch)):
        losses = []
        for idx,(x,y) in tqdm(enumerate(loader)):
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in milestones:
            print('Epoch {}:train_loss:{} val_acc:{}'.format(epoch,np.mean(losses),validate()))
            model.train()
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }
            torch.save(state,'ckpts/{}.pth'.format(epoch))

def train_from_checkpoint(ckpt_file):
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model'])
    epoch = ckpt['epoch']
    optimizer = SGD(model.parameters(),lr = 1e-5)
    loader = DataLoader(trainset,batch_size = 64,shuffle = True,num_workers = 4,pin_memory = True)
    criterion = CrossEntropyLoss().to(device)

    for epoch in tqdm(range(epoch,n_epoch)):
        losses = []
        for idx,(x,y) in tqdm(enumerate(loader)):
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in milestones:
            print('Epoch {}:train_loss:{} val_acc:{}'.format(epoch,np.mean(losses),validate()))
            model.train()
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch
            }
            torch.save(state,'ckpts/{}.pth'.format(epoch))

def validate():
    model.eval()
    loader = DataLoader(valset,batch_size = 1,shuffle = True,num_workers = 4)
    y,pred = [],[]
    for idx,(x0,y0) in enumerate(loader):
        x0,y0 = torch.Tensor(x0).to(device),y0.numpy()
        y.append(y0[0])
        pred0 = model(x0).cpu().detach().numpy()
        pred.append(np.where(pred0[0] == np.max(pred0[0]))[0][0])
    y,pred = np.array(y),np.array(pred)
    return accuracy_score(y,pred)

def evaluate(ckpt_file):
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model'])
    epoch = ckpt['epoch']

    model.eval()
    loader = DataLoader(testset,batch_size = 1,shuffle = True,num_workers = 4)
    y,pred = [],[]
    for idx,(x0,y0) in tqdm(enumerate(loader)):
        x0,y0 = torch.Tensor(x0).to(device),y0.numpy()
        y.append(y0[0])
        pred0 = model(x0).cpu().detach().numpy()
        pred.append(np.where(pred0[0] == np.max(pred0[0]))[0][0])
    y,pred = np.array(y),np.array(pred)
    return accuracy_score(y,pred)

if __name__ == '__main__':
    if device != torch.device('cpu'):
        summary(model,input_size = (3,224,224),batch_size = -1)
    #train()
    #train_from_checkpoint('ckpts/5.pth')
    print('test_acc:{}'.format(evaluate('ckpts/0.pth')))