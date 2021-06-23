from torchvision.datasets import CIFAR10
from dataset import TargetedAttackCIFAR10
from torchvision.transforms import ToTensor,Scale,Compose
from attacker import AcivationAttacker
from torchvision.models import resnet18
from pytorch_pretrained_vit import ViT
from torch import nn
from whitebox_models import ResNet18FeatureExtractor
import torch
import cv2
import numpy as np

transform = Compose([
        Scale(224),
        ToTensor(),
])
cifar10 = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = transform)
ds = TargetedAttackCIFAR10(cifar10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wb_model = resnet18()
wb_model.fc = nn.Linear(512,10)
wb_model.to(device)

ckpt = torch.load('modelzoo/resnet18.pth',map_location = lambda storage, loc: storage)
wb_model.load_state_dict(ckpt['model'])

wb_model = ResNet18FeatureExtractor(wb_model)

bb_model = ViT('B_16_imagenet1k',pretrained = False)
bb_model.fc = nn.Linear(in_features = 768,out_features = 10)
bb_model = bb_model.to(device)
ckpt = torch.load('modelzoo/vit.pth',map_location = lambda storage, loc: storage)
bb_model.load_state_dict(ckpt['model'])

attacker = AcivationAttacker(eps = 0.07)

idx = 2
adv = attacker(wb_model,ds[idx][0][0].unsqueeze(0).to(device),ds[idx][1][0].unsqueeze(0).to(device))
print(ds[idx][0][1],ds[idx][1][1])
print(bb_model(Scale(384)(adv.detach().squeeze(0)).unsqueeze(0)))
cv2.imshow('1',np.transpose(adv[0].detach().squeeze(0).numpy(),(1,2,0)))
cv2.imshow('2',np.transpose(ds[idx][0][0].numpy(),(1,2,0)))
cv2.waitKey(0)