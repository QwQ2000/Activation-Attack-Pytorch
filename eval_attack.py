from torchvision.datasets import CIFAR10
from dataset import TargetedAttackCIFAR10
from torchvision.transforms import ToTensor,Scale,Compose
from attacker import AcivationAttacker
from torchvision.models import resnet18
from torch import nn
from whitebox_models import ResNet18FeatureExtractor
import torch

transform = Compose([
        Scale(224),
        ToTensor(),
])
cifar10 = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = transform)
ds = TargetedAttackCIFAR10(cifar10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bb_model = resnet18()
bb_model.fc = nn.Linear(512,10)
bb_model.to(device)

ckpt = torch.load('modelzoo/resnet18.pth')
bb_model.load_state_dict(ckpt['model'])

wb_model = ResNet18FeatureExtractor(bb_model)

attacker = AcivationAttacker()

attacker(wb_model,ds[0][0][0].unsqueeze(0).to(device),ds[0][1][0].unsqueeze(0).to(device))