from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from random import randint
from torchvision.transforms import ToTensor

class TargetedAttackCIFAR10(Dataset):
    def __init__(self,cifar10 = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = ToTensor())):
        super(TargetedAttackCIFAR10,self).__init__()
        self.cifar10 = cifar10
    
    def __getitem__(self, idx):
        t = randint(0,len(self.cifar10) - 1)
        while self.cifar10[t][1] == self.cifar10[idx][1]:
             t = randint(0,len(self.cifar10))
        return self.cifar10[idx],self.cifar10[t]

    def __len__(self):
        return len(self.cifar10)
