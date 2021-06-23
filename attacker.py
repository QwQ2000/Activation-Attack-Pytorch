from torch import nn
from abc import ABCMeta,abstractmethod
import torch

class Attacker(metaclass = ABCMeta):
    @abstractmethod
    def generate(self,wb_model,src,tgt = None):
        return None

    def __call__(self,wb_model,src,tgt = None):
        return self.generate(wb_model, src, tgt)

class AcivationAttacker(Attacker):
    def __init__(self,eps = 0.07,k = 10):
        self.eps = eps
        self.k = k
    
    def generate(self, wb_model, src, tgt):
        #adv = torch.Tensor(src.cpu()).to(src.device)
        adv = src
        adv.requires_grad = True
        alpha = self.eps / self.k
        momentum = torch.zeros(src.shape).to(src.device)
        criterion = nn.MSELoss().to(src.device)

        wb_model.train()
        for _ in range(self.k):
            if adv.grad is not None:
                adv.grad.data.fill_(0)
            wb_model.zero_grad()
            loss = criterion(wb_model(adv),wb_model(tgt))
            loss.backward()
            momentum = momentum + adv.grad / torch.norm(adv.grad,p = 1)
            adv = torch.clip(adv - alpha * torch.sign(momentum),min = 0,max = 1)

        return adv
        