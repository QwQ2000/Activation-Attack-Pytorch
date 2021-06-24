from torchvision.datasets import CIFAR10
from dataset import TargetedAttackCIFAR10
from torchvision.transforms import ToTensor,Resize,Compose,ToPILImage
from attacker import AcivationAttacker
from torchvision.models import resnet18,densenet121
from pytorch_pretrained_vit import ViT
from torch import nn
from torch.utils.data import DataLoader
from whitebox_models import ResNet18FeatureExtractor
import torch
import numpy as np
from tqdm import tqdm
import cv2

transform = Compose([
        Resize(224),
        ToTensor(),
])
cifar10 = CIFAR10(root = './CIFAR-10',download = True,train = False,transform = transform)
ds = TargetedAttackCIFAR10(cifar10)
#_,ds = torch.utils.data.random_split(ds, [int(0.99 * len(ds)),len(ds) - int(0.99 * len(ds))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wb_eval_model = resnet18()
wb_eval_model.fc = nn.Linear(512,10)
wb_eval_model.to(device)

if device == torch.device('cpu'):
    ckpt = torch.load('modelzoo/resnet18.pth',map_location = lambda storage, loc: storage)
else:
    ckpt = torch.load('modelzoo/resnet18.pth')
wb_eval_model.load_state_dict(ckpt['model'])

wb_model = ResNet18FeatureExtractor(wb_eval_model)

#bb_model = ViT('B_16_imagenet1k',pretrained = False)
#bb_model.fc = nn.Linear(in_features = 768,out_features = 10)
#bb_model = bb_model.to(device)
#ckpt = torch.load('modelzoo/vit.pth',map_location = lambda storage, loc: storage)
bb_model = densenet121()
bb_model.classifier = nn.Linear(1024,10)
bb_model = bb_model.to(device)
if device == torch.device('cpu'):
    ckpt = torch.load('modelzoo/densenet121.pth',map_location = lambda storage, loc: storage)
else:
    ckpt = torch.load('modelzoo/densenet121.pth')
bb_model.load_state_dict(ckpt['model'])

def eval():
    attacker = AcivationAttacker(eps = 0.07,k = 10)
    loader = DataLoader(ds,batch_size = 16,shuffle = True,pin_memory = True,num_workers = 4)
    
    errors,utrs,tsucs,ttrs = [],[],[],[]
    wb_errors,wb_tsucs = [],[]
    
    for idx,((src_x,src_label),(tgt_x,tgt_label)) in tqdm(enumerate(loader)):
        src_x,tgt_x,src_label,tgt_label = src_x.to(device),tgt_x.to(device),src_label.to(device),tgt_label.to(device)
        adv = attacker(wb_model,src_x,tgt_x)
        with torch.no_grad():
            tr_res = torch.argmax(bb_model(Resize(224)(adv)),dim = 1)
            wb_res = torch.argmax(wb_eval_model(adv),dim = 1)

            f = lambda x:torch.sum(x).cpu().detach().numpy() / len(x)
            
            wb_error = f(wb_res != src_label)
            wb_tsuc = f(wb_res == tgt_label)
            wb_errors.append(wb_error)
            wb_tsucs.append(wb_tsuc)

            errors.append(f(tr_res != src_label))
            utrs.append(f((tr_res != src_label) & (wb_res != src_label)) / wb_error)
            tsucs.append(f(tr_res == tgt_label))
            ttrs.append(f((tr_res == tgt_label) & (wb_res == tgt_label)) / wb_tsuc)

    mean = lambda x:np.mean(np.array(x))
    error,utr,tsuc,ttr = mean(errors),mean(utrs),mean(tsucs),mean(ttrs)
    wb_error,wb_tsuc = mean(wb_errors),mean(wb_tsucs)

    return error,utr,tsuc,ttr,wb_error,wb_tsuc


def visualize(idx):
    (src_x,src_label),(tgt_x,tgt_label) = ds[idx]
    src_x,tgt_x = src_x.unsqueeze(0).to(device),tgt_x.unsqueeze(0).to(device)
    attacker = AcivationAttacker(eps = 0.07,k = 20)
    adv = attacker(wb_model,src_x,tgt_x)
    with torch.no_grad():
        tr_res = nn.functional.softmax(bb_model(Resize(224)(adv)),dim = 1).cpu().detach().numpy()[0]
        wb_res = nn.functional.softmax(wb_eval_model(adv),dim = 1).cpu().detach().numpy()[0]
    print(torch.argmax(torch.Tensor(wb_res)),torch.argmax(torch.Tensor(tr_res)))
    if torch.argmax(torch.Tensor(tr_res)) == tgt_label and torch.argmax(torch.Tensor(wb_res)) == tgt_label:
        print('Success targetted attack on both models.')
    trans = ToPILImage()
    src_img,adv_img = np.array(trans(src_x[0].cpu().detach())),np.array(trans(adv[0].cpu().detach()))
    cv2.imshow('Source Image',src_img)
    cv2.imshow('Adversarial Sample',adv_img)
    print('Blackbox model output probs:{}'.format(tr_res))
    print('Whitebox model output probs:{}'.format(wb_res))
    print('Source Label:{}    Target label:{}'.format(src_label,tgt_label))
    cv2.waitKey(0)

if __name__ == '__main__':
    print(eval())