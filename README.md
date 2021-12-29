# Activation-Attack-Pytorch
Apply targeted attacks on black box ViT with the activation attack method. The white box model is ResNet-18.

Based on CVPR 2019 Paper **Feature Space Perturbations Yield More Transferable Adversarial Examples**.

My additional expreriment proves that the transferablitiy is low between heterogeneous deep models.

All the models are finetuned on **CIFAR-10** with the **Adam** optimizer with **lr = 1e-5**.

## Pretrained models
| Model Name | CIFAR-10 Test Accuracy | ImageNet-1K Pretrain |
| :--------: | :--------------------: | :-----------------------: |
| ResNet-18 | 0.9435 | Yes |
| DenseNet-121 | 0.9571 | Yes |
| ViT-B/16 | 0.9838 | Yes |

## Results
| Model | Error | uTr | tSuc | tTr |
| :---: | :-----: | :-----: | :-----: | :-----: |
| ResNet-18 | 96.39 | / | 75.48 | / |
| DenseNet-121 | 80.16 | 82.27 | 9.39 | 10.65 |
| ViT | 22.33 | 22.78  | 2.5 | 4.2 |
