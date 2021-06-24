# Activation-Attack-Pytorch
Apply targeted attacks on black box ViT with the activation attack method. The white box model is ResNet-18.

Based on CVPR 2019 Paper **Feature Space Perturbations Yield More Transferable Adversarial Examples**.

This expreriment proves that the transferablitiy is low between heterogeneous deep models.

## ResNet-18 Result
| ResNet-18 Error | ResNet-18 tSuc |
| :-----: | :----: |
| 92.88 | 59.56 | 

## ViT Result 

| Error | uTr | tSuc | tTr |
| :-----: | :-----: | :-----: | :-----: |
| 22.33 | 22.78  | 2.5 | 4.2 |
