# Adaptive Reweighting Based on the Effective Area of Feature Space for Long-Tailed Classification (ID 8748)

## Prerequisite
* PyTorch >= 1.2.0
* Python3
* torchvision
* argparse
* numpy

## Dataset

* Imbalanced CIFAR. The original data will be downloaded and converted by imbalancec_cifar.py
* Imbalanced ImageNet
* The paper also reports results on iNaturalist 2018(https://github.com/visipedia/inat_comp). 


## CIFAR100
In the code, we calculate the accuracy.
```
CIFAR-100-LT,long-tailed imabalance ratio of 200
python AREA_cifar.py --gpu 3 --imb_type exp --imb_factor 0.005 --batch-size 64 --loss_type CE --dataset cifar100 --train_rule None 
```
```
CIFAR-100-LT,long-tailed imabalance ratio of 50
python AREA_cifar.py --gpu 2 --imb_type exp --imb_factor 0.1 --batch-size 64 --loss_type CE --dataset cifar100 --train_rule None 
```
```
CIFAR-10-LT,long-tailed imabalance ratio of 200
python AREA_cifar.py --gpu 1 --imb_type exp --imb_factor 0.005 --batch-size 64 --loss_type CE --dataset cifar10 --train_rule None 
```
More details will be uploaded soon.




```

python AREA_cifar_rebutt.py --gpu 3 --imb_type exp --imb_factor 0.1 --batch-size 64 --loss_type CE --dataset cifar100 --train_rule None 


```






    









