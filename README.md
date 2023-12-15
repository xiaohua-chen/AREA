# AREA: Adaptive Reweighting via Effective Area for Long-Tailed Classification (ICCV 2023)

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



## Citation

If you find this code useful for your research, please cite our paper.
```
@inproceedings{chen2023area,
  title={AREA: Adaptive Reweighting via Effective Area for Long-Tailed Classification},
  author={Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Yang, Chule and Li, Bo and Hu, Qinghua and Wang, Weiping},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19277--19287},
  year={2023}
}
```







    









