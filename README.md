# SFDA-DE

This is a third-party implementation for CVPR 2022 paper [Source-Free Domain Adaptation via Distribution Estimation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Source-Free_Domain_Adaptation_via_Distribution_Estimation_CVPR_2022_paper.pdf).

## Requirements
- Python 3.7
- scipy
- PyYAML
- easydict
- torch==1.9.1
- torchvision==0.10.1

Please use a conda env and run `pip install -r requirements.txt`


## Dataset
The structure of the dataset should be like

```
visda2017
|_ category.txt
|_ train
|  |_ aeroplane
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bicycle
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ validation
|  |_ aeroplane
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bicycle
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
```
The "category.txt" contains the names of all the categories, which is like
```
aeroplane
bicycle
bus
car
...
```
"category.txt" can be found in ./experiments/dataset/VisDA-2017/category.txt

Remember to change the `DATAROOT` item in the 3rd line of the cfg file
```
./experiments/config/VisDA-2017/visda17_train2val_cfg.yaml
```

## Training
On VisDA-2017 dataset, first download the pretrained weight, password: sfda

https://pan.baidu.com/s/1RbLqvBtqJWNtBJN6Eh3Xzg

Then run:
```
python train.py --cfg ./experiments/config/VisDA-2017/visda17_train2val_cfg.yaml --exp_name SFDA-DE
```

The experiment log and checkpoints will be stored at ./experiments/ckpt/*{exp_name}**


## Citing 
Please cite the paper if you find this code helpful:
```
@inproceedings{ding2022source,
  title={Source-Free Domain Adaptation via Distribution Estimation},
  author={Ding, Ning and Xu, Yixing and Tang, Yehui and Xu, Chao and Wang, Yunhe and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7212--7222},
  year={2022}
}
```

## Thanks to third party
The code is based on  <https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation>.

