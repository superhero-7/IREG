# IREG
Official implement for paper: ["Whether you can locate or not? Interactive Referring Expression Generation"](https://arxiv.org/abs/2308.09977)

## 🔥 News
- 2024.3.20: Release the codebase.
- 2023.7.26: Our paper is accepted by ACM MM 2023 Main Track. 

## Step1 Feature extract

First download two sets of data:

1. [COCO2014](https://cocodataset.org/#download): [Train images](http://images.cocodataset.org/zips/train2014.zip) and [Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

2. [Refcoco](https://github.com/lichengunc/refer): Refcoco,Refcoco+,Refcocog


The feature extraction code is under misc/feature_extraction and is divided into two types:
1. Extract the features of the 36 proposal bboxes that touch the base: the code is in refcocog_proposal.py
2. Extract features of a given box: The code is in refcocog_target.py

Feature extraction requires the installation of detectron2. Just refer to the installation link in VLT5 github. It can be solved with one line of commands;

## Pretrained Checkpoints

Download from：
1. VLT5 Epoch30.pth link: https://drive.google.com/drive/folders/12Acv2YLQSxgrx_-4mahUvqNikcz7XfPi
2. OFA Refcoco, Refcoco+, Refcocog base ckpt, details can be seen in:https://github.com/OFA-Sys/OFA/blob/main/checkpoints.md#finetuning-ofa-base

### Enviroment setup
python version 3.7.4
```python
pip install -r requirements.txt
```

### Start
```python
cd Dialog
bash scripts/REG_VLT5.sh 2 refcoco unc 0 1 25552
```

### Project structure
#### 1.1 ckpt
Store all checkpoints during the training process;
#### 1.2 misc
Including feature extraction, bad re collection, testing, visualization and draft code, etc.;
#### 1.3 OFA
OFA's base warehouse has modified the refcoco_eval part;
#### 1.4 scripts
The training startup script needs to change the data and pre-training model weight paths accordingly;
#### 1.5 src
The main code is here
##### 1.5.1 eval_utils
Ref test codebase;
##### 1.5.2 modeling
Main model file, huggingface style;
##### 1.5.3 tools
Various functional functions, parameter files, distributed tool functions, training base framework, etc. are all here;

reg_data.py, reg_model.py, reg.py: Mainly responsible for base and RL training. The tests here only include the most basic one-shot test;

multitask_reg_data.py, multitask_reg_model.py, multitask_reg.py are mainly responsible for:
* Dialog Training, Dialog Training only needs to modify the Dataset. In fact, a DialogDataset is added to reg_data;
* The process of Dialog Test is a little more complicated. It needs to be determined whether it has passed the OFA test, but in fact there is only one function written in multitask_reg_model;
* In the main process of multitask_reg.py, it is necessary to change the logic that one more base model is needed to serve the test. In multitask_reg, self.model is the refiner and basemodel is the basemodel.


## ✒ Citation
Please cite our paper if you find it helpful :)
```
@misc{ye2023locate,
      title={Whether you can locate or not? Interactive Referring Expression Generation}, 
      author={Fulong Ye and Yuxing Long and Fangxiang Feng and Xiaojie Wang},
      year={2023},
      eprint={2308.09977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

