# Semi-Supervised Learning via Weight-aware Distillation under Class Distribution Mismatch

Official PyTorch implementation of
["**Semi-Supervised Learning via Weight-aware Distillation under Class Distribution Mismatch**"]


## 1. Requirements
### Environments
(1) The main packages that require in our work are listed as follows. 

- CUDA 10.1+
- python == 3.7.13
- pytorch == 1.12.0
- torchvision == 0.13.0
- scikit-learn == 1.0.2
- tensorboardx == 2.2
- matplotlib  == 3.5.3
- numpy == 1.21.5
- scipy == 1.7.3
- tqdm == 4.64.1

(2) We also provide a requirements.text that list all the packages in our environment. You can construct the environment by running the following command.
```
conda install --yes --file requirements.txt 
```



### Datasets 
For CIFAR10 and CIFAR100, we provide a function to automatically download and preprocess the data, you can also download the datasets from the link.
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)


## 2. Training
The training process of WAD could be split into two steps. The first is to learn a teacher by leveraging contrastive learning(SimCLR). The second is to distill the knowledge from the teacher and train the target model(student). 

### Train a teacher

In the paper, the teacher in trained by contrastive learning(SimCLR). To train a teacher in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python contrast_main.py --mismatch <MISMATCH> --dataset <DATASET> --model <NETWORK> --mode teacher --shift_trans_type none --batch_size 32 --epochs <EPOCH> --logdir './result_model/teacher/<DIR>'
```

* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* In our experiment, we set --epochs 700 in cfar10 and --epoch 2000 in cifar100 .
* And we set mismatch = 0.2, 0.4, 0.6, 0.8.
* NETWORK indicates the network for training teacher models. It is set as resnet18 in our work.
* We provide the folders of "mis20", "mis40", "mis60", and "mis80" for the learned teacher model, i.e., --logdir.


### Train the SSL model
To train target model in the paper, run this command:

```

CUDA_VISIBLE_DEVICES=0 python distillation_main.py --cuda --seed <SEED> --alpha <ALPHA> --network wideresnet_28_2 --mismatch <MISMATCH> --split <SPLIT> --epochs <EPOCH> --mode eval --dataset <DATASET> --model <NETWORK>  --shift_trans_type none --load_teacher_path './result_model/teacher/<DIR>/last.model' --load_path './result_model/student/<DIR_0>' 
```

* **Option** 
* For CIFAR10, set --datatset cifar10, else set --datatset cifar100.
* The alpha in our experiments is set as 0.1. You can varing it in a wide range but not too large to prevent introducing more unknown categories.
* The value of mismatch is between 0 and 1. In our experiment, we set mismatch = 0.2, 0.4, 0.6, 0.8.
* --split represents the times that added the reliable instances for training.It is set as 5 in our work.
* In our experiment, we set --epochs 1500 in cfar10 and --epoch 2000 in cifar100.
* NETWORK indicates the network for training teacher models. It should keep consistent with the former and is set as resnet18 in our work.
* The model here is the same as that when training the teacher, as well as the --load_teacher_path(--logdir).
* --load_path indicates the path to save the target model. Here, we provide some folders under the one "result_model" to save it.

Then, we can get a SSL model trained by WAD in --load_path, and the accuracy also be reported in the logger. 

## 3. Reference
```
@inproceedings{tack2020csi,
  title={CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances},
  author={Jihoon Tack and Sangwoo Mo and Jongheon Jeong and Jinwoo Shin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```




