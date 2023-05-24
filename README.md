# SSA-SC

## Semantic Segmentation-assisted Scene Completion for LiDAR Point Clouds

This repository is for SSA-SC introduced in the following paper. [[arxiv paper]](https://arxiv.org/pdf/2109.11453.pdf)

If you find our work useful, please cite

```
@inproceedings{yang2021semantic,
  title={Semantic Segmentation-assisted Scene Completion for LiDAR Point Clouds},
  author={Yang, Xuemeng and Zou, Hao and Kong, Xin and Huang, Tianxin and Liu, Yong and Li, Wanlong and Wen, Feng and Zhang, Hongbo},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3555--3562},
  year={2021},
  organization={IEEE}
}
```

## Results on SemanticKITTI Benchmark

Quantitative results on **SemanticKITTI: Semantic Scene Completion** at the submission time.

![](figure/completion_result_small.png)


## Preperation

### Prerequisites
Tested with
* PyTorch 1.4.0
* CUDA 10.2
* Python 3.6.5
* Numpy 1.17.42
* torch-scatter 2.0.4
* torch-geometric 1.3.0
* spconv 1.2.1

### Dataset

Please download the Semantic Scene Completion dataset (v1.1) from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html) and extract it.

Or you can use [voxelizer](https://github.com/jbehley/voxelizer) to generate ground truths of semantic scene completion.

The dataset folder should be organized as follows.
```angular2
SemanticKITTI
├── dataset
│   ├── sequences
│   │  ├── 00
│   │  │  ├── labels
│   │  │  ├── velodyne
│   │  │  ├── voxels
│   │  │  ├── [OTHER FILES OR FOLDERS]
│   │  ├── 01
│   │  ├── ... ...
```

## Getting Start
Clone the repository:
```
git clone https://github.com/jokester-zzz/SSA-SC.git
```

We provide training routine examples in the `SSC_configs/examples/` folder. Make sure to change the dataset path to your extracted dataset location in such files if you want to use them for training. Additionally, you can change the folder where the performance and states will be stored, this folder has been defined as `SSC_out` by default.
* `config_dict['DATASET']['ROOT_DIR']` should be changed to the root directory of the SemanticKITTI dataset (containing `dataset` folder)
* `config_dict['OUTPUT']['OUT_ROOT'] ` should be changed to desired output folder.

### Train SSA-SC Net

```
$ cd <root dir of this repo>
$ python networks/train_SSA_SC.py --cfg SSC_configs/examples/SSA_SC.yaml --dset_root <path/dataset/root>
```
### Validation

Validation passes are done during training routine. Additional pass in the validation set with saved model can be done by using the `networks/validate.py` file. You need to provide the path to the saved model and the dataset root directory.

```
$ cd <root dir of this repo>
$ python networks/validate.py --weights </path/to/model.pth> --dset_root <path/dataset/root>
```
### Test

Since SemantiKITTI contains a hidden test set, we provide test routine to save predicted output in same format of SemantiKITTI, which can be compressed and uploaded to the [SemanticKITTI Semantic Scene Completion Benchmark](http://www.semantic-kitti.org/tasks.html#semseg).

We recommend to pass compressed data through official checking script provided in the [SemanticKITTI Development Kit](http://www.semantic-kitti.org/resources.html#devkit) to avoid any issue.

You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training. For testing, you can use the following command.

```
$ cd <root dir of this repo>
$ python networks/test.py --weights </path/to/model.pth> --dset_root <path/dataset/root> --out_path <predictions/output/path>
```
### Pretrained Model

You can download the models with the scores below from this [Google drive link](https://drive.google.com/file/d/1pzvtuk3A9V_M-8a0rTAh5_E-ZjqTpPWN/view?usp=sharing), 

| Model  | Segmentation | Completion |
|--|--|--|
| SSA-SC | 23.51 | 58.79 | 

<sup>*</sup> Results reported to SemanticKITTI: Semantic Scene Completion (OLD) leaderboard ([link](https://competitions.codalab.org/competitions/22037#results)).

## Acknowledgement
This project is not possible without multiple great opensourced codebases.
* [spconv](https://github.com/traveller59/spconv)
* [LMSCNet](https://github.com/cv-rits/LMSCNet)
