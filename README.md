# Official Code for TiCLS: Tightly Coupled Language Text Spotter

## News
- TiCLS has been accepted to WACV 2026! 

## Installation

**Python 3.8 + PyTorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6)**

```
git clone https://github.com/knowledge-computing/TiCLS.git
cd ticls

conda create -n ticls python=3.8 -y
conda activate ticls

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

pip uninstall -y urllib3
pip install urllib3==1.26.6
pip install transformers==4.46.3

python setup.py build develop
```


## How to Run TiCLS

### Train
```
python tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 4
```
### Evaluate
```
python tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 2 --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```

## How to Run PLM 
### Train PLM
```
python plm_train/pretrain_LM_for_scenetext.py
```

### Get PLM-decoder only for TiCLS
```
python plm_train/get_decoder_from_PLM.py
```
## Model Weights

### PLM Weights
| Component            | Download URL |
|----------------------|-------------|
| PLM (Encoder&Decoder)    | [Download PLM Weights](https://drive.google.com/file/d/1hRItsRE2gT3FwBaQNzqrZDksb3ORJZaU/view?usp=drive_link) |
| PLM Decoder for TiCLS   | [Download PLM Decoder Weights](https://drive.google.com/file/d/1asBTIUrhkq0QoknvxaE4vNmZ0Q-2X-Or/view?usp=drive_link) |

## Dataset
### TiCLS Dataset

For downloading the dataset required to train TiCLS, please refer to the [DeepSolo repository](https://github.com/ViTAE-Transformer/DeepSolo/blob/main/DeepSolo/README.md). We provide the corresponding annotation files (.json) below for training TiCLS.

| Component            | Download URL |
|----------------------|-------------|
| Annotations    | [Download Annotations](https://drive.google.com/file/d/1rq3sWh2NxcQOh6wCjUDTl3OoL5OiLwQi/view?usp=sharing) |

### PLM Dataset
| Component            | Download URL |
|----------------------|-------------|
| PLM Tokenizer        | [Download Tokenizer](https://drive.google.com/drive/folders/160uSNy0_UpBR6-NVn_1PKPZTdwaDb0xy?usp=drive_link) |
| PLM Train Dataset    | [Download Train](https://drive.google.com/file/d/1I_xR6omIMgvzn4YBB6EQmXQwdikkOKVm/view?usp=drive_link) |
| PLM Test Dataset     | [Download Test](https://drive.google.com/file/d/10uxsdOpAsua7uHIbWHVmsrB8Xlx23X4Y/view?usp=drive_link) |

## To-Do List 
- [ ] Release pretrained and fine-tuned TiCLS model weights
- [x] Release pretrained PLM model weights and scripts for training 
- [x] Release detailed dataset information for TiCLS and PLM
