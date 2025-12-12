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


## How to Run

### Train
```
python tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 4
```
### Evaluation
```
python tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 2 --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```
## Citation

```bibtex
@inproceedings{jang2026ticls,
  title     = {TiCLS: Tightly Coupled Language Text Spotter},
  author    = {Jang, Leeje and Lin, Yijun and Chiang, Yao-Yi and Weinman, Jerod},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

## To-Do List 

- [ ] Release TiCLS pretrained/finetuned weights  
- [ ] Release a PLM model weight  
- [ ] Release dataset information
