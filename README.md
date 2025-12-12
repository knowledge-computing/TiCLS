# Official Code for TiCLS: Tightly Coupled Language Text Spotter

## News
- TiCLS has been accepted to WACV 2026.

## How to Run

### Train
```
python3.10 tools/train_net.py \
  --config-file ${CONFIG_FILE} \
  --num-gpus 4
```
### Evaluation
```
python3.10 tools/train_net.py --config-file ${CONFIG_FILE} --num-gpus 2 --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```
