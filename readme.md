# Caltech-101 Classification with Fine-tuned CNN

## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- tensorboard
- numpy
- pandas

## Dataset Preparation
1. Download Caltech-101 dataset from [official link](https://data.caltech.edu/records/mzrjq-6wc02)
2. Unzip the downloaded file
```unzip Caltech101.zip```
3. Extract dataset and maintain original directory structure
4. Run dataset split script:
```python preprocess_data.py ```

## Training
```bash
python train.py
```

## Parameters search
```bash
python param_search.py
```

## Tensorboard visualization
```bash
tensorboard --logdir hw1/best_param/log
```

## Model weights download
[百度网盘](https://pan.baidu.com/s/1X36pC0MBamGYEdwDzDy53w?pwd=239f)