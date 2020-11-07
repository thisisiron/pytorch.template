# GAN Template

## Requirements
- Pytorch
- Albumentations
- Tensorboard
- OpenCV

## Directory Structure
```
GAN-Template.pytorch/
│
├── main.py - main script to start training
├── run.py
│
├── utils.py 
│
├── losses.py 
│
├── configuration/
│   ├── const.py
│   └── logging.conf
│
├── datasets/ 
    ├── dataset.py
    └── transform.py
```

## Usage
Ex. Run main.py in background
```
nohup python -u main.py &
```

### Tensorboad
```
tensorbaord --bind_all --logdir LOG_DIR
```
