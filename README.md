# Pytorch Template Runner 🏃‍♂️🏃‍♀️

## Requirements
- Pytorch
- Albumentations
- Tensorboard
- OpenCV

## Directory Structure
```
pytorch.template/
│
├── main.py          -> main script to start training
│
├── config/ 
│   └── logging.conf
│
├── dataloaders/ 
│   ├── DATASET.py   ->
│   └── transform.py
│
├── models/          ->
├── options/         ->
├── runners/         ->
│   └── base.py
│
└── utils/ 
    ├── logger.py
    ├── general.py
    ├── losses.py
    ├── meter.py
    ├── optims.py
    ├── schedulers.py
    └── tensorboard.py
```

## Usage
Ex. Run main.py in background
```
python main.py options/OPTS.yaml
```

### Tensorboad
```
tensorbaord --bind_all --logdir LOG_DIR/
```
