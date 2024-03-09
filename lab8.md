# Lab 8: Contine to Object Detection with YOLO

--

# Overview

In this lab, we will continue to work on object detection with YOLO. We will load the trained YOLO model  and use it to detect objects in images and videos. We will try to evaluate the performance of the YOLO model with e validation dataset.

## Objectives

- Load the trained YOLO model
- Use the YOLO model to detect objects in images and videos
- Evaluate the performance of the YOLO model with a validation dataset

## Prerequisites

- Lab 7
- the trained YOLO model checkpoint file

# Prepare the environment

## Define the data dictionary

```python
import os
data_dict = {'path': os.path.abspath('./datasets/VOC'),
             'train': ['images/train2012', 'images/train2007', 'images/val2012', 'images/val2007'],
             'val': ['images/test2007'], 'test': ['images/test2007'],
             'names': {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                       8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                       15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}}
```

## Download the dataset

```python
import torch
import tarfile
from pathlib import Path

# Download
dir = Path(data_dict['path'])  # dataset root dir
dir.mkdir(parents=True, exist_ok=True)  # create dir
(dir / 'images').mkdir(parents=True, exist_ok=True)  # create dir

urls = ['http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # 446MB, 5012 images
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # 438MB, 4953 images
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar']  # 1.95GB, 17126 images
# download(urls, dir=dir / 'images', delete=False, curl=True, threads=3)
for url in urls:
    torch.hub.download_url_to_file(url, dir / 'images' / url.split('/')[-1], progress=False)

# Extract
for file in dir.glob('images/*.tar'):
    dir.mkdir(exist_ok=True, parents=True)  # create dir
    print(f'Extracting {file}...')
    tar = tarfile.open(file)
    tar.extractall(dir / 'images')
    tar.close()
    file.unlink()  # delete tar after extraction
```

## Convert the VOC Dataset to YOLO Format

Next, we will convert the VOC dataset to the YOLO format.

```python
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    names = list(data_dict['names'].values())
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# Convert
path = dir / 'images/VOCdevkit'
for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
    imgs_path = dir / 'images' / f'{image_set}{year}'
    lbs_path = dir / 'labels' / f'{image_set}{year}'
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    with open(path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
        image_ids = f.read().strip().split()
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
        lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        f.rename(imgs_path / f.name)  # move image
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format
```


## Define hyperparameters and training settings

Next, we will define the hyperparameters and training settings for training the YOLO model. We will use the hyperparameters and training settings from the YOLOv5 model. We will define the hyperparameters and training settings as a dictionary and pass it to the `train` method of the YOLO model.

```python
from yolov5.utils.general import check_dataset

data = check_dataset(data_dict)

train_path, val_path = data["train"], data["val"]
nc, names = len(data["names"]), data["names"]

hyp = {'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
       'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
       'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
       'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0,
       'mixup': 0.0, 'copy_paste': 0.0}

cfg = {'nc': nc, 'depth_multiple': 0.33, 'width_multiple': 0.25,
       'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
       'backbone': [[-1, 1, 'Conv', [64, 6, 2, 2]], [-1, 1, 'Conv', [128, 3, 2]], [-1, 3, 'C3', [128]],
                    [-1, 1, 'Conv', [256, 3, 2]], [-1, 6, 'C3', [256]], [-1, 1, 'Conv', [512, 3, 2]],
                    [-1, 9, 'C3', [512]], [-1, 1, 'Conv', [1024, 3, 2]], [-1, 3, 'C3', [1024]],
                    [-1, 1, 'SPPF', [1024, 5]]],
       'head': [[-1, 1, 'Conv', [512, 1, 1]], [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]], [-1, 3, 'C3', [512, False]], [-1, 1, 'Conv', [256, 1, 1]],
                [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], [[-1, 4], 1, 'Concat', [1]],
                [-1, 3, 'C3', [256, False]], [-1, 1, 'Conv', [256, 3, 2]], [[-1, 14], 1, 'Concat', [1]],
                [-1, 3, 'C3', [512, False]], [-1, 1, 'Conv', [512, 3, 2]], [[-1, 10], 1, 'Concat', [1]],
                [-1, 3, 'C3', [1024, False]], [[17, 20, 23], 1, 'Detect', ['nc', 'anchors']]]}

device = "cuda"
num_epochs = 1
batch_size = 4
img_size = 640
```

## Build the YOLO Model

Next, we will define the YOLO model. We will use the `yolov5` package to define the YOLO model. We will use the `yolov5` package to define the YOLO model with the hyperparameters and training settings.

```python
from yolov5.utils.torch_utils import model_info
from yolov5.models.yolo import Model

model = Model(cfg).to(device)
model.half().float()
model.hyp = hyp # attach hyperparameters to model
model.nc = nc  # attach number of classes to model
model.names = names  # attach class names to model
model.verbose = True
model_info(model)
model
```

## Construct the DataLoader

Next, we will construct the DataLoader for the VOC dataset. We will use the `DataLoader` class from the `torch.utils.data` module to construct the DataLoader for the VOC dataset. We will use the DataLoader to load the images and labels from the VOC dataset and pass them to the YOLO model for training.

```python
import numpy as np
from yolov5.utils.dataloaders import create_dataloader

train_loader, dataset = create_dataloader(
    train_path,
    img_size,
    batch_size,
    max(int(model.stride.max()), 32),
    hyp=hyp,
    augment=True,
    shuffle=True,
)
labels = np.concatenate(dataset.labels, 0)
```

## Define the optimizer and loss function

YOLO use the `smart_optimizer` and `ComputeLoss` to define the optimizer and loss function. The `smart_optimizer` function is used to define the optimizer for the YOLO model. The `ComputeLoss` function is used to define the loss function for the YOLO model.

```python
# Define the optimizer and loss function

from yolov5.utils.torch_utils import smart_optimizer
from yolov5.utils.loss import ComputeLoss

optimizer = smart_optimizer(model)
compute_loss = ComputeLoss(model)  # Define loss function
```

# Evaluate the performance of the YOLO model

Next, we will evaluate the performance of the YOLO model with a validation dataset. 

## Load the trained YOLO model checkpoint file

we will first load the trained YOLO model checkpoint file saved in the previous lab. Previously, we have saved the trained YOLO model checkpoint file as `yolov5n_voc.pth` with `torch.save` function. We will load the trained YOLO model checkpoint file with `torch.load` function and then use the `load_state_dict` function to load the trained model weights.

```python
import torch

# Load the trained model weights
model.load_state_dict(torch.load('yolov5n_voc.pth'))
```

## Continue training the YOLO model

Next, we will continue training the YOLO model with the validation dataset. We will use the DataLoader to load the images and labels from the validation dataset and pass them to the YOLO model for training.

```python
from tqdm import tqdm

model.train()
model.to(device)

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, position=0, leave=True)
    running_loss = 0.0
    for (imgs, targets, paths, _) in pbar:
        # Forward pass
        imgs = imgs.to(device).float() / 255
        pred = model(imgs)
        loss, loss_items = compute_loss(pred, targets.to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(f"Epoch [{epoch}/{num_epochs}]")
        pbar.set_postfix(loss=loss.item(), running_loss=running_loss)
    model.load_state_dict(torch.load('yolov5n_voc.pth'))
```

## Construct the DataLoader for the validation dataset

Next, we will construct the DataLoader for the validation dataset. We will use the `DataLoader` class from the `torch.utils.data` module to construct the DataLoader for the validation dataset. We will use the DataLoader to load the images and labels from the validation dataset and pass them to the YOLO model for evaluation.

```python
val_loader, dataset = create_dataloader(
    val_path,
    img_size,
    batch_size,
    max(int(model.stride.max()), 32),
    hyp=hyp,
    augment=True,
    shuffle=True,
)
```

## Make predictions with the YOLO model

Next, we will make predictions with the YOLO model.

```python
hubmodel = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)

model.eval()
# swap in our trained model
hubmodel.model.model = model

pbar = tqdm(val_loader, position=0, leave=True)
for (imgs, targets, paths, _) in pbar:
    pred = hubmodel(paths)
    pred.print()
    pred.show()
    break
```