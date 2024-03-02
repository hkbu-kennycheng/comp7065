# Lab 7: Object Detection

---

# Overview

Continuing the theme from the previous lab, we will go through the models in a Full Self Driving (FSD) stack. We will start with the perception module, which is responsible for detecting objects in the environment. We will use the YOLO (You Only Look Once) model, which is a state-of-the-art, real-time object detection system. We will use the pre-trained YOLO model to detect objects in the images and visualize the results. We will try out the pre-trained model on a few images and videos and see how it performs. Then we will go through the process of training the YOLO model on a dataset from scratch. Finally, we will use the trained model to detect objects in the images captured from the car's camera.

## Background

Object detection is important for self-driving cars to understand the environment around them. It is used to detect other vehicles, pedestrians, traffic signs, and other objects in the environment. It has been a challenging problem in computer vision, and there have been many approaches to solve it. The YOLO model is a popular choice for object detection in self-driving cars because it is fast and accurate. It can detect objects in real-time, which is important for self-driving cars to make decisions quickly.

YOLO is a state-of-the-art, real-time object detection system. It is based on a single neural network that divides the input image into a grid and predicts bounding boxes and class probabilities for each grid cell. Where other object detection systems like R-CNN and its variants use a pipeline to perform the detection, YOLO uses a single neural network to predict the bounding boxes and class probabilities directly from the full image in one evaluation. This makes YOLO extremely fast, capable of processing images in real-time. The current version of YOLO is YOLOv8, different versions of YOLO may have different architectures and performance.

## Objectives

In this lab, you will:

- Use the pre-trained YOLO model to detect objects in images and videos
- Train the YOLO model on a dataset from scratch
- Use the trained model to detect objects in images captured from the car's camera

## Prerequisites

Before starting this lab, you should have:

- Basic knowledge of Python
- Basic knowledge of deep learning
- Basic knowledge of computer vision
- Installed the required libraries and packages

# Getting Started

In this section, we will use the YOLO model to detect objects in images and videos. We will use the pre-trained YOLO model to detect objects in the images and videos and visualize the results. There are various pre-trained YOLO models available, trained with different hyperparameters for different requirements. We will use the smallest version of the YOLOv5 model, which is YOLOv5n. The YOLOv5n model is trained on the COCO dataset, which contains 80 classes of objects. We will use the pre-trained YOLOv5n model to detect objects in the images and videos and visualize the results.

## The model architecture

The YOLOv5 model architecture is based on the YOLOv3-v4 model architecture with some improvements. The YOLOv5 model architecture consists of a backbone network, a neck network, and a head network. The backbone network is responsible for extracting features from the input image. The neck network is responsible for combining the features from the backbone network. The head network is responsible for predicting the bounding boxes and class probabilities for each grid cell. The YOLOv5 model architecture is designed to be fast and accurate, making it suitable for real-time object detection.

Here is the [architecture](https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png) of the YOLOv5 model:

![](https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png)

## Load the Pre-trained YOLO Model

First, we will load the pre-trained YOLO model. We will use the `torch.hub.load` function to load the pre-trained YOLOv5n model. The `torch.hub.load` function loads the model from the PyTorch Hub. The PyTorch Hub is a repository of pre-trained models and datasets. It provides a convenient way to load the pre-trained models and datasets for various tasks. We will use the `torch.hub.load` function to load the pre-trained YOLOv5n model from the PyTorch Hub.

```python
import torch

# Load the pre-trained YOLOv5n model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, verbose=False)
print(model)
```

The `torch.hub.load` function takes three arguments:

- The first argument is the name of the repository. In this case, it is `ultralytics/yolov5`.
- The second argument is the name of the model. In this case, it is `yolov5n`.
- The third argument is the flag to load the pre-trained model. In this case, it is `pretrained=True`.
- The fourth argument is the flag to suppress the output. In this case, it is `verbose=False`.

After loading the model, we will print the summary of the pre-trained YOLO model. We will use the `print` function to print the summary of the pre-trained YOLO model. The `print` function takes the model summary as an argument.

## Detect Objects in Images

Next, we will detect objects in the images using the pre-trained YOLO model. We will use the `detect` method of the pre-trained YOLO model to detect objects in the images. The `detect` method takes the input images as an argument and returns the detected objects in the images. We will use the `detect` method to detect objects in a few images and visualize the results.

```python
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Inference
results = model(imgs)
results.print()

results.pandas().xyxy[0]
```

## Visualize the Results

Finally, we will visualize the results of the object detection. We will use the `show` method of the pre-trained YOLO model to visualize the results of the object detection. The `show` method takes the input images and the detected objects as arguments and visualizes the results of the object detection. We will use the `show` method to visualize the results of the object detection in the images.

```python
# Visualize the results
results.show()
```

## Detect Objects in Videos

Next, we will detect objects in the videos using the pre-trained YOLO model. We will make use of `supervision` package to process the video and detect objects in the video. The `supervision` package provides a convenient way to process the video and detect objects in the video. We will use the `supervision` package to process the video and detect objects in the video.

```juypyter
!pip install supervision
```

```python
import supervision as sv
import numpy as np
import torch

video_path = 'car_chase_01.mp4'
torch.hub.download_url_to_file('https://github.com/dannylee1020/object-detection-video/raw/master/videos/car_chase_01.mp4', video_path, progress=False)

video_info = sv.VideoInfo.from_video_path(video_path)
```

After that,

```python
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame)
    results.save()
    results.show()
    return frame

sv.process_video(source_path=video_path, target_path=f"result.mp4", callback=process_frame)
```



# Case Study: Training the YOLO Model from Scratch with VOC Dataset

In this section, we will go through the process of training the YOLO model on a dataset from scratch. We will use the YOLOv5 model to train the model on a dataset of images and labels. We will use the pre-trained YOLOv5 model as the starting point and fine-tune the model on the dataset. We will use the COCO dataset as the starting point and fine-tune the model on the dataset. We will use the pre-trained YOLOv5 model to detect objects in the images and videos and visualize the results.

[Pascal Visual Object Classes (VOC)](http://host.robots.ox.ac.uk/pascal/VOC/) dataset is a standard dataset for object detection. It contains 20 classes of objects, including person, car, bus, and bicycle. We will use the VOC dataset to train the YOLO model. We will use the YOLOv5 model artitecture with `yolov5n`'s hyperparameters and training from scratch on the VOC dataset.

## Install the Required Libraries

First, we will install the required libraries for training the YOLO model. We will use the `pip` package manager to install the required libraries. We will install the `yolov5` package, which contains the YOLOv5 model and the required utilities for training the model. We will also install the `pycocotools` package, which contains the COCO dataset and the required utilities for training the model.

```juypyter
!pip install yolov5 pycocotools
```

## Download the VOC Dataset

Next, we will start with download the VOC dataset. We define the `data_dict` dictionary with the path to the VOC dataset and the names of the classes.

```python
import os
data_dict = {'path': os.path.abspath('./datasets/VOC'),
             'train': ['images/train2012', 'images/train2007', 'images/val2012', 'images/val2007'],
             'val': ['images/test2007'], 'test': ['images/test2007'],
             'names': {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
                       8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                       15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}}
```

And then we will download the VOC dataset.

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
    1,
    max(int(model.stride.max()), 32),
    hyp=hyp,
    augment=True,
    shuffle=True,
)
labels = np.concatenate(dataset.labels, 0)
mlc = int(labels[:, 0].max())  # max label class
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

## Train the YOLO Model

Finally, we will train the YOLO model on the VOC dataset. We will use the `train` method of the YOLO model to train the model on the VOC dataset. The `train` method takes the DataLoader and the training settings as arguments and trains the model on the dataset. We will use the `train` method to train the YOLO model on the VOC dataset.

```python
from tqdm import tqdm

model.train()
model.to(device)

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, position=0, leave=True)
    for (imgs, targets, paths, _) in pbar:
        # Forward pass
        imgs = imgs.to(device).float() / 255
        pred = model(imgs)
        loss, loss_items = compute_loss(pred, targets.to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Epoch [{epoch}/{num_epochs}]")
        pbar.set_postfix(loss=loss.item())
```

## Save the Trained Model

Finally, we will save the trained model to a file. We will use the `torch.save` function to save the trained model to a file. We will use the `torch.save` function to save the trained model to a file.

```python
torch.save(model.state_dict(), 'yolov5n_voc.pth')
```

## Train with CLI

Each YOLOv5 model comes with a command-line interface (CLI) that allows you to train, test, and run the model. The YOLOv5 CLI is a convenient way to interact with the YOLOv5 model and perform various tasks. You can use the YOLOv5 CLI to train the model on a dataset, test the model on a dataset, and run the model on images and videos. The YOLOv5 CLI provides a simple and intuitive way to interact with the YOLOv5 model. Here is the official [documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for training custom data.

But the drawback is that ome of the services involved in the YOLOv5 CLI are commerial, so you may need to pay for them.

# Exploretory Exercise: Detect Objects in Images Captured from the Car's Camera

In this exercise, please use the [Udacity Self Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car) to train the YOLO model and detect objects in the images captured from the car's camera. You will need to train from scratch and visualize the results.

## Udacity Self Driving Car Dataset

The Udacity Self Driving Car Dataset is a standard dataset for self-driving cars. It contains images and labels of objects in the environment, including other vehicles, pedestrians, traffic signs, and other objects. We will use the Udacity Self Driving Car Dataset to train the YOLO model and detect objects in the images captured from the car's camera. There are only 11 classes in the dataset, including car, truck, pedestrian, traffic light, and other objects.