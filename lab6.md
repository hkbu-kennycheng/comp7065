# Lab 6: Facial Landmarks Detection in Full Self-Driving (FSD) System

---

# Overview

In this series of labs, we will be working on the problem Full Self-Driving (FSD) system. By this problem, we would go through how AI can be used to improve the safety of the Full Self-Driving (FSD) system. It covers facial landmarks detection and object detection with camera built into the car. The goal is to detect the driver's facial landmarks and make sure that the driver is paying attention to the road. In this lab, we continue to work on facial landmarks detection with pre-trained models. It is a continuation of the previous lab, where we have learned how to train a facial landmarks detection model from scratch with a simple CNN architecture and doing data augmentation in pytorch `Dataset`. In this lab, we will try to do data augmentation using `transforms` pipeline and fine-tune a pre-trained model to detect facial landmarks.

## Background

The current Full Self-Driving (FSD) system requires the driver to pay attention to the road and be ready to take control of the vehicle at any time. In order to improve the safety of the Full Self-Driving (FSD) system, Manufacturers has decided to add facial landmarks detection to the system. The goal is to detect the driver's facial landmarks and make sure that the driver is paying attention to the road.

![](https://o.aolcdn.com/images/dims3/GLOB/crop/1723x969+0+0/resize/800x450!/format/jpg/quality/85/https://s.aolcdn.com/os/ab/_cms/2021/05/28161425/Screen-Shot-2021-05-28-at-4.14.06-PM.png)

The above image shows the debug message for driver status monitoring system in the Full Self-Driving (FSD) system. The system uses a camera to capture the driver's facial landmarks and make sure that the driver is paying attention to the road.

## Facial Landmarks Detection

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQeBPFRQeFAD-q0Hf0YxfK65QzdsghPJNg7oE-nLjpb2g&s)

Facial landmarks detection is a computer vision task that involves detecting key points on a person's face, such as the eyes, nose, and mouth. It is a crucial task in many applications, such as face recognition, facial expression analysis, and driver status monitoring system in the Full Self-Driving (FSD) system. There are numerous methods to detect facial landmarks, such as regression-based methods, heatmap-based methods, and deep learning-based methods.

## Objectives

In this lab, we will try to do data augmentation using `transforms` pipeline and fine-tune a pre-trained model to detect facial landmarks. The lab will cover the following topics:

1. Data Augmentation using `transforms` pipeline
2. Fine-tuning a pre-trained model
3. Evaluating the model

## Requirements

Before starting this lab, you should have the following:

- Basic knowledge of Python
- Basic knowledge of PyTorch
- Basic knowledge of Convolutional Neural Networks (CNNs)
- Basic knowledge of Facial Landmarks Detection

## Dataset

The dataset we will be using in this lab is same  as the previous lab. It contains images of faces with facial landmarks annotations. The dataset is divided into two parts: a training set and a validation set. Both  set contains 2,500 images with a face bounding box and 5 facial landmarks annotations. The dataset is available in [this link](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip). You can download the dataset and extract it to the `data` directory in the same directory as this notebook. The dataset directory should have the following structure:

```bash
lfw_5590/
    Aaron_Eckhart_0001.jpg
    ...
net_7876/
    Aaron_Eckhart_0001.jpg
    ...
testImageList.txt
trainImageList.txt
```

In `trainImageList.txt` and `testImageList.txt`, each line contains the path to an image, face bounding box and the corresponding facial landmarks annotations. The annotations are in the format of `image_path, bbox_x1, bbox_x2, bbox_y1, bbox_y2, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5`, where `(x1, y1)` to `(x5, y5)` are the coordinates of the 5 facial landmarks.

## Outline

The remainder of this lab is organized as follows. We will start by loading and visualizing the dataset. Then, we will define a `Dataset` class and a `DataLoader` to load the dataset. Next, we will define a pre-trained model and fine-tune it to detect facial landmarks. Finally, we will evaluate the model and visualize the results.

# Let's get started!

Please create a new notebook for this lab and follow the instructions below. The code snippets are provided to help you get started, and it's optimized for running in 2GB GPU environment. Thus, please feel free to use the provided code snippets in our lab's machines.

## Download and extract the dataset

First, we need to download the dataset and extract it to the current directory. The dataset is available in [this link](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip). We can re-use the code from the previous lab to download and extract the dataset.

```python
import urllib.request
import zipfile
import os

def download_and_extract_zip(url):
    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Download the zip file
    urllib.request.urlretrieve(url, filename)

    # Extract the contents of the zip file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()

    # Remove the downloaded zip file
    os.remove(filename)

download_and_extract_zip("http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip")
```

After running the above code, the dataset should be extracted to the current directory.

## Construct the dataset

Next, we need to construct the dataset by loading the images and annotations from the dataset directory using `pandas` and `PIL`. Here is the code to construct the dataset:

```python
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class FaceLandmarksDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        # Specify the column names
        columns = [
            'image_path',
            'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2',
            'landmark1_x', 'landmark1_y',
            'landmark2_x', 'landmark2_y',
            'landmark3_x', 'landmark3_y',
            'landmark4_x', 'landmark4_y',
            'landmark5_x', 'landmark5_y'
        ]

        # Read the file into a DataFrame
        self.df = pd.read_csv(annotations_file, delimiter=' ', names=columns)
        self.df['image_path'] = self.df['image_path'].str.replace('\\', '/').apply(lambda x: os.path.join(img_dir, x))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.df.iloc[idx, 0])
        bbox = self.df.iloc[idx, 1:5]
        bbox = np.array([bbox[0], bbox[2], bbox[1], bbox[3]], dtype=int).reshape(-1, 2)
        landmarks = self.df.iloc[idx, 5:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'bbox':bbox, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

In the above code, we define a `FaceLandmarksDataset` class that inherits from `Dataset`. The `__init__` method reads the annotations file into a DataFrame and constructs the paths to the images. The `__len__` method returns the number of samples in the dataset. The `__getitem__` method loads the image and annotations for a given index and applies the transformation pipeline to the sample.

## Visualize the dataset

We can visualize the dataset by plotting the images and annotations. Here is the code to visualize the dataset:

```python
import matplotlib.pyplot as plt

face_dataset = FaceLandmarksDataset(annotations_file='trainImageList.txt', img_dir='./', transform=None)

fig = plt.figure(figsize=(18,6))

for i, sample in enumerate(face_dataset):

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    plt.imshow(sample['image'].permute(1, 2, 0)) # rearrange color channel from BGR to RGB
    plt.scatter(sample['landmarks'][:, 0], sample['landmarks'][:, 1], s=50, marker='.', c='r')
    plt.scatter(sample['bbox'][:, 0], sample['bbox'][:, 1], s=50, marker='.', c='g')

    if i == 3:
        plt.show()
        break
```

In the above code, we visualize the first 4 samples in the dataset. We plot the images, facial landmarks using red dots, and face bounding box using green dots.

## Data Augmentation with custom `transforms` function

In the previous lab, we have learned how to do data augmentation by defining a static method in `Dataset` class. In this lab, we will try to do data augmentation using `transforms` pipeline. The `transforms` pipeline is a powerful tool for data augmentation in PyTorch. It allows us to apply a series of transformations to the input data, such as resizing, cropping, flipping, rotating, and normalizing. Unfortunately, the `transforms` pipeline does not support the augmentation of bounding boxes and landmarks. We need to define a custom transformation to augment the bounding boxes and landmarks. Let's do it one by one.

### Crop the face region

The first transformation we will apply is to crop the face region from the image using the bounding box. We can define a custom transformation to crop the face region from the image. Here is the code to define the `BBoxCrop` transformation:

```python
from torchvision.transforms import functional as F

# transform for cropping the face
class BBoxCrop(object):

    def __call__(self, sample):
        image, bbox, landmarks = sample['image'], sample['bbox'], sample['landmarks']

        x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]
        h, w = image.shape[1], image.shape[2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        image = image[:, y1:y2, x1:x2]
        landmarks = landmarks - [x1, y1]

        return {'image': image, 'landmarks': landmarks}
```

In the above code, we define a `BBoxCrop` transformation that crops the face region from the image using the bounding box. The `__call__` method takes a sample as input and returns the cropped image and landmarks.

### Resize the face region

The second transformation we will apply is to resize the face region to a fixed size. We can define a custom transformation to resize the face region to a fixed size. Here is the code to define the `Rescale` transformation:

```python
from torchvision.transforms import functional as F

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[1], image.shape[2]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        image = F.resize(image.clone(), (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': image, 'landmarks': landmarks}
```

In the above code, we define a `Rescale` transformation that resizes the face region to a fixed size. The `__call__` method takes a sample as input and returns the resized image and landmarks.

### Normalize values and convert to tensor

The third transformation we will apply is to normalize the pixel values and convert the image and landmarks to tensors. Here is the code to define the `ToTensor` transformation:

```python
from torchvision.transforms import functional as F

class ToTensor(object):
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        return {'image': (image / 255).to(self.device), 'landmarks': torch.tensor(landmarks / 224, dtype=torch.float).to(self.device)}
```

In the above code, we define a `ToTensor` transformation that normalizes the pixel values and converts the image and landmarks to tensors. The `__call__` method takes a sample as input and returns the normalized image and landmarks as tensors.


## Define the `DataLoader` with custom `transforms` pipeline

After defining the custom transformations, we can define the `DataLoader` with the custom `transforms` pipeline. A `DataLoader` is a PyTorch class that provides an efficient way to load and iterate over the dataset. It takes a `Dataset` as input and provides various options for batching, shuffling, and parallel loading. A `transforms` pipeline is a series of transformations that are applied to the input data. It needs to be defined as a `Compose` object, which takes a list of transformations as input. Here is the code to define the `DataLoader` with the custom `transforms` pipeline:

```python
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

batch_size = 16
device = 'cuda'

train_dataset = FaceLandmarksDataset(annotations_file='trainImageList.txt', img_dir='./', transform=Compose([
    BBoxCrop(),
    Rescale((224, 224)),
    ToTensor(device=device)
]))
test_dataset = FaceLandmarksDataset(annotations_file='testImageList.txt', img_dir='./', transform=Compose([
    BBoxCrop(),
    Rescale((224, 224)),
    ToTensor(device=device)
]))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataloader), len(test_dataloader))
```

## Define the pre-trained model

Next, we need to define the pre-trained model and fine-tune it to detect facial landmarks. We can use a pre-trained model such as MobileNetV2 to replace the last layer with a new layer that outputs the 5 facial landmarks.

### MobileNetV2

MobileNetV2 is a lightweight deep learning model that is designed for mobile and embedded vision applications. It is based on the MobileNetV1 architecture and uses inverted residual blocks with linear bottlenecks. It has a small memory footprint and low computational cost, making it suitable for real-time applications on mobile and embedded devices. The model is pre-trained on the ImageNet dataset and has been widely used for various computer vision tasks. Here is the architecture of MobileNetV2:

![](https://pytorch.org/assets/images/mobilenet_v2_1.png)

![](https://pytorch.org/assets/images/mobilenet_v2_2.png)

Here is the code to define the pre-trained MobileNetV2 model and print the model architecture:

```python
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
model
```

In the above code, we define the pre-trained MobileNetV2 model and print the model architecture. The model has a series of convolutional layers followed by a fully connected layer that outputs the class scores. We can replace the last layer with a new layer that outputs the 5 facial landmarks.


### Replace the last layer

We can replace the last layer of the pre-trained model with a new layer that outputs the 5 facial landmarks. Here is the code to replace the last layer:

```python
import torch.nn as nn

model.classifier[1] = nn.Linear(1280, 10)
model
```

In the above code, we replace the last layer of the pre-trained MobileNetV2 model with a new layer that outputs the 5 facial landmarks. The new layer has 10 output units, corresponding to the 5 facial landmarks (x, y) coordinates.

## Fine-tune the pre-trained model

After replacing the last layer, we can fine-tune the pre-trained model to detect facial landmarks. We can define the loss function, optimizer, and training loop to fine-tune the model.

### Loss function and optimizer

Let's define the loss function and optimizer for fine-tuning the pre-trained model. We can use the Mean Squared Error (MSE) loss function to measure the difference between the predicted and ground truth facial landmarks. We can use the Adam optimizer to update the model parameters based on the loss function.

```python
from torch import nn
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training loop

Let's define the training loop to fine-tune the pre-trained model. The training loop consists of iterating over the training dataset, forward and backward passes through the model, and updating the model parameters based on the loss function and optimizer.

```python
from tqdm import tqdm

model.to(device)
model.train()

for epoch in range(3):
    running_loss = 0.0
    bar = tqdm(enumerate(train_dataloader))
    for i, data in bar:
        inputs, labels = data['image'], data['landmarks']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.reshape(batch_size, 5, 2), labels)
        loss.to(device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        bar.set_postfix(epoch=f'[{epoch + 1}, {i + 1}]', loss=running_loss)
```

In the above code, we put the model in training mode using `model.train()` and iterate over the training dataset to fine-tune the model. We use the Adam optimizer to update the model parameters based on the Mean Squared Error (MSE) loss function.

## Test the model

After fine-tuning the pre-trained model, we can test the model on the test dataset and evaluate its performance. We can define the evaluation loop to iterate over the test dataset, forward pass through the model, and compute the loss and accuracy.

```python
model.eval()

running_loss = 0.0

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, labels = data['image'], data['landmarks']
        outputs = model(inputs)
        loss = criterion(outputs.reshape(batch_size, 5, 2), labels)
        loss.to(device)
        running_loss += loss.item()

print(f'Loss: {running_loss / len(test_dataloader)}')
```

In the above code, we put the model in evaluation mode using `model.eval()` and iterate over the test dataset to compute the loss. We can also visualize the results by plotting the images and predicted facial landmarks.

## Evaluate the model

We can evaluate the model prediction with the ground truth facial landmarks. We can define a function to compute the Euclidean distance between the predicted and ground truth facial landmarks.

```python
def euclidean_dist(vector_x, vector_y):
    vector_x, vector_y = np.array(vector_x), np.array(vector_y)
    return np.sqrt(np.sum((vector_x - vector_y)**2, axis=-1))
```

## Visualize the results

Finally, we can visualize the results by plotting the images and predicted facial landmarks. We can define a function to plot the images and predicted facial landmarks.

```python
model.eval()

fig = plt.figure(figsize=(18,6))

for i, sample in enumerate(test_dataset):
    inputs, labels = sample['image'], sample['landmarks']
    outputs = model(inputs.unsqueeze(0))
    outputs = (outputs.reshape(5, 2).cpu().detach().numpy() * 224).astype(int)
    landmarks = (labels.cpu().detach().numpy() * 224).astype(int)
    image = (inputs.cpu().permute(1, 2, 0).numpy() * 255).astype(int)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{} distance {}'.format(i, np.sum(euclidean_dist(outputs, landmarks))))
    ax.axis('off')

    plt.imshow(image) # rearrange color channel from BGR to RGB
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, marker='.', c='g')
    plt.scatter(outputs[:, 0], outputs[:, 1], s=50, marker='.', c='r')

    if i == 3:
        plt.show()
        break
```

In the above code, we put the model in evaluation mode using `model.eval()` and iterate over the test dataset to plot the images and predicted facial landmarks. We can visualize the results by plotting the images, ground truth facial landmarks using green dots, and predicted facial landmarks using red dots.
