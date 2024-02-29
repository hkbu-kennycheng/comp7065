# Mini-projects in a group of 2-3 students

This mini-project is a part of the course requirement which contributes 60% to the final grade. The mini-project is
designed to test the student's understanding of the course material and the ability to apply the knowledge to solve
real-world problems. There are four suggested topics for the mini-project, and each group can choose one of the topics. There is a suggested dataset for each topic, and also tasks for each student in the group.

## Project requirements

The project should be done in a group of 2-3 students. Each student in the group should work on a different task in
the under the same project topic. The project should
be done using Jupyter Notebook. The project should be submitted as a Jupyter Notebook and python file with the following sections:

1. **Introduction**: A brief introduction to the project and the problem you are trying to solve.
2. **Data collection**: A description of the data you collected and how you collected it.
3. **Data preprocessing**: A description of the data preprocessing steps you performed.
4. **Model development**: A description of the model you developed to solve the problem.
5. **Model evaluation**: A description of the evaluation metrics you used to evaluate the model.
6. **Results**: A description of the results you obtained and how they compare to the state-of-the-art.
7. **Application**: A demonstration of how the model can be used in real-world applications.
8. **Conclusion**: A brief conclusion summarizing your work in the project.
9. **References**: A list of references you used in your work.
10. **Appendix**: Any additional information you want to include in your report.

Each student requires to submit an individual report of their work in the project. The report should include the following sections:

1. **Introduction**: A brief introduction to the task you worked on in the project.
2. **Task description**: A description of the task you worked on and the approach you used to solve it.
3. **Methodology**: A description of the methodology you used to complete the task.
3. **Results**: A description of the results you obtained and how they compare to the state-of-the-art.
4. **Conclusion**: A brief conclusion summarizing your work in the project.
5. **References**: A list of references you used in your work.
6. **Appendix**: Any additional information you want to include in your report.

# Project Topics

There are three suggested topics for the mini-project, and each group can choose one of the topics. Project topics are closely related to the course material, and they are designed to test the student's understanding of the course material and the ability to apply the knowledge to solve real-world problems. You are welcome to propose your own topics, but they must be approved by the instructor.

## Topic 1: Facial analysis

Facial analysis is the process of automatically detecting human faces in digital images and analyzing facial features. This is typically used in security systems, marketing, and entertainment. In this project, you will develop a model to detect human faces in images and analyze facial features.

### Suggested Dataset

Here is a dataset you can use for the project. You are welcome to propose and use your another dataset, but it must be approved by the instructor.

#### Large-scale CelebFaces Attributes (CelebA) Dataset

You can use the Large-scale CelebFaces Attributes (CelebA) Dataset, which contains over 200,000 celebrity images with annotations. The dataset can be downloaded from the following link: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

### Project details

The theme of the project is to develop a model to detect human faces in images and analyze facial features. The project should include the following steps:

1. Identify the task you will work on in the project.
   - You are expected to identify the task you will work on in the project. For example with the CelebA dataset, you can work on face filter, facial attribute recognition, and a face ID detector. You are welcome to identify your own task and provide a brief description of the task.
     - Face filter: You can develop a model to apply face filters to camera images. For example, you can develop a model to apply sunglasses, hats, and other accessories to camera images.
     - Facial attribute recognition: You can develop a model to recognize facial attributes such as wearing glasses, smiling, and facial hair.
     - Face ID detector: You can develop a model to detect human faces and assign a unique ID to each face. 
2. Data collection, preprocessing, and visualization.
   - You are expected to collect the data you will use in the project, preprocess the data, and visualize the data. Typically, you will need to build a `Dataset` class to load the data, and use `DataLoader` to load the data in batches. You may also need to preprocess the data, for example, by normalizing the data, and resizing the images using `transforms` pipe. You are also expected to visualize the data to understand the data distribution and the data quality.
3. Model development.
   - You are expected to develop a model to solve the task you identified in step 1. You are expected to train a model or use a pre-trained model and fine-tune it on the CelebA dataset. You are also expected to use transfer learning to improve the model performance.
4. Model evaluation.
   - You are expected to evaluate the model you developed in step 3. You are expected to use appropriate evaluation metrics to evaluate the model performance.
5. Results
    - You are expected to present the results you obtained in the project. You may compare the results to the state-of-the-art. You are also expected to visualize the results to understand the model performance.
6. Application
    - You are expected to demonstrate how the model can be used in real-world applications. For example, you can demonstrate how to apply face filters to camera images with your model. You are welcome to propose your own application and provide a brief description of the application.

## Topic 2: Person analysis

Person analysis is the process of automatically detecting human bodies in digital images and analyzing human body features. This is typically used in security systems, marketing, and entertainment. In this project, you will develop a model to detect human bodies in images and analyze human body features.

### Suggested Datasets and pre-trained model

Here are datasets you can use for the project. You are welcome to propose and use another dataset, but it must be approved by the instructor.

#### COCO keypoint detection dataset

You can use the COCO keypoint detection dataset, which contains over 200,000 images with human body keypoints annotations. The dataset can be downloaded from the following link: [COCO keypoint detection dataset](https://cocodataset.org/#keypoints-2020).

#### UCF101 - Action Recognition Data Set

UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This dataset is widely used for action recognition. The dataset can be downloaded from the following link: [UCF101 - Action Recognition Data Set](https://www.crcv.ucf.edu/data/UCF101.php).

#### Torchreid

Torchreid is a library for deep-learning person re-identification. The library provides a collection of pre-trained models for person re-identification. The library can be downloaded from the following link: [Torchreid](https://kaiyangzhou.github.io/deep-person-reid/). It contains [pre-trained models](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) for person re-identification including ResNet-50, MobileNetv2, and other models.

### Project details

The theme of the project is to develop a model to detect human bodies in images and analyze human body features. The project should include the following steps:

1. Identify the task you will work on in the project.
   - You are expected to identify the task you will work on in the project. For example, you can work on human body keypoint detection, human body attribute recognition, and human body action recognition. You are welcome to identify your own task and provide a brief description of the task.
     - Human body keypoint detection: You can develop a model to detect human body keypoints such as nose, eyes, shoulders etc.
     - Human action recognition: You can develop a model to recognize human actions in videos.
     - Person re-identification: You can develop a model to recognize a person across different cameras. You will need to implement a function to assign a unique ID to each person and track the person across different cameras.
2. Data collection, preprocessing, and visualization.
    - You are expected to collect the data you will use in the project, preprocess the data, and visualize the data. Typically, you will need to build a `Dataset` class to load the data, and use `DataLoader` to load the data in batches. You may also need to preprocess the data, for example, by normalizing the data, and resizing the images using `transforms` pipe. You are also expected to visualize the data to understand the data distribution and the data quality.
3. Model development.
   - You are expected to develop a model to solve the task you identified in step 1. You are expected to train a model or use a pre-trained model and fine-tune it on the dataset. You are also expected to use transfer learning to improve the model performance.
4. Model evaluation.
    - You are expected to evaluate the model you developed in step 3. You are expected to use appropriate evaluation metrics to evaluate the model performance.
5. Results
    - You are expected to present the results you obtained in the project. You may compare the results to the state-of-the-art. You are also expected to visualize the results to understand the model performance.
6. Application
    - You are expected to demonstrate how the model can be used in real-world applications. For example, you can demonstrate how to detect human body keypoints in camera/video with your model. You are welcome to propose your own application and provide a brief description of the application.

## Topic 3: Time series analysis

Time series analysis is the process of analyzing time series data to extract meaningful statistics and other characteristics of the data. This is typically used in finance, economics, and environmental science. In this project, you will develop a model to analyze traffic data of strategic/major roads in Hong Kong.

### Suggested Dataset

Here is a dataset you can use for the project. You are welcome use any dataset on [gov.hk](https://data.gov.hk/en-data/dataset). If you would like to use another dataset, it must be approved by the instructor.

#### Traffic Data of Strategic / Major Roads in Hong Kong

You can use the Traffic Data of Strategic / Major Roads in Hong Kong, which contains traffic sensors data in major roads in Hong Kong. The dataset can be downloaded from the following link: [Traffic Data of Strategic / Major Roads in Hong Kong](https://data.gov.hk/en-data/dataset/hk-td-sm_4-traffic-data-strategic-major-roads).

### Project details

The theme of the project is to develop a model to analyze traffic data of strategic/major roads in Hong Kong. The project should include the following steps:

1. Identify the task you will work on in the project.
   - You are expected to identify the task you will work on in the project. For example, you can work on traffic flow prediction, traffic congestion detection, and traffic accident detection. You are welcome to identify your own task and provide a brief description of the task.
2. Data collection, preprocessing, and visualization.
   - You are expected to collect the data you will use in the project, preprocess the data, and visualize the data. Typically, you will need to build a `Dataset` class to load the data, and use `DataLoader` to load the data in batches. You may also need to preprocess the data, for example, by normalizing the data, and resizing the images using `transforms` pipe. You are also expected to visualize the data to understand the data distribution and the data quality.
3. Model development.
   - You are expected to develop a model to solve the task you identified in step 1. You are expected to train a model or use a pre-trained model and fine-tune it on the dataset. You are also expected to use transfer learning to improve the model performance.
4. Model evaluation.
   - You are expected to evaluate the model you developed in step 3. You are expected to use appropriate evaluation metrics to evaluate the model performance.
5. Results
    - You are expected to present the results you obtained in the project. You may evaluate the results with traditional statistical methods. You are also expected to visualize the results to understand the model performance.
6. Application
    - You are expected to demonstrate how the model can be used in real-world applications. For example, you can demonstrate how to predict traffic flow in strategic/major roads in Hong Kong with your model. You are welcome to propose your own application and provide a brief description of the application

## Topic 4: Self-proposed project

You can also propose your own projects. However, the topic of the project should be closely related to the course material, and it must be approved by the instructor. The requirements for the self-proposed project are the same as the other projects. You should submit a single page proposal for your project at before 23:59 of 15 March 2024, including the following information:

- Project description
- Dataset description
- Tasks description
- Expected results
- References
- Any additional information you want to include

