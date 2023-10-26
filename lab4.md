# Lab3: House Price Prediction (Part 1) with Orange3

---

## Introduction

In this lab, we will be using Orange3 to predict the house price. We will be using the Boston House Prices dataset. The Boston House Prices dataset is a real-world dataset that contains information collected by the U.S Census Service concerning housing in the area of Boston Mass during 1978.

Hose price prediction is a regression problem. It's a supervised learning problem. In supervised learning, we have a dataset that contains both the input and the output. The input is the independent variable and the output is the dependent variable. The goal of supervised learning is to learn a function that maps the input to the output. The function is called a model. The model is learned from the training dataset. The model is then used to predict the output for the unseen input.

### About the Dataset

It's a real-world dataset that contains information collected by the U.S Census Service concerning housing in the area of Boston Mass during 1978. The dataset contains 506 instances and 14 attributes. The dataset contains the following attributes:

- `CRIM`: per capita crime rate by town
- `ZN`: proportion of residential land zoned for lots over 25,000 sq.ft.
- `INDUS`: proportion of non-retail business acres per town
- `CHAS`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- `NOX`: nitric oxides concentration (parts per 10 million)
- `RM`: average number of rooms per dwelling
- `AGE`: proportion of owner-occupied units built prior to 1940
- `DIS`: weighted distances to five Boston employment centres
- `RAD`: index of accessibility to radial highways
- `TAX`: full-value property-tax rate per \$10,000
- `PTRATIO`: pupil-teacher ratio by town
- `MEDV`: Median value of owner-occupied homes in \$1000's

## Getting Started

Let's start by opening Orange3 and creating a new workflow. We will be using the Boston House Prices dataset. You can download the dataset from [https://www.kaggle.com/vikrishnan/boston-house-prices](https://www.kaggle.com/vikrishnan/boston-house-prices) and load the csv file into Orange3 with `File` widget. But actually Orange3 already has the dataset built-in, so we don't need to download the dataset.

Let's drag the `File` widget to the canvas. The `File` widget allows us to load a dataset from a file. We can also use the `URL` widget to load a dataset from a URL. We will be using the `File` widget to load the dataset from the `housing.tab` file.

## Data Exploration

After loading a dataset, we usually want to explore the dataset. We want to know the number of instances and attributes in the dataset. We also want to know the data type of each attribute. We can use the `Data Table` widget to explore the dataset. The `Data Table` widget displays the dataset in a tabular format. It displays the number of instances and attributes in the dataset. It also displays the data type of each attribute.

### Data Table

Please connect the `File` widget to the `Data Table` widget. The `Data Table` widget will display the dataset in a tabular format. It displays the number of instances and attributes in the dataset. It also displays the data type of each attribute.

![Data Table](images/data-table.png)

We may check `Visualize numeric values` to display the distribution of each attribute in a histogram. We may also check `Visualize categorical values` to display the distribution of each attribute in a bar chart.

### Distributions

We can use the `Distributions` widget to explore the distribution of each attribute. The `Distributions` widget displays the distribution of each attribute in the dataset. It displays the distribution of each attribute in a histogram. It also displays the mean, median, standard deviation, and other statistics of each attribute.

Please connect the `File` widget to the `Distributions` widget. The `Distributions` widget will display the distribution of each attribute in the dataset. It displays the distribution of each attribute in a histogram. It also displays the mean, median, standard deviation, and other statistics of each attribute.

### Scatter Plot

We can use the `Scatter Plot` widget to explore the relationship between two attributes. The `Scatter Plot` widget displays the relationship between two attributes in a scatter plot. It displays the relationship between two attributes in a scatter plot. It also displays the correlation coefficient between two attributes.

![Distributions](images/distributions.png)

Please connect the `File` widget to the `Scatter Plot` widget. The `Scatter Plot` widget will display the relationship between two attributes in a scatter plot. It displays the relationship between two attributes in a scatter plot. It also displays the correlation coefficient between two attributes.

You may click on `Find informative projections` to see which two attributes are the most informative. It will short all pairs in descending order of their mutual information.

## Data Preprocessing

After exploring the dataset, we usually want to preprocess the dataset. But for this dataset, we don't need to preprocess the dataset. The dataset is already clean. It doesn't contain any missing values. It doesn't contain any categorical attributes. All attributes are numeric attributes. All attributes are continuous attributes. All attributes are independent attributes. There is no dependent attribute.

## Data Modeling

After preprocessing the dataset, we usually want to build a model. We want to learn a function that maps the input to the output. The function is called a model. The model is learned from the training dataset. The model is then used to predict the output for the unseen input.

### Linear Regression

We can use the `Linear Regression` widget to build a linear regression model. The `Linear Regression` widget builds a linear regression model. It learns a function that maps the input to the output. The function is called a model. The model is learned from the training dataset. The model is then used to predict the output for the unseen input.