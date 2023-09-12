# Lab 2: Data Mining

---

# Overview

Let's start with a brief introduction to data mining, the tools and the dataset we will be using in this lab.

## Background

Data mining is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems. It is an essential process where intelligent methods are applied to extract data patterns. It is an interdisciplinary subfield of computer science.

The overall goal of the data mining process is to extract information from a data set and transform it into an understandable structure for further use. Aside from the raw analysis step, it involves database and data management aspects, data pre-processing, model and inference considerations, interestingness metrics, complexity considerations, post-processing of discovered structures, visualization, and online updating. Data mining is the analysis step of the "knowledge discovery in databases" process (KDD).

## Data Mining Tools: Orange3 in Python

In this series of labs, we will learn how to use the data mining tools in Orange to analyze datasets. Orange is a component-based data mining and machine learning software suite written in the Python Programming language. It features a visual programming front-end for explorative rapid qualitative data analysis and interactive data visualization. It allows user to create data analysis workflows, assemble and run them, and visualize the obtained data and intermediate results cooperatively with Python code. Making us write less code and focus on the data analysis.

It is a free software released under the terms of the GNU General Public License. Orange is cross-platform and works on Windows, macOS, and Linux. It can be installed in a **Python virtual environment** via `pip` package manager or `conda` package and environment manager.

```bash
pip install -U orange3
```

## Dataset: Amazon Review Data (2018)

Amazon review data (2018) is a large collection of reviews and metadata from Amazon products. The data is available on [Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews). The dataset contains 233.1 million reviews spanning May 1996 - Oct 2018. It contains reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014 for various products like books, electronics, movies, etc. This dataset is a slightly cleaned-up version of the data available at [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

---

# Data Mining with Orange

Let's move on to the data mining part of this lab. We will be using Orange3 to analyze the Amazon Review Data (2018) dataset.

## Getting Started

Let's start by opening Orange3 and creating a new workflow. We will be using the Amazon Review Data (2018) dataset. You can download the dataset from [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). The dataset is available in json format. We will be using the `Magazine_Subscriptions.json.gz` file.

After that, we will need to decompress the file. You can use the `gunzip` command to decompress the file in Linux or macOS. For Windows, you can use [7-Zip](https://www.7-zip.org/) to decompress the file.

```bash
gunzip Magazine_Subscriptions.json.gz
```

## Loading the Dataset

Let's start by loading the dataset into Orange3. We will be using the `File` widget to load the dataset. The `File` widget allows us to load a dataset from a file. We can also use the `URL` widget to load a dataset from a URL. We will be using the `File` widget to load the dataset from the `reviews_Electronics.json.gz` file.

We will need a Python Script widget to load the dataset into Orange3. Please create a new Python Script widget with following code.

```python
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
import gzip
import json

domain = Domain([
    ContinuousVariable("overall"),
    ContinuousVariable("verified"),
    ContinuousVariable("unixReviewTime")
], metas=[
    StringVariable("reviewTime"),
    StringVariable("reviewerID"),
    StringVariable("asin"),
    StringVariable("style"),
    StringVariable("reviewerName"),
    StringVariable("reviewText"),
    StringVariable("summary")
])

arr = []
meta = []

with gzip.open('Magazine_Subscriptions.json.gz', 'rb') as f:
    for line in f:
        record = json.loads(line)
        if record['verified']:
            record['verified'] = 1
        else:
            record['verified'] = -1

        if 'style' in record.keys():
            record['style'] = record['style']['Format:']
        else:
            record['style'] = 'None'

        if 'reviewText' not in record.keys():
            record['reviewText'] = ''
        if 'summary' not in record.keys():
            record['summary'] = ''
        if 'reviewerName' not in record.keys():
            record['reviewerName'] = ''

        arr.append([record['overall'], record['verified'], record['unixReviewTime']])
        meta.append([record['reviewTime'], record['reviewerID'], record['asin'], record['style'], record['reviewerName'], record['reviewText'], record['summary']])

out_data = Table.from_numpy(domain, arr, metas=meta)
```