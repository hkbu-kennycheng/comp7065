# Lab 2: Data Mining

## Background

Data mining is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems. It is an essential process where intelligent methods are applied to extract data patterns. It is an interdisciplinary subfield of computer science.

The overall goal of the data mining process is to extract information from a data set and transform it into an understandable structure for further use. Aside from the raw analysis step, it involves database and data management aspects, data pre-processing, model and inference considerations, interestingness metrics, complexity considerations, post-processing of discovered structures, visualization, and online updating. Data mining is the analysis step of the "knowledge discovery in databases" process (KDD).

## Data Mining Tools: Orange3 in Python

In this series of labs, we will learn how to use the data mining tools in Orange to analyze datasets. Orange is a component-based data mining and machine learning software suite written in the Python Programming language. It features a visual programming front-end for explorative rapid qualitative data analysis and interactive data visualization. It allows user to create data analysis workflows, assemble and run them, and visualize the obtained data and intermediate results cooperatively with Python code. Making us write less code and focus on the data analysis.

It is a free software released under the terms of the GNU General Public License. Orange is c
```bash
```ross-platform and works on Windows, macOS, and Linux. It can be installed in a **Python virtual environment** via `pip` package manager or `conda` package and environment manager.

```bash
pip install -U orange3
```

## Dataset: Amazon Review Data (2018)

```bash
```
Amazon review data (2018) is a large collection of reviews and metadata from Amazon products. The data is available on [Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews). The dataset contains 233.1 million reviews spanning May 1996 - Oct 2018. It contains reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014 for various products like books, electronics, movies, etc. This dataset is a slightly cleaned-up version of the data available at [http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/).