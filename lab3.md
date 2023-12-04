# Lab 3: Data Mining - Similarity Matching

---

# Overview

In this lab, you will learn how to use the similarity matching techniques to find similar items in a dataset. You will try out different distance metrics and similarity measures to find similar items in a dataset. To order to work with a real-world dataset, you will learn how to convert a dataset into a format that can be used by the similarity matching algorithms.

## Background

Similarity Matching is a technique used to find similar items in a dataset. It is used in many applications such as finding similar documents, finding similar images, finding similar products, etc. In this lab, you will learn how to use similarity matching to find similar items in a dataset. You will also learn how to convert a dataset into a format that can be used by the similarity matching algorithms.

The key idea behind similarity matching is that similar items are close to each other in the dataset. Therefore, we can use the distance between two items to measure their similarity. The smaller the distance, the more similar the two items are. There are many different distance metrics and similarity measures. The most commonly used distance metrics are Euclidean distance and Manhattan distance. The most commonly used similarity measures are cosine similarity and Pearson correlation.

![screenshot https://en.wikipedia.org/wiki/Similarity_measure](https://url2png.hkbu.app/en.wikipedia.org/wiki/Similarity_measure)

A typical usage for similarity matching is in recommendation systems. For example, if you are watching a movie on Netflix, Netflix will recommend other movies that are similar to the one you are watching. In this lab, you will use similarity matching to find similar movies in the MovieLens dataset.

## Learning Objectives

After completing this lab, you should be able to:

- Use similarity matching to find similar items in a dataset.
- Convert a dataset into a format that can be used by the similarity matching algorithms.
- Use different distance metrics and similarity measures to find similar items in a dataset.
- Use similarity matching to find similar movies in the MovieLens dataset.
- Use Orange3 to find similar movies in the MovieLens dataset.

# Case Study: Movie Recommendation

In this lab, you will use similarity matching to find similar movies in the MovieLens dataset. Let's take a look at the MovieLens 100k dataset README on [https://files.grouplens.org/datasets/movielens/ml-100k-README.txt](files.grouplens.org/datasets/movielens/ml-100k-README.txt).

The MovieLens dataset contains 100,000 ratings from 943 users on 1,682 movies. The dataset can be downloaded from [here](https://files.grouplens.org/datasets/movielens/ml-100k.zip). The dataset contains the following files:

- `u.data`: The full dataset, 100,000 ratings by 943 users on 1,682 movies.
- `u.info`: The number of users, items, and ratings in the dataset.
- `u.item`: Information about the 1,682 movies. This is a tab separated list of:
  - movie id
  - movie title
  - release date
  - video release date
  - IMDb URL
  - unknown
  - Action
  - Adventure
  - Animation
  - Children's
  - Comedy
  - Crime
  - Documentary
  - Drama
  - Fantasy
  - Film-Noir
  - Horror
  - Musical
  - Mystery
  - Romance
  - Sci-Fi
  - Thriller
  - War
  - Western
- `u.genre`: A list of the genres.
- `u.user`: Demographic information about the users. This is a tab separated list of:
  - user id
  - age
  - gender
  - occupation
  - zip code
- `u.occupation`: A list of the occupations.

## Loading the Dataset into Orange3

Let's load the dataset into Orange3.

```python
import requests
from zipfile import ZipFile 

zip_url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

download_file(zip_url)
filename = zip_url.split('/')[-1]

with ZipFile(filename, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

```

First, download the dataset from [here](http://files.grouplens.org/datasets/movielens/ml-100k.zip). Then, unzip the dataset and open the `u.data` file in Orange3. The `u.data` file contains the full dataset, 100,000 ratings by 943 users on 1,682 movies. The `u.data` file is a tab separated list of: