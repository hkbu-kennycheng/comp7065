# Lab 5: Recurrent neural network (RNN)

---

# Overview

In this lab, you will learn how to use Pytorch to build a recurrent neural network (RNN) for stock price prediction. You will learn some of the technical indicators that are commonly used in stock price prediction. You will also learn how to evaluate the performance of the RNN.

## Introduction

Stock price prediction is a challenging task. There are many factors that can affect the stock price. Some of the factors are related to the company itself, such as the company's financial status, the company's management, and the company's products. Some of the factors are related to the market, such as the market sentiment, the market trend, and the market volatility. They made the stock price fluctuate a lot. It is very difficult to predict the stock price accurately. However, recent studies have shown that deep learning may feasible to predict the stock price. It's also a good demonstration of how to use deep learning to tackle time series data problems in real life.

### Recurrent neural network (RNN)

Recurrent neural network (RNN) is a family of neural network that is suitable for processing sequential data. RNN is widely used in natural language processing (NLP) and speech recognition. In this lab, you will learn how to use RNN to predict stock price. Since stock price is a time series data, RNN is a good choice for this task.  You will also learn how to evaluate the performance of the RNN.

There are three types of RNN: RNN, LSTM, and GRU. In this lab, we will use GRU to predict the stock price. GRU is a simplified version of LSTM. It is easier to train and faster to compute than LSTM.

### Deep learning framework

[Tensorflow](https://www.tensorflow.org/) and [Pytorch](https://pytorch.org/) are two most popular deep learning frameworks. They are both open source and powerful. Recently, Pytorch is getting more and more popular due to community support, especially the [Hugging Face](https://huggingface.co/) community. There are nearly half of million of pre-trained models on Hugging Face that you can use directly.

In this lab, we will use Pytorch to build the RNN model for stock price prediction. Pytorch is a deep learning framework developed by Facebook. It is very popular in the research community. It is also very easy to use. You can learn more about Pytorch from [here](https://pytorch.org/).

## Objectives

In this lab, you will learn how to:

- Fetch financial data from Finnhub API and Yahoo Finance
- Build a recurrent neural network (RNN) for stock price prediction
- Understand RSI, MACD, Bollinger Bands technical indicators and some more technical indicators
- Predict stock price change percentage with RNN
- Evaluate the performance of the model
- Understand the limitations of the model

## Dependencies

You will need the following dependencies installed to complete this lab:

- Python
- Pytorch
- Jupyter notebook
- Numpy
- Pandas
- [Pandas TA](https://github.com/twopirllc/pandas-ta)
- Matplotlib
- yfinance
- tqdm
- finnhub-python

You could install the dependencies with `pip` command. For example, you can install Pytorch with the following command:

```jupyter
pip install torch torchvision torchaudio jupyter numpy pandas pandas_ta matplotlib yfinance tqdm finnhub-python
```

# Getting Started

In this lab, we will use jupyter notebook to build the RNN model. Please create a new jupyter notebook file and name it as `lab5.ipynb`.

## Fetch and explore the financial data

There are many ways to fetch financial data. We will first take a look to Finnhub API. Finnhub is a free API that you can use to fetch financial data. You can register a free account and get an API key from [here](https://finnhub.io/). After you get the API key, you could use their Python SDK to fetch the financial data.

### Market news

Market news could affect the stock price a lot. Let's first take a look at how to fetch the market news from Finnhub API. You could use the following command to fetch the market news:

```python
import finnhub
finnhub_client = finnhub.Client(api_key="YOUR_API_KEY")
print(finnhub_client.general_news('general', min_id=0))
```

### Search for stock symbols

To search for stock symbols, at least you will need to know the company name. You could use the following command to search for stock symbols:

```python
print(finnhub_client.symbol_lookup('apple'))
```

### Company news

Previously, we have learned how to fetch the market news. Now, let's take a look at how to fetch the news of a specific company. You could use the following command to fetch the company

```python
print(finnhub_client.company_news('AAPL', _from="2023-12-01", to="2020-01-27"))
```

### Real-time stock price

For getting the real-time quote, you could use the following command to fetch the real-time stock price:

```python
print(finnhub_client.quote('AAPL'))
```

### Historical stock price

For fetching historical stock price, we will use [yfinance](https://pypi.org/project/yfinance/) to fetch the stock price data. It is a very convenient package to fetch stock price data from Yahoo Finance. After you install the package, you could import it with the following command:

```python
import yfinance as yf

aapl = yf.Ticker('AAPL').history(interval='1d', start='2020-01-01', end='2020-12-31')
aapl
```

It would return a pandas dataframe that contains the historical stock price data.

### Explore the stock price data

Let's take a look at the stock price data. You could use the following command to print the stock price data:

```python
aapl.info()
```

```bash
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 252 entries, 2020-01-02 00:00:00-05:00 to 2020-12-30 00:00:00-05:00
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Open          252 non-null    float64
 1   High          252 non-null    float64
 2   Low           252 non-null    float64
 3   Close         252 non-null    float64
 4   Volume        252 non-null    int64  
 5   Dividends     252 non-null    float64
 6   Stock Splits  252 non-null    float64
dtypes: float64(6), int64(1)
memory usage: 15.8 KB
```

We may also interested in the statistical information of the stock price data. You could use the following command to print the statistical information of the stock price data:

```python
aapl.describe()
```

```bash
    Open        High         Low       Close        Volume   
count  252.000000  252.000000  252.000000  252.000000  2.520000e+02  \
mean    93.164491   94.532509   91.841041   93.247567  1.577966e+08   
std     21.678211   21.768848   21.255574   21.489125  6.987198e+07   
min     55.682552   55.785089   51.905767   54.776806  4.669130e+07   
25%     75.274138   76.108155   74.430931   75.405560  1.113394e+08   
50%     89.365563   90.862496   88.850268   89.518570  1.381294e+08   
75%    113.602338  115.031478  111.992970  113.514061  1.875871e+08   
max    135.654209  136.381357  132.008588  134.317825  4.265100e+08   

        Dividends  Stock Splits  
count  252.000000    252.000000  
mean     0.003204      0.015873  
std      0.025291      0.251976  
min      0.000000      0.000000  
25%      0.000000      0.000000  
50%      0.000000      0.000000  
75%      0.000000      0.000000  
max      0.205000      4.000000  
```

## Visualize the stock price

After fetching the stock price data, we could visualize the stock price data with [matplotlib](https://matplotlib.org/). You could use the following command to plot the historical Open and Close price of each day:

```python
import matplotlib.pyplot as plt

plt.plot(aapl.index, aapl['Open'], label="Open")
plt.plot(aapl.index, aapl['Close'], label="Close")
plt.legend()
plt.show()
```

We could also plot the historical high and low price of each day:

```python
plt.plot(aapl.index, aapl['High'], label="High")
plt.plot(aapl.index, aapl['Low'], label="Low")
plt.legend()
plt.show()
```

## Technical indicators

Technical indicators are widely used in stock price prediction. There are many technical indicators that you could use. In this lab, we will use the following technical indicators:

- Moving Average Convergence Divergence (MACD)
- Bollinger Bands (BB)
- Average Directional Index (ADX)
- Average true range (ATR)
- T3 Moving Average (T3)
- Money Flow Index (MFI)
- On-balance volume (OBV)
- LogReturn indicator
- Rolling Z Score indicator
- Qstick indicator

# Case Study: Microsoft (MSFT)

In this lab, we will use Microsoft (MSFT) as an example to demonstrate how to use RNN to predict the stock price. We will use the historical stock price data from `2020-01-01` to `2020-12-31` to train the model. Then, we will use the model to predict the stock price from `2021-01-01` to `2021-01-31`. We will use the following technical indicators to predict the stock price:

