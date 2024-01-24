# Lab 5: Recurrent neural network (RNN)

---

# Overview

In this lab, you will learn how to use Pytorch to build a recurrent neural network (RNN) for stock price prediction. You will learn some of the technical indicators that are commonly used in stock price prediction. You will also learn how to evaluate the performance of the RNN.

## Introduction

Stock price prediction is a challenging task. There are many factors that can affect the stock price. Some of the factors are related to the company itself, such as the company's financial status, the company's management, and the company's products. Some of the factors are related to the market, such as the market sentiment, the market trend, and the market volatility. They made the stock price fluctuate a lot. It is very difficult to predict the stock price accurately. However, recent studies have shown that deep learning maybe feasible to predict the stock price. It's also a good demonstration of how to use deep learning to tackle time series data problems in real life.

Recurrent neural network (RNN) is a family of neural network that is suitable for processing sequential data. RNN is widely used in natural language processing (NLP) and speech recognition. In this lab, you will learn how to use RNN to predict stock price. Since stock price is a time series data, RNN is a good choice for this task.  You will also learn how to evaluate the performance of the RNN.

There are three types of RNN: RNN, LSTM, and GRU. In this lab, we will use GRU to predict the stock price. GRU is a simplified version of LSTM. It is easier to train and faster to compute than LSTM.

## Objectives

In this lab, you will learn how to:

- Build a recurrent neural network (RNN) for stock price prediction
- Understand RSI, MACD, Bollinger Bands technical indicators and some more technical indicators
- Predict stock price change percentage with RNN
- Evaluate the performance of the model
- Understand the limitations of the model

# Getting Started


In this lab, we will use jupyter notebook to build the RNN model. Please create a new jupyter notebook and name it as `lab5.ipynb`.

