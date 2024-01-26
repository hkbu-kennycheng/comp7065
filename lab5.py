#!/usr/bin/env python
# coding: utf-8

# In[8]:


#get_ipython().system('pip install pandas_ta yfinance tqdm TA-Lib')


# In[9]:


import pandas as pd
import pandas_ta as ta

df = pd.DataFrame()
df = df.ta.ticker("MSFT")

df


# In[10]:


# Create your own Custom Strategy
CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    description="SMA, BBANDS, RSI, MACD and Volume SMA",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 100},
        {"kind": "sma", "length": 150},
        {"kind": "bbands", "length": 10},
        {"kind": "bbands", "length": 20},
        {"kind": "bbands", "length": 50},
        {"kind": "bbands", "length": 100},
        {"kind": "bbands", "length": 1500},
        {"kind": "rsi", "length": 10},
        {"kind": "rsi", "length": 20},
        {"kind": "rsi", "length": 50},
        {"kind": "rsi", "length": 100},
        {"kind": "rsi", "length": 150},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "bop"},
        {"kind": "percent_return"},
        {"kind": "wcp"},
        {"kind": "pvi"},
        {"kind": "log_return"},
    ]
)
# To run your "Custom Strategy"
df.ta.strategy(CustomStrategy)
df


# In[11]:


df.dropna(inplace=True)
df.drop(columns=['Stock Splits'], inplace=True)
df['Volume'] = df['Volume'] / df['Volume'].max()


# In[12]:


# df['Target'] = df['Close'].shift(-1) - df['Open'].shift(-1)
df['Target'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
df.dropna(inplace=True)
df


# In[13]:


train_df = df[:'2023-03-31']
test_df = df['2023-04-01':]


# In[14]:


test_df


# In[15]:


import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader

sequence_size = 50
batch_size = 256
features_size = len(train_df.drop(['Target'], axis=1).columns)

class SequenceDataset(Dataset):

    def __init__(self, df=pd.DataFrame(), label='', sequence_size=30):
        self.df = df
        self.label = label

    def __len__(self):
        return len(self.df) - sequence_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = Tensor(np.array(self.df.drop(self.label, axis=1).iloc[idx:idx+sequence_size, :], dtype=float))
        label = Tensor(np.array(self.df[[self.label]].iloc[idx+sequence_size, :], dtype=float))

        return (seq, label)

train_data = TensorDataset(Tensor(np.array(train_df.drop(['Target'], axis=1))), Tensor(np.array(train_df['Target'])))
test_data = TensorDataset(Tensor(np.array(test_df.drop(['Target'], axis=1))), Tensor(np.array(test_df['Target'])))

# train_loader = DataLoader(train_data)
# test_loader = DataLoader(test_data)

# train_loader = DataLoader(PandasDataset(train_df, labels=['Target']))
# test_loader = DataLoader(PandasDataset(test_df, labels=['Target']))

train_loader = DataLoader(SequenceDataset(train_df, label='Target', sequence_size=sequence_size), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SequenceDataset(test_df, label='Target', sequence_size=sequence_size), batch_size=batch_size, shuffle=True)


# In[16]:


print(batch_size, features_size)


# In[17]:


list(enumerate(test_loader))[0]


# In[18]:


hiddenSize = 256

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]

model = nn.Sequential(
    nn.LSTM(features_size, hiddenSize, num_layers=4, batch_first=True),
    nn.Sequential(
      extract_tensor(),
      nn.Linear(hiddenSize, 1),
    )
)


# In[19]:


from torch import optim

loss_function = nn.GaussianNLLLoss()
# loss_function = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
# optimizer = optim.RAdam(model.parameters(), lr=1e-3)


# In[20]:


from tqdm import tqdm
import torch

NUM_EPOCHS = 400

model.train() # put model in training mode

for epoch in range(NUM_EPOCHS):
  
  loop = tqdm(train_loader, position=0, leave=True)
  
  running_loss = 0.0
  
  for (batch, labels) in loop:

    optimizer.zero_grad()
    model.to('cuda')
    output = model.forward(batch.to('cuda'))

    var = torch.ones(output.shape).to('cuda')
    loss = loss_function(output.to('cuda'), labels.to('cuda'), var)
#     loss = loss_function(output.to('cuda'), labels.to('cuda'))
    loss.to('cuda')
    loss.backward()

    optimizer.step()

    running_loss += loss.item()
    loop.set_postfix(epoch=epoch, loss=running_loss)
  
#   if running_loss < 700:
#     torch.save(model.state_dict(), f'lab5-loss{running_loss}.pt')
#     break


# In[129]:


model.eval() # put model in evaluation mode


# In[130]:


correct = 0
loop = tqdm(test_loader, position=0, leave=True)

for (batch, labels) in loop:
#     print(labels)
    output = model.forward(batch.to('cuda'))
    print(labels, labels.to('cuda') - output.to('cuda'))
# print(f'Acc = {correct/len(loop)}')


# In[ ]:


#correct = 0
#loop = tqdm(test_loader, position=0, leave=True)
#
#for (batch, labels) in loop:
#    output = model.forward(batch.to('cuda'))
#    print(output, labels)


# In[ ]:




