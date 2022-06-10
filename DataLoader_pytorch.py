#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import shutil


# In[3]:




# In[4]:




# In[5]:




# In[6]:

# In[2]:


def createFolder(directory): #현재 경로에 Train / Test / Val 폴더 생성
    if not os.path.exists(directory):
        os.mkdir(directory)
        
    elif os.path.exists(directory):
        print('Already exists')
        
    else:
        print('Error: Creating directory. ' + directory)
        





# In[62]:


def window_torch(df, x_size, shift, x_columns, y_columns): # size = 윈도우 사이즈 / shift = 윈도우 시작점 얼만큼 이동할건지
    df_X =False
    for i in range(len(df)):
        if df_X is False:
            df_X = torch.tensor(df[i][x_columns].values, dtype=torch.float32)
            df_X = df_X.unfold(0, x_size, shift).transpose(1,2)
            
        else:
            tmp1 = torch.tensor(df[i][x_columns].values, dtype=torch.float32)
            tmp1 = tmp1.unfold(0,x_size, shift).transpose(1,2)
            df_X = torch.cat([df_X, tmp1],0)
            
    df_Y = False
    for i in range(len(df)):
        if df_Y is False:
            df_Y = torch.tensor(df[i][y_columns][(x_size-1):].values, dtype=torch.float32)
            df_Y = df_Y.unfold(0, 1, shift).transpose(1,2)
        
        else:
            tmp2 = torch.tensor(df[i][y_columns][(x_size-1):].values, dtype=torch.float32)
            tmp2 = tmp2.unfold(0, 1, shift).transpose(1,2)
            df_Y = torch.cat([df_Y, tmp2],0)
            
    return df_X, df_Y


# In[63]:




# In[64]:



