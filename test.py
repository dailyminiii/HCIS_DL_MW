import pandas as pd
import torch
from scipy import interpolate
import numpy as np
import scipy.stats as ss
import dataset
from torch.utils.data import DataLoader

trainset = dataset.NumDataset("data/ProcessedData_1.csv",1, 3, 1)
print(trainset)

trainloader = DataLoader(trainset,
                             batch_size=16,
                             shuffle=True, drop_last=True)

for i, (X, y) in enumerate(trainloader):

    # print(i, 'X :', X, 'Y :', y)
    print('X shape : ' ,X.shape, 'y shape :', y.shape)