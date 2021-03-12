import numpy as np
from numpy.linalg import svd
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import sparse
import os


def load_hadden():
    #unique_sid 란 train data의 에서 뽑은 unique한 moveId
    unique_sid = list()
    with open(os.path.join("datasets", "Hadden","5", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    #unique_sid size
    n_items = len(unique_sid)

    # set args
    #default binary
    global input_size
    input_size = [1, 1, n_items]
    """
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False
    """

    # start processing
    # data 구성 뒤섞여 있는 data 기반으로 unique uid 가 mapping 되어있음
    # 따라서 uid의 갯수 0부터 20000개 제외
    # train test uid 10000개씩
    # data는 sparse matrix를 만듬 data userId*moveId 평면에 해당 죄표를 1로 찍는다
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1
        print(n_users)
        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data
    # data는 sparse matrix를 만듬 data userId*moveId 평면에 해당 죄표를 1로 찍는다

    # train, validation and test data
    # data는 sparse matrix를 만듬 train dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_train = load_train_data(os.path.join("datasets", "Hadden", "5",'train.csv'))
    # x_train 섞기
    np.random.shuffle(x_train)
    return x_train
train_data = load_hadden()
print(train_data.shape)

U,Sigma,Vt = svd(train_data,full_matrices=True)
print(U.shape,Sigma.shape,Vt.shape)
Sigma_mat = np.zeros(train_data.shape)
Sigma_mat[:Sigma.shape[0],:Sigma.shape[0]] = np.diag(Sigma)
reconstruct_data = np.dot(U, np.dot(Sigma_mat, Vt))

mse = ((train_data - reconstruct_data)**2).mean()
print(mse)