import numpy as np
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


batch_size = 300
learning_rate = 0.0002
num_epoch = 10
targetepoch = -1
arrayname = 'filter_'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_hadden():
    #unique_sid 란 train data의 에서 뽑은 unique한 moveId
    unique_sid = list()
    with open(os.path.join("datasets", "Hadden","5", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    #unique_sid size
    n_items = len(unique_sid)
    print(n_items)
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
    x_train = load_train_data(os.path.join("datasets", "Hadden", "5",'train_3.csv'))
    # x_train 섞기
    np.random.shuffle(x_train)

    # idle y's
    # x_train.shape[0] train uid 갯수
    # (uid,1) 이 0인 y_train
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    #input output
    #DataLoader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True,)
    print("train",len(train_loader))
    return train_loader

train_loader = load_hadden()
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(np.prod(input_size), 200)
        self.decoder = nn.Linear(200, np.prod(input_size))

    def forward(self, x, epoch,batch_idx):
        encoded = self.encoder(x)
        if epoch  == targetepoch :
            additionalTestPath = './array'
            xArray =np.array(encoded.tolist())
            np.save(os.path.join(additionalTestPath, arrayname + str(epoch) + "_" + str(batch_idx) + ".npy"),
                    xArray)
        out = self.decoder(encoded)
        return out

model = Autoencoder().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def getDifferenceRate(x,output):
    cpu_x = x.cpu().numpy()
    cpu_output = output.cpu()
    n_cpu_output = cpu_output.detach().numpy()
    print(cpu_x.shape)
    diff = cpu_x - n_cpu_output
    diff = np.abs(diff)
    print(n_cpu_output.shape)
    print("sum",np.sum(diff))
for epoch in range(10):
    for batch_idx, [image, label] in enumerate(train_loader):
        x = image.to(device)

        optimizer.zero_grad()
        output = model.forward(x,epoch,batch_idx)
        loss = loss_func(output, x)
        loss.backward()
        optimizer.step()
        if(epoch == 99):
            getDifferenceRate(x,output)
    print(loss)
