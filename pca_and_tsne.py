import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
from scipy import sparse
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
imagepath = './image/'
path = './array/'
trainfilter = 'filter_'
trainfilterx = 'filterx_'
targetepoch = 10
x_array_fiter = np.load(path+trainfilter+str(targetepoch)+'_0.npy')
x_array_filter_x = np.load(path+trainfilterx+str(targetepoch)+'_0.npy')

for i in range(1, 15):
    tmep_arry = np.load(path + trainfilter + str(targetepoch) + '_' + str(i) + '.npy')
    x_array_fiter = np.concatenate((x_array_fiter, tmep_arry), axis=0)
for i in range(1, 29):
    tmep_arry = np.load(path + trainfilterx + str(targetepoch) + '_' + str(i) + '.npy')
    x_array_filter_x = np.concatenate((x_array_fiter, tmep_arry), axis=0)
print(x_array_fiter.shape)
print(x_array_filter_x.shape)
total_arry = np.concatenate((x_array_fiter, x_array_filter_x), axis=0)
print(total_arry.shape)

pca = PCA(n_components=30)
pcaData =pca.fit_transform(total_arry)
X_embedded2 = TSNE(n_components=2).fit_transform(pcaData)
colors = ['#e31809', '#7851B8']
colorindex =0
for i in range(total_arry.shape[0]): # 0부터  digits.data까지 정수
    print(i)
    print(colorindex)
    if int(i/4334) == 0:
        colorindex = 0
    else :
        colorindex = 1
    plt.plot(X_embedded2[i, 0], X_embedded2[i, 1], color=colors[colorindex],marker='.',alpha=0.5) # 색상
plt.xlim(X_embedded2[:, 0].min(), X_embedded2[:, 0].max()) # 최소, 최
plt.ylim(X_embedded2[:, 1].min(), X_embedded2[:, 1].max()) # 최소, 최대
plt.xlabel('t-SNE ') # x축 이름
plt.ylabel('t-SNE ') # y축 이름
green_patch = mpatches.Patch(color= colors[0], label='filter')
purple_patch = mpatches.Patch(color=colors[1], label='filter_x')
plt.legend(handles=[green_patch, purple_patch])
plt.savefig(imagepath+trainfilter+'.png')
plt.close()