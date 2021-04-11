from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy import sparse
import os

import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ======================================================================================================================
def load_ml20m(dataset_index,args, **kwargs):
    pro_dir = ""
    if int(dataset_index/5) == 0:
        pro_dir = 'productfilter50'
    else:
        pro_dir = 'productfilter50_module2'
        dataset_index -=5

    #unique_sid 란 train data의 에서 뽑은 unique한 moveId
    unique_sid = list()
    with open(os.path.join("datasets", "ML_20m",pro_dir, 'unique_sid_'+str(dataset_index)+'.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    #unique_sid size
    n_items = len(unique_sid)

    # set args
    #default binary
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    # data 구성 뒤섞여 있는 data 기반으로 unique uid 가 mapping 되어있음
    # 따라서 uid의 갯수 0부터 20000개 제외
    # train test uid 10000개씩
    # data는 sparse matrix를 만듬 data userId*moveId 평면에 해당 죄표를 1로 찍는다
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data
    # data는 sparse matrix를 만듬 data userId*moveId 평면에 해당 죄표를 1로 찍는다
    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    # data는 sparse matrix를 만듬 train dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_train = load_train_data(os.path.join("datasets", "ML_20m",pro_dir, 'train_'+str(dataset_index)+'.csv'))
    # x_train 섞기
    np.random.shuffle(x_train)
    # data는 sparse matrix를 만듬 val dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "ML_20m",pro_dir, 'validation_tr_'+str(dataset_index)+'.csv'),
                                         os.path.join("datasets", "ML_20m",pro_dir, 'validation_te_'+str(dataset_index)+'.csv'))
    # data는 sparse matrix를 만듬 test dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "ML_20m",pro_dir, 'test_tr_'+str(dataset_index)+'.csv'),
                                           os.path.join("datasets", "ML_20m",pro_dir, 'test_te_'+str(dataset_index)+'.csv'))

    # idle y's
    # x_train.shape[0] train uid 갯수
    # (uid,1) 이 0인 y_train
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    #input output
    #DataLoader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    # default false
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        # userId[0: input(default 1000)]의 movieId의 대칭행렬
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_netflix(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("datasets", "Netflix", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "Netflix", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Netflix", 'validation_tr.csv'),
                                         os.path.join("datasets", "Netflix", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Netflix", 'test_tr.csv'),
                                           os.path.join("datasets", "Netflix", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)
    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_pinterest(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("datasets", "Pinterest", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "Pinterest", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Pinterest", 'validation_tr.csv'),
                                         os.path.join("datasets", "Pinterest", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Pinterest", 'test_tr.csv'),
                                           os.path.join("datasets", "Pinterest", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_melon1d(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'validation_tr.csv'),
                                         os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'test_tr.csv'),
                                           os.path.join("datasets", "Melon/v2-mat-mm-1d/pro_sg", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_melon8d(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te


    # train, validation and test data
    x_train = load_train_data(os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'train.csv'))
    np.random.shuffle(x_train)
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'validation_tr.csv'),
                                         os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'validation_te.csv'))

    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'test_tr.csv'),
                                           os.path.join("datasets", "Melon/v2-mat-mm-8d/pro_sg", 'test_te.csv'))

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args
def load_hadden(dataset_index,args, **kwargs):
    if dataset_index >= 5:
        dataset_index -= 5;
    print("kwargs",kwargs)
    dataset_index_arr = ['module1','module12']
    #unique_sid 란 train data의 에서 뽑은 unique한 moveId
    unique_sid = list()
    with open(os.path.join("datasets", "Hadden",dataset_index_arr[int(dataset_index/5)], 'unique_sid_'+str(dataset_index)+'.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    #unique_sid size
    n_items = len(unique_sid)

    # set args
    #default binary
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

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
    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())

        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
        
        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']
        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx+1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx+1, n_items)).toarray()
        return data_tr, data_te

    # dataset_index[int(dataset_index/5)], 'unique_sid_'+str(dataset_index)+'.txt'
    # train, validation and test data
    # data는 sparse matrix를 만듬 train dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_train = load_train_data(os.path.join("datasets", "Hadden", dataset_index_arr[int(dataset_index/5)],'train_'+str(dataset_index)+'.csv'))
    # x_train 섞기
    np.random.shuffle(x_train)
    # data는 sparse matrix를 만듬 val dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_val_tr, x_val_te = load_tr_te_data(os.path.join("datasets", "Hadden",dataset_index_arr[int(dataset_index/5)], 'validation_tr_'+str(dataset_index)+'.csv'),
                                         os.path.join("datasets", "Hadden",dataset_index_arr[int(dataset_index/5)], 'validation_te_'+str(dataset_index)+'.csv'))
    # data는 sparse matrix를 만듬 test dataset, userId*moveId 평면에 해당 죄표를 1로 찍는다
    x_test_tr, x_test_te = load_tr_te_data(os.path.join("datasets", "Hadden",dataset_index_arr[int(dataset_index/5)], 'test_tr_'+str(dataset_index)+'.csv'),
                                           os.path.join("datasets", "Hadden",dataset_index_arr[int(dataset_index/5)], 'test_te_'+str(dataset_index)+'.csv'))


    # idle y's
    # x_train.shape[0] train uid 갯수
    # (uid,1) 이 0인 y_train
    y_train = np.zeros((x_train.shape[0], 1))


    # pytorch data loader
    #input output
    #DataLoader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)
    print("train",len(train_loader))
    validation = data_utils.TensorDataset(torch.from_numpy(x_val_tr), torch.from_numpy(x_val_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test_tr).float(), torch.from_numpy(x_test_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    # default false
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        # userId[0: input(default 1000)]의 movieId의 대칭행렬
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()

    return train_loader, val_loader, test_loader, args
# ======================================================================================================================
def load_dataset(dataset_index,args, **kwargs):
    #ml20m
    if args.dataset_name == 'ml20m':
        train_loader, val_loader, test_loader, args = load_ml20m(dataset_index,args, **kwargs)
    elif args.dataset_name == 'netflix':
        train_loader, val_loader, test_loader, args = load_netflix(dataset_index,args, **kwargs)
    elif args.dataset_name == 'pinterest':
        train_loader, val_loader, test_loader, args = load_pinterest(dataset_index,args, **kwargs)
    elif args.dataset_name == 'melon1d':
        train_loader, val_loader, test_loader, args = load_melon1d(dataset_index,args, **kwargs)
    elif args.dataset_name == 'melon8d':
        train_loader, val_loader, test_loader, args = load_melon8d(dataset_index,args, **kwargs)
    elif args.dataset_name == 'hadden':
        train_loader, val_loader, test_loader, args = load_hadden(dataset_index,args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
