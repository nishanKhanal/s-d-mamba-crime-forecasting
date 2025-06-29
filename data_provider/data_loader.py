import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        print(type(data))
        print(data)
        print(data['data'].shape)
        print(data['data'])
        data = data['data'][:, :, 0]

        # print(data.shape)
        # print(data)
        # exit()
        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Crime(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='crime_1_week_pivot.csv',
                 target='OT', scale=True, timeenc=1, freq='w'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = True
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # variables = [(k,v) for k, v in self.__dict__.items() 
        
        #     # variable is not a python internal variable
        #     if not k.startswith("__") 
        #     and not k.endswith("__")
            
        #     # and is not a function
        #     and not "method" in str(v)
        #     and not "function" in str(v)]

        # print(variables)
        # exit()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        print(cols)
        if self.target in cols:
            cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols ] # + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Synthetic(Dataset):
    def __init__(self, root_path=None, data_path= None, flag='train', 
                size=None, features='M', syn_data_params={}, 
                scale=True, timeenc=1, freq='w', target=None):
        # Sequence lengths
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.syn_data_params = syn_data_params

        # --- Generate synthetic data ---
        self.N = syn_data_params.get('N', 77)
        self.T = syn_data_params.get('T', 1253)
        self.mode = syn_data_params.get('mode', 'none')

        self.__read_data__()

    def __read_data__(self):
        # Generate synthetic data: shape [N, T]
        Y = self.generate_dataset(
            N=self.N,
            T=self.T,
            mode=self.mode,
            w=self.syn_data_params.get('w', 1.0),
            noise_strength=self.syn_data_params.get('noise_strength', 0.1),
            dt=self.syn_data_params.get('dt', np.pi/45)
        )  # [N, T]

        data = Y.T # [T, N]

        # Prepare as DataFrame for feature compatibility
        node_cols = [f"node_{i}" for i in range(self.N)]
        df_raw = pd.DataFrame(data, columns=node_cols)
        self.target = node_cols[0]  # Use first node as target for 'S' mode

        # Train/val/test splits
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scaling
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_scaled = self.scaler.transform(df_data.values)
        else:
            data_scaled = df_data.values

        # --- TIME FEATURES ---
        if self.timeenc == 0:
            # Option 1: Calendar features
            dates = pd.date_range(start='2000-01-01', periods=len(df_raw), freq=self.freq)
            df_stamp = pd.DataFrame({'date': dates})
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour if self.freq[0] == 'H' else 0
            data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
        elif self.timeenc == 1:
            dates = pd.date_range(start='2001-01-01', periods=len(df_raw), freq='7D')

            data_stamp = time_features(pd.to_datetime(dates.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        else:
            # Option 2: Cyclical features
            t = np.arange(len(df_raw))
            sin1 = np.sin(2 * np.pi * t / 24)
            cos1 = np.cos(2 * np.pi * t / 24)
            sin2 = np.sin(2 * np.pi * t / 7)
            cos2 = np.cos(2 * np.pi * t / 7)
            data_stamp = np.stack([sin1, cos1, sin2, cos2], axis=1)  # [T, 4]

        self.data_x = data_scaled[border1:border2]
        self.data_y = data_scaled[border1:border2] 
        self.data_stamp = data_stamp[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def generate_dataset(self, 
        N=77,                # Number of nodes
        T=200,               # Number of time steps
        mode="temporal",        # "none", "temporal", "spatial", "spatio-temporal"
        w=1.0,              # Frequency for sin
        noise_strength=0.1,      # Standard deviation of additive noise
        dt= np.pi / 45,      # time increment
        ):
        """
        Generate synthetic spatio-temporal data using f(a, b, t) = a * sin(w * t) + b.
        Modes:
            - "none": a, b, t all random
            - "temporal": a, b random per node; t evolves over time
            - "spatial": a, b spatially correlated; t random
            - "spatio-temporal": a, b and t all evolve, using simple AR(1) and spatial neighbor mean
        """
        Y = np.zeros((N, T))

        if mode == "none":
            for i in range(N):
                for t in range(T):
                    a = np.random.uniform(-1, 1)
                    b = np.random.uniform(-1.0, 1.0)
                    t_rand = np.random.uniform(0, 2 * np.pi)
                    Y[i, t] = a * np.sin(w * t_rand) + b + np.random.randn() * noise_strength

        elif mode == "temporal":
            a = np.random.uniform(-1, 1, size=N)
            b = np.random.uniform(-1.0, 1.0, size=N)
            t0 = np.random.uniform(0, 2 * np.pi, size=N)
            for i in range(N):
                for t in range(T):
                    t_curr = t0[i] + dt * t
                    Y[i, t] = a[i] * np.sin(w * t_curr) + b[i] + np.random.randn() * noise_strength

        elif mode == "spatial":
            t_rand = np.random.uniform(0, 2 * np.pi, T)
            a = np.zeros(N)
            b = np.zeros(N)
            # Node 0 random, others depend on previous
            a[0] = np.random.uniform(-1, 1)
            b[0] = np.random.uniform(-1.0, 1.0)
            for i in range(1, N):
                a[i] = 0.8 * a[i-1] + 0.2 * np.random.uniform(-1, 1)
                b[i] = 0.8 * b[i-1] + 0.2 * np.random.uniform(-1.0, 1.0)
            for i in range(N):
                for t in range(T):
                    Y[i, t] = a[i] * np.sin(w * t_rand[t]) + b[i] + np.random.randn() * noise_strength

        elif mode == "spatio-temporal":
            a = np.zeros((N, T))
            b = np.zeros((N, T))
            t_seq = np.zeros((N, T))
            # Start from random for all
            a[:, 0] =  np.random.uniform(-1,1, (N))
            b[:, 0] =  np.random.uniform(-1,1, (N))
            t_seq[:, 0] = np.random.uniform(0, 2* np.pi, (N))
            for t in range(1, T):
                for i in range(N):
                    # For spatial effect, use previous node if exists, else just self
                    neighbor_a = a[i-1, t-1] if i > 0 else a[i, t-1]
                    neighbor_b = b[i-1, t-1] if i > 0 else b[i, t-1]
                    a[i, t] = 0.45 * a[i, t-1] + 0.45 * neighbor_a + 0.1 * np.random.uniform(-1, 1)
                    b[i, t] = 0.45 * b[i, t-1] + 0.45 * neighbor_b + 0.1 * np.random.uniform(-1.0, 1.0)
                    t_seq[i, t] = t_seq[i, t-1] + dt + np.random.randn() * 0.01
            for i in range(N):
                for t in range(T):
                    Y[i, t] = a[i, t] * np.sin(w * t_seq[i, t]) + b[i, t] + np.random.randn() * noise_strength
        else:
            raise ValueError("mode must be 'none', 'temporal', 'spatial', or 'spatio-temporal'")

        return Y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Synthetic_Through_Rotation(Dataset):
    def __init__(self, root_path=None, data_path= None, flag='train', 
                size=None, features='M', syn_data_params={}, 
                scale=True, timeenc=1, freq='w', target=None):
        # Sequence lengths
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.syn_data_params = syn_data_params

        self.N = syn_data_params.get('N', 77)
        self.T = syn_data_params.get('T', 1253)
        self.pad_mode = syn_data_params.get('pad_mode', 'manual')
        self.theta_deg = syn_data_params.get('theta_deg', None)

        self.__read_data__()

    def __read_data__(self):
        # Generate synthetic data: shape [N, T]

        Y, _ = self.generate(N=self.N, T=self.T, theta_deg=self.theta_deg, pad_mode=self.pad_mode, 
                      dt=self.syn_data_params.get('dt', np.pi/45),
                      noise_strength=self.syn_data_params.get('noise_strength', 0.1))
        

        data = Y.T # [T, N]

        # Prepare as DataFrame for feature compatibility
        node_cols = [f"node_{i}" for i in range(self.N)]
        df_raw = pd.DataFrame(data, columns=node_cols)
        self.target = node_cols[0]  # Use first node as target for 'S' mode

        # Train/val/test splits
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scaling
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_scaled = self.scaler.transform(df_data.values)
        else:
            data_scaled = df_data.values

        # --- TIME FEATURES ---
        if self.timeenc == 0:
            # Option 1: Calendar features
            dates = pd.date_range(start='2000-01-01', periods=len(df_raw), freq=self.freq)
            df_stamp = pd.DataFrame({'date': dates})
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour if self.freq[0] == 'H' else 0
            data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
        elif self.timeenc == 1:
            dates = pd.date_range(start='2001-01-01', periods=len(df_raw), freq='7D')

            data_stamp = time_features(pd.to_datetime(dates.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        else:
            # Option 2: Cyclical features
            t = np.arange(len(df_raw))
            sin1 = np.sin(2 * np.pi * t / 24)
            cos1 = np.cos(2 * np.pi * t / 24)
            sin2 = np.sin(2 * np.pi * t / 7)
            cos2 = np.cos(2 * np.pi * t / 7)
            data_stamp = np.stack([sin1, cos1, sin2, cos2], axis=1)  # [T, 4]

        self.data_x = data_scaled[border1:border2]
        self.data_y = data_scaled[border1:border2] 
        self.data_stamp = data_stamp[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def compute_required_padded_shape(self, N, T, theta_deg):
        theta = np.deg2rad(theta_deg)
        N_pad = int(np.ceil(abs(N * np.cos(theta)) + abs(T * np.sin(theta))))
        T_pad = int(np.ceil(abs(T * np.cos(theta)) + abs(N * np.sin(theta))))
        return N_pad, max(T_pad, T)

    def pad_to_center(self, arr, target_shape, pad_mode='wrap'):
        pad_N = max(target_shape[0] - arr.shape[0], 0)
        # pad_T = max(target_shape[1] - arr.shape[1], 0)
        pad_top = pad_N // 2
        pad_bottom = pad_N - pad_top
        # pad_left = pad_T // 2
        # pad_right = pad_T - pad_left
        padded = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode=pad_mode)
        return padded

    def crop_center(self, arr, N, T):
        N_pad, T_pad = arr.shape
        start_row = (N_pad - N) // 2
        start_col = (T_pad - T) // 2
        return arr[start_row:start_row+N, start_col:start_col+T]

    def generate_base_data(self, N, T, dt= np.pi / 45, noise_strength=0.1):
        # Generate temporal signals: shape [N, T]
        a = np.random.uniform(0.5, 1.5, N)
        b = np.random.uniform(-1, 1, N)
        w = np.random.uniform(2, 5, N)  # Different freq for each node
        s = np.random.uniform(0, 2 * np.pi, N) # Phase shift for each node
        t = dt * np.arange(1, T+1)

        data = np.zeros((N, T))
        for i in range(N):
            data[i, :] = a[i] * np.sin(w[i] * t + s[i] ) + b[i] + np.random.randn(T) * noise_strength
        
        return data

    def generate(self,
        N, T,     # desired safe region (final output shape)
        theta_deg,       
        pad_mode='wrap',
        dt = np.pi / 45,
        noise_strength=0.1,
    ):
        N_pad, T_pad = self.compute_required_padded_shape(N, T, theta_deg)

        if pad_mode == 'manual':
            data = self.generate_base_data(N_pad, T_pad, dt, noise_strength)
        else: # Can be 'wrap', 'edge', 'reflect', etc.
            data = self.generate_base_data(N, T_pad, dt, noise_strength)
            data = self.pad_to_center(data, (N_pad, T_pad), pad_mode=pad_mode)
        
        tensor = torch.from_numpy(data).unsqueeze(0).float()  # [1, N_pad, T_pad]
        rotated = TF.rotate(tensor, angle=theta_deg, interpolation=TF.InterpolationMode.BILINEAR, expand=False)
        rotated_np = rotated.squeeze(0).numpy()

        original = self.crop_center(data, N, T)
        result = self.crop_center(rotated_np, N, T)

        return result, original

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
