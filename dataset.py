import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from datetime import datetime, timedelta

class TemporalDataset(Dataset):
    def __init__(self, df, columns_to_drop, label_columns, locs):
        super().__init__()
        self.df = df[df.apply(lambda row : (row['longitude'], row['latitude']) in locs, axis=1)]
        self.columns = [x for x in self.df.columns if x not in columns_to_drop]
        self.label_columns = label_columns

        self.data = self._process_data()

    def _process_data(self):
        data = []
        columns_to_keep = [x for x in self.columns if x not in {'timestamp', 'longitude', 'latitude'}]
        # print(f'Label: {self.label_columns}\n Keep: {columns_to_keep}')

        df_locs = self.df.groupby(['longitude', 'latitude'])
        for loc, group in df_locs:
            group = group.sort_values(by='timestamp')
            X, y = group[columns_to_keep].to_numpy(), group[self.label_columns].to_numpy()
            X, y = X[:-1], y[1:]
            data.append({'loc': loc, 'data': X, 'label': y})

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        row = self.data[index]
        loc, X, y = row['loc'], np.float32(row['data']), np.float32(row['label'])

        return torch.Tensor(loc), torch.Tensor(X), torch.Tensor(y)