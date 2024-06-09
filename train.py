import time
import yaml
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TemporalDataset
from LSTM import LSTM
from utils import process_df, generate_embeddings
import pickle
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{curr_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ----------------- Config Params Load ----------------- #
data_dir = config['filepath']['data_dir']
bihar_loc_fp = data_dir + config['filepath']['loc_fp']

num_epochs = config['train']['num_epochs']
lr = config['train']['lr']
batch_size = config['train']['batch_size']
# ----------------- Config Params End ----------------- #

# ----------------- Variables Declaration Start ----------------- #
'''
    0: Include only pm25 (or exclude ventilation_coeff)
    1: Include only ventilation_coeff (or exclude pm25)
    2: Include both (or exclude none)
'''
settings = {
                0: {
                    'drop': ['ventilation_coeff'], 
                    'label': ['pm25']
                },
                1: {
                    'drop': ['pm25'],
                    'label': ['ventilation_coeff']
                },
                2: {
                    'drop': [],
                    'label': ['pm25', 'ventilation_coeff']
                }
            }
seasons = ['JJAS', 'ON', 'DJF', 'MA']
df_fp = f'{data_dir}/airshed/bihar_june_apr_imputed.csv'
# ----------------- Variables Declaration End ----------------- #

def train(model, loader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad()

        _, X, y = data
        y = y.to(device)
        preds, _ = model(X)

        y, preds = torch.squeeze(y), torch.squeeze(preds)
        loss = criterion(y, preds)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= (batch_idx+1)
    return train_loss


def test(model, loader):
    model.eval()
    test_loss = 0

    for batch_idx, data in enumerate(loader):
        _, X, y = data
        y = y.to(device)
        preds, _ = model(X)

        y, preds = torch.squeeze(y), torch.squeeze(preds)
        loss = criterion(y, preds)
        test_loss += loss.item()

    test_loss /= (batch_idx+1)
    return test_loss

if __name__ == '__main__':

    bihar_locs = pd.read_csv(bihar_loc_fp, header=None, delimiter='|')

    '''
        bihar_locs: list (longitude, latitude)
        rain-val-test split: [40, 10, 50]
    '''
    bihar_locs = [(x, y) for x, y in zip(bihar_locs.iloc[:, -2], bihar_locs.iloc[:, -1])]
    train_len, val_len = (2*len(bihar_locs))//5, len(bihar_locs)//10

    train_bihar_locs, val_bihar_locs, test_bihar_locs = bihar_locs[:train_len], bihar_locs[train_len: train_len+val_len],\
                                                            bihar_locs[train_len+val_len:]

    HID_DIM = [64, 128, 256]

    for setting, columns in settings.items():
        for season in seasons:
            start_date, end_date = config[season]['start_date'], config[season]['end_date']

            df = process_df(df_fp, start_date, end_date)
            num_days = (datetime(*end_date) - datetime(*start_date)).days
            assert df.shape[0] == 511 * 24 * num_days

            columns_to_drop, label_columns = columns['drop'], columns['label']

            train_dataset = TemporalDataset(df, columns_to_drop, label_columns, set(train_bihar_locs))
            val_dataset = TemporalDataset(df, columns_to_drop, label_columns, set(val_bihar_locs))
            test_dataset = TemporalDataset(df, columns_to_drop, label_columns, set(test_bihar_locs))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            _, X, y = train_dataset[0]
            input_dim, out_dim = X.size(-1), y.size(-1)

            for hid_dim in HID_DIM:

                print(f"---------\t Season Info: {season} \t start_date={start_date} \t end_date={end_date} \t label={columns['label']}\
                      hid_dim={hid_dim}\t---------")

                model = LSTM(input_dim, hid_dim, out_dim, device)
                model.to(device)

                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

                start_time = time.time()

                for epoch in range(num_epochs):
                    train_loss = train(model, train_loader, optimizer)
                    val_loss = test(model, val_loader)

                    if (epoch+1) % (num_epochs//5) == 0:
                        print(f'Epoch: {epoch+1} | {num_epochs} \t Train Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f}\
                          \t Time taken: {(time.time()-start_time)/60:.4f} mins')
                        start_time = time.time()

                train_loss = test(model, train_loader)
                val_loss = test(model, val_loader)
                test_loss = test(model, test_loader)
                    
                print(f"---------\t Training Completed: \t Train Loss: {train_loss:.4f} \t Val Loss: {val_loss:.4f} \t\
                      Test Loss: {test_loss:.4f} \t---------\n")

                locs, embeddings = [], []

                loc, emb = generate_embeddings(model, train_loader)
                locs.extend(loc)
                embeddings.extend(emb)

                loc, emb = generate_embeddings(model, val_loader)
                locs.extend(loc)
                embeddings.extend(emb)

                loc, emb = generate_embeddings(model, test_loader)
                locs.extend(loc)
                embeddings.extend(emb)

                locs, embeddings = np.array(locs), np.array(embeddings)

                loc_embeddings = {}
                for loc, emb in zip(locs, embeddings):
                    loc_embeddings[(loc[0], loc[1])] = emb

                with open(f'{data_dir}/airshed/bihar_{season}_{setting}_{hid_dim}.pkl', 'wb') as f:
                    pickle.dump(loc_embeddings, f)