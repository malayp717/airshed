import numpy as np
import pandas as pd
from datetime import datetime
import os
import yaml

curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{curr_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def process_df(df_fp, start_date, end_date):
    df = pd.read_csv(df_fp)
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')
    df = df[(df['timestamp'] >= datetime(*start_date)) & (df['timestamp'] < datetime(*end_date))]

    df['is_weekend'] = df['timestamp'].apply(lambda x : 1 if x.weekday() >= 5 else 0)
    df['hour_of_day_cos'] = df['timestamp'].apply(lambda x : np.cos(2.0 * np.pi * x.hour / 24.0))
    df['hour_of_day_sin'] = df['timestamp'].apply(lambda x : np.sin(2.0 * np.pi * x.hour / 24.0))

    df['pm25'] = (df['pm25'] - df['pm25'].mean()) / df['pm25'].std()
    df['ventilation_coeff'] = (df['ventilation_coeff'] - df['ventilation_coeff'].mean()) / df['ventilation_coeff'].std()

    return df