import numpy as np
import pandas as pd
import os

def check(df, seq):
    for order_id in seq[:, 0]:
        if order_id not in df['order_id'].values:
            print('.')


def random_pick(df, seq, percent):
    df_val_day = df.sample(frac=percent)
    df_new = df[~ df['order_id'].isin(df_val_day['order_id'])]

    seq_df = pd.DataFrame(seq, columns=['order_id', 'id', 'time'])
    seq_val_day = seq[seq_df['order_id'].isin(df_val_day['order_id'])]
    seq_new = seq[~ seq_df['order_id'].isin(df_val_day['order_id'])]

    check(df_val_day, seq_val_day)
    return df_val_day, seq_val_day, df_new, seq_new


def sort_val(df, seq):
    seq_df = pd.DataFrame(seq, columns=['order_id', 'link_id', 'link_time'])
    df = df[['order_id']]
    df_sum = pd.merge(df, seq_df, how='left', on='order_id')
    seq_df = df_sum[['order_id', 'link_id', 'link_time']]
    return seq_df.values


def main(feature_path, out_path):
    val_dict = {}
    df_val = pd.DataFrame()
    seq_val = 0
    for i in range(1, 32):
        if i == 3:
            continue

        percent = 1 / 30

        df = pd.read_csv(feature_path + f'/df_data/df_{i}.csv', index_col=0)
        df['order_id'] = df['order_id'].astype(str)
        seq_data = np.load(feature_path + f'/id_time/id_time_{i}.npy', allow_pickle=True)

        df_val_day, seq_val_day, df_new, seq_new = random_pick(df, seq_data, percent)

        df_new.to_csv(out_path + f'/df_data/df_{i}.csv')
        np.save(out_path + f'/id_time/id_time_{i}.npy', seq_new)

        df_val = pd.concat([df_val, df_val_day])
        if i == 1:
            seq_val = seq_val_day
        else:
            seq_val = np.concatenate([seq_val, seq_val_day])

    df_val.to_csv(out_path + f'/df_data/df_val.csv')
    seq_val = sort_val(df_val, seq_val)
    np.save(out_path + f'/id_time/id_time_val.npy', seq_val)


if __name__ == '__main__':
    ws = 'xxx'
    feature_path = ws + '/data/feature/'
    out_path = ws + '/data/feature/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        os.makedirs(out_path + '/df_data')
        os.makedirs(out_path + '/id_time')
    main(feature_path, out_path)
