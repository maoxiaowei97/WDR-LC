from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

def get_data(df):
    def get_cross_info(x):
        if x == ['nan']:
            return [[], [], [], []]
        id = []
        time = []
        start_id = []
        end_id = []
        for xx in x:
            id.append(str(xx.split(':')[0]))
            time.append(float(xx.split(':')[1]))
            start_id.append(int(xx.split(':')[0].split('_')[0]))
            end_id.append(int(xx.split(':')[0].split('_')[1]))
        return [id, time, start_id, end_id]

    def get_link_info(x):
        if x == ['nan']:
            return [[], [], [], [], []]
        id = []
        time = []
        ratio = []
        cur_state = []
        arri_state = []
        for xx in x:
            id.append(int(xx.split(':')[0]))
            time.append(float(xx.split(':')[1].split(',')[0]))
            ratio.append(float(xx.split(':')[1].split(',')[1]))
            cur_state.append(int(xx.split(':')[1].split(',')[2]))
            arri_state.append(int(xx.split(':')[1].split(',')[3]))
        return [id, time, ratio, cur_state, arri_state]

    # head
    df['order_id'] = df['head'].apply(lambda x: x.split(' ')[0])
    df['order_id'] = df['order_id'].astype(str)
    df['label'] = df['head'].apply(lambda x: x.split(' ')[1])
    df['label'] = df['label'].astype(float)
    df['distance'] = df['head'].apply(lambda x: x.split(' ')[2])
    df['distance'] = df['distance'].astype(float)
    df['simple_eta'] = df['head'].apply(lambda x: x.split(' ')[3])
    df['simple_eta'] = df['simple_eta'].astype(float)
    df['driver_id'] = df['head'].apply(lambda x: x.split(' ')[4])
    df['driver_id'] = df['driver_id'].astype(int)
    df['slice_id'] = df['head'].apply(lambda x: x.split(' ')[5])
    df['slice_id'] = df['slice_id'].astype(int)
    del df['head']
    # link
    df['link'] = df['link'].apply(lambda x: x.split(' '))
    df['link_num'] = df['link'].apply(lambda x: len(x))
    df['link_info'] = df['link'].apply(lambda x: get_link_info(x))
    df['link_id_lst'] = df['link_info'].apply(lambda x: x[0])
    df['link_time_lst'] = df['link_info'].apply(lambda x: x[1])
    del df['link']
    del df['link_info']

    # cross
    df['cross'] = df['cross'].apply(lambda x: x.split(' '))
    df['cross_info'] = df['cross'].apply(lambda x: get_cross_info(x))
    df['cross_id_lst'] = df['cross_info'].apply(lambda x: x[0])
    df['cross_time_lst'] = df['cross_info'].apply(lambda x: x[1])
    df['cross_start_lst'] = df['cross_info'].apply(lambda x: x[2])
    df['cross_end_lst'] = df['cross_info'].apply(lambda x: x[3])
    df['cross_num'] = df['cross_id_lst'].apply(lambda x: len(x))
    del df['cross']
    del df['cross_info']

    return df


def get_new_seq(x, df, cross_id_dic):
    df1 = df[df['order_id'] == x['order_id']]
    cross_id_lst = df1['cross_id_lst'].values.tolist()[0]
    cross_time_lst = df1['cross_time_lst'].values.tolist()[0]
    cross_start_lst = df1['cross_start_lst'].values.tolist()[0]
    cross_end_lst = df1['cross_end_lst'].values.tolist()[0]

    link_id = x['link_id_lst']
    link_len = len(link_id)
    link_time = x['link_time_lst']

    pos = -1
    i = -1
    global max_flag
    while True:     # 对cross序列遍历
        pos_i = pos
        i += 1
        if i == len(cross_id_lst): break

        cross_id = cross_id_lst[i]
        cross_time = cross_time_lst[i]
        cross_end = cross_end_lst[i]
        if cross_id not in cross_id_dic.keys():
            max_flag += 1
            cross_id_dic[cross_id] = max_flag
            cross_id = max_flag
        else:
            cross_id = cross_id_dic[cross_id]

        while True:     # 对link序列遍历
            pos_i += 1
            if cross_end == link_id[pos_i]:
                pos = pos_i
                link_id.insert(pos, cross_id)
                link_time.insert(pos, cross_time)
                break
            if pos_i == len(link_id):  # 当发现有cross找不到位置
                print('not found')
                pos += 1
                link_id.insert(pos, cross_id)
                link_time.insert(pos, cross_time)
                break

    if len(link_id) != link_len + len(cross_id_lst):
        print('error')

    # return link_id, link_time


def new_id_time(train_dir, test_path, mode='link', is_test=False):
    feature_path = ws + '/data/feature/id_time/'
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    df_path = ws + '/data/feature/df_data/'
    if not os.path.exists(df_path):
        os.makedirs(df_path)

    train_files = os.listdir(train_dir)
    train_files.sort()

    cross_id_dic = {}
    # 训练集
    for file in tqdm(train_files):
        k = int(file.split('.')[0][-2:])
        # 读取原始数据
        file_path = train_dir + '/' + file
        df = pd.read_csv(file_path, sep=';;', names=['head', 'link', 'cross'], engine='python')
        df['cross'] = df['cross'].astype(str)
        if is_test:
            df = df.iloc[:200, :]

        df = get_data(df)
        df['link_time_sum'] = df['link_time_lst'].apply(lambda x: sum(x))
        df['cross_time_sum'] = df['cross_time_lst'].apply(lambda x: sum(x))
        df[['order_id', 'label', 'distance', 'simple_eta', 'driver_id', 'slice_id', 'link_num', 'link_time_sum',
            'cross_time_sum', 'cross_num']].to_csv(df_path + 'df_%d.csv' % k, index=False)

        feature = df[['order_id', 'link_id_lst', 'link_time_lst']]
        feature.apply(lambda x: get_new_seq(x, df, cross_id_dic), axis=1)
        feature = feature.values
        np.save(feature_path + 'id_time_%d.npy' % k, feature)

        if is_test:
            break

    # 测试集
    df = pd.read_csv(test_path, sep=';;', names=['head', 'link', 'cross'], engine='python')
    df['cross'] = df['cross'].astype(str)
    if is_test:
        df = df.iloc[:200, :]

    df = get_data(df)
    df['link_time_sum'] = df['link_time_lst'].apply(lambda x: sum(x))
    df['cross_time_sum'] = df['cross_time_lst'].apply(lambda x: sum(x))
    df[['order_id', 'label', 'distance', 'simple_eta', 'driver_id', 'slice_id', 'link_num', 'link_time_sum',
        'cross_time_sum', 'cross_num']].to_csv(df_path + 'df_test.csv', index=False)

    feature = df[['order_id', 'link_id_lst', 'link_time_lst']]
    feature.apply(lambda x: get_new_seq(x, df, cross_id_dic), axis=1)
    feature = feature.values
    np.save(feature_path + 'id_time_test.npy', feature)

    fout = ws + '/data/feature/id_time/cross_dict.pkl'
    f = open(fout, "wb")
    pickle.dump(cross_id_dic, f)
    f.close()


if __name__ == "__main__":
    ws = 'xxx'
    train_dir = ws + '/data/raw_data/train'
    test_path = ws + '/data/raw_data/20200901_test.txt'
    max_flag = 639877
    new_id_time(train_dir, test_path, is_test=True)