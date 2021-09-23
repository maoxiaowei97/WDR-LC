import numpy as np
import torch
import pandas as pd
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import tqdm
import argparse
import nni
import os

# layer部分
class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim):
        super(FeaturesLinear, self).__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)  # fc: Embedding:(610 + 193609, 1) 做一维特征的嵌入表示
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))  # Tensor: 1

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x:tensor([[554, 2320], [304, 3993]])
        # x: Tensor: 2048, 每个Tensor维度为2, x.new_tensor(self.offsets).unsqueeze(0): tensor([[0, 610]])
        # x: Tensor: [2048, 2]
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)  # embeddingL Embedding:(610+193609, 16)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # embedding weight的初始化通过均匀分布的采用得到

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class FactorizationMachine(torch.nn.Module):

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Regressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, wide, deep, recurrent):
        fuse = self.linear_wide(wide) + self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)


# model部分
class Wide_Deep_RecurrentModel(torch.nn.Module):

    def __init__(self):
        super(Wide_Deep_RecurrentModel, self).__init__()

        wide_field_dims = np.array([80887, 288, 1, 1, 1, 1, 1, 1])
        wide_embed_dim = 20
        wide_mlp_dims = (256,)

        deep_field_dims = np.array([80887, 288])
        deep_embed_dim = 20
        deep_real_dim = 6
        deep_category_dim = 2
        deep_mlp_input_dim = deep_embed_dim * deep_category_dim + deep_real_dim
        deep_mlp_dims = (256,)

        id_dims = 684192
        id_embed_dim = 20
        slice_dims = 289
        slice_embed_dim = 20
        all_real_dim = 1
        mlp_out_dim = 256
        lstm_hidden_size = 256

        reg_input_dim = 256
        reg_output_dim = 256

        self.wide_embedding = FeaturesEmbedding(sum(wide_field_dims), wide_embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(),
            torch.nn.BatchNorm1d(wide_embed_dim),
        )
        self.wide_mlp = MultiLayerPerceptron(wide_embed_dim, wide_mlp_dims)  # 不batchnorm

        self.deep_embedding = FeaturesEmbedding(sum(deep_field_dims), deep_embed_dim)
        self.deep_mlp = MultiLayerPerceptron(deep_mlp_input_dim, deep_mlp_dims)

        self.slice_embedding = nn.Embedding(slice_dims, slice_embed_dim)
        self.id_embedding = nn.Embedding(id_dims, id_embed_dim)
        self.all_mlp = nn.Sequential(
            nn.Linear(id_embed_dim + slice_embed_dim + all_real_dim , mlp_out_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=mlp_out_dim, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)
        self.regressor = Regressor(reg_input_dim, reg_output_dim)

    def forward(self, wide_index, wide_value, deep_category, deep_real,
                all_id, all_num, all_slice, all_real):
        # wide
        wide_embedding = self.wide_embedding(wide_index)  # 对所有item特征做embedding,对连续特征做一维embedding
        cross_term = self.fm(wide_embedding * wide_value.unsqueeze(2))  # wide_value前两列为1，之后为dense feature数值
        wide_output = self.wide_mlp(cross_term)

        # deep part
        batch_size = deep_real.shape[0]
        deep_embedding = self.deep_embedding(deep_category).view(batch_size, -1)
        deep_input = torch.cat([deep_embedding, deep_real], dim=1)
        deep_output = self.deep_mlp(deep_input)

        # recurrent part
        all_id_embedding = self.id_embedding(all_id)
        all_slice_embedding = self.slice_embedding(all_slice)
        all_real = all_real.unsqueeze(2)
        all_input = torch.cat([all_id_embedding, all_slice_embedding, all_real], dim=2)
        recurrent_input = self.all_mlp(all_input)
        packed_all_input = pack_padded_sequence(recurrent_input, all_num.cpu(), enforce_sorted=False, batch_first=True)
        out, (hn, cn) = self.lstm(packed_all_input)
        hn = hn.squeeze()
        hn = hn[1, :, :]

        # regressor
        result = self.regressor(wide_output, deep_output, hn)

        return result.squeeze(1)


# 构建Dataset
class GiscupDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_data, padding_size, device):
        all_num = []
        all_id = []
        all_time = []
        all_slice = []

        slices = df['slice_id'].values.tolist()
        for i in tqdm.tqdm(range(len(seq_data))):
            length = len(seq_data[i][1])
            all_num.append(length)

            ids = seq_data[i][1] + [-1] * (padding_size - length)
            all_id.append(ids)

            time = seq_data[i][2] + [-1] * (padding_size - length)
            all_time.append(time)

            all_s = [slices[i]] * length + [-1] * (padding_size - length)
            all_slice.append(all_s)

            # all_s = [slices[i]] * padding_size
            # all_slice.append(all_s)

        self.all_num = torch.tensor(all_num, dtype=torch.int)
        self.all_id = torch.tensor(all_id, dtype=torch.long) + 1
        self.all_slice = torch.tensor(all_slice, dtype=torch.long) + 1
        self.all_real = torch.tensor(all_time, dtype=torch.float)

        wide_deep_raw = torch.tensor(df[['driver_id', 'slice_id', 'distance', 'simple_eta',
                                         'link_num', 'link_time_sum', 'cross_time_sum', 'cross_num']].values)

        self.deep_category = wide_deep_raw[:, :2] + torch.tensor([0, 80887])
        self.deep_category = self.deep_category.long()
        self.deep_real = wide_deep_raw[:, 2:].float()

        self.wide_index = wide_deep_raw.clone()  # [256, 8]
        self.wide_index[:, 2:] = 0
        self.wide_index += torch.tensor([0, 80887, 80887 + 288, 80887 + 288 + 1, 80887 + 288 + 1 + 1,
                                         80887 + 288 + 1 + 1 + 1, 80887 + 288 + 1 + 1 + 1 + 1,
                                         80887 + 288 + 1 + 1 + 1 + 1 + 1])
        self.wide_index = self.wide_index.long()
        self.wide_value = wide_deep_raw.float()
        self.wide_value[:, :2] = 1.0

        self.targets = torch.tensor(df['label'].values)

    def __getitem__(self, index):
        return self.wide_index[index], \
               self.wide_value[index], \
               self.deep_category[index], \
               self.deep_real[index], \
               self.all_id[index], \
               self.all_num[index], \
               self.all_slice[index], \
               self.all_real[index], \
               self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


def train(model, optimizer, data_loader, loss):
    model.train()
    train_loss = []
    # print('training...')
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, data in enumerate(tk0):
        wide_index, wide_value, deep_category, deep_real, \
            all_id, all_num, all_slice, all_real, targets = [d.to(device) for d in data]
        y = model(wide_index, wide_value, deep_category, deep_real,
                  all_id, all_num, all_slice, all_real)
        l = loss(y, targets.float(), deep_real[:, 1])
        model.zero_grad()
        l.backward()
        optimizer.step()
        train_loss.append(l.item())
    print('train loss of one day in one epoch: ' + str((sum(train_loss) / len(train_loss))))


def train_process(batch_size, df, seq_data, model, optimizer, loss):
    dataset = GiscupDataset(df, seq_data, 935, device)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    train(model, optimizer, data_loader, loss)


def test(model, data_loader, epoch, day, val_eval):
    model.eval()
    predicts = []
    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, data in enumerate(tk0):
            wide_index, wide_value, deep_category, deep_real, \
                all_id, all_num, all_slice, all_real, targets = [d.to(device) for d in data]
            y = model(wide_index, wide_value, deep_category, deep_real,
                      all_id, all_num, all_slice, all_real)
            predicts.append(y.tolist())

    predicts = np.array(predicts, dtype=object)
    np.save(output_path + f'/{model_name}_epoch{epoch}_day{day}', predicts)
    print('predict work done')


def test_process(batch_size, file_name, seq_data_name, model, epoch, day, val_eval):
    df = pd.read_csv(file_name)  # file name 为取出需要信息后的csv
    seq_data = np.load(seq_data_name, allow_pickle=True)
    if is_test:
        df = df[:500]
        seq_data = seq_data[:500]
    dataset = GiscupDataset(df, seq_data, 935, device)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    test(model, data_loader, epoch, day, val_eval)


def val(model, val_data_loader):
    model.eval()
    predicts = []
    label = []
    with torch.no_grad():
        tk0 = tqdm.tqdm(val_data_loader, smoothing=0, mininterval=1.0)
        for i, data in enumerate(tk0):
            wide_index, wide_value, deep_category, deep_real, \
                all_id, all_num, all_slice, all_real, targets = [d.to(device) for d in data]
            y = model(wide_index, wide_value, deep_category, deep_real,
                      all_id, all_num, all_slice, all_real)
            predicts += y.tolist()
            label += targets.tolist()
    predicts = np.array(predicts)
    label = np.array(label)

    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    def mape_(label, predicts):
        return (abs(predicts - label) / label).mean()

    val_mape = mape_(label, predicts)
    val_mse = mse(label, predicts)
    val_mae = mae(label, predicts)
    return val_mape, val_mse, val_mae


def val_process(batch_size, df, seq_data, model):
    val_dataset = GiscupDataset(df, seq_data, 935, device)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    mape, mse, mae = val(model, val_data_loader)
    print('Eval at validation set:')
    print('MAPE:%.3f\tMSE:%.2f\tMAE:%.2f' % (mape * 100, mse, mae))
    return {'mape': mape, 'mse': mse, 'mae': mae}


def mape(y_hat, y, length):
    length = length.cpu().numpy().tolist()
    weight = [1 if i > 300 else 1.222 for i in length]
    weight = torch.tensor(weight, device=device)
    l = torch.abs(y_hat - y) / y
    l *= weight
    return l.mean()


def main(ws, epochs, is_test=True):
    batch_size = 256
    learning_rate = 1e-3

    # nni
    parser = argparse.ArgumentParser()
    parser.add_argument('--hhh', type=int, default=0, metavar='N')
    args, _ = parser.parse_known_args()
    params = vars(args)
    tuner_params = nni.get_next_parameter()  # 这会获得一组搜索空间中的参数
    params.update(tuner_params)

    model = Wide_Deep_RecurrentModel().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_decay_step = 10 if not is_test else 1
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.9)
    loss = mape

    filepath = ws + '/data/feature'
    test_seq_name = filepath + '/id_time/id_time_test.npy'
    test_file_name = filepath + '/df_data/df_test.csv'

    # weather_path = ws + '/data/raw_data/weather.csv'
    # weather = get_weather(weather_path)

    # train process
    val_eval = {}
    for i in range(epochs):
        print('#' * 50)
        print('num epoch: ' + str(i))
        for j in range(1, 32):
            if j == 3 or (i >= 2 and j == 1) or (i >= 2 and j == 2):
                continue
            else:
                print('num epoch: ' + str(i) + '_day: ' + str(j))
                print('training...')
                # print(f'current learning rate: {optimizer.param_groups[0]["lr"]}')

                df = pd.read_csv(filepath + f'/df_data/df_{j}.csv')
                seq_data = np.load(filepath + f'/id_time/id_time_{j}.npy', allow_pickle=True)
                if is_test:
                    df = df[:500]
                    seq_data = seq_data[:500]
                train_process(batch_size, df, seq_data, model, optimizer, loss)
                StepLR.step()

            if is_test or (i == 2 and j == 31) :
                # 验证集计算
                print('validating...')
                df = pd.read_csv(filepath + '/df_data/df_val.csv')
                seq_data = np.load(filepath + '/id_time/id_time_val.npy', allow_pickle=True)
                if is_test:
                    df = df[:500]
                    seq_data = seq_data[:500]
                val_eval = val_process(batch_size, df, seq_data, model)
                nni.report_intermediate_result(val_eval['mape'])

                # 保存模型
                model_path = output_path + f'{model_name}_val-mape{(val_eval["mape"] * 100):.4f}' \
                                           f'mse{val_eval["mse"]:.2f}mae{val_eval["mae"]:.2f}_epoch{i}_day{j}.params'
                torch.save(model.state_dict(), model_path)

                # 输出测试集
                print('testing...')
                test_process(batch_size, test_file_name, test_seq_name, model, i, j, val_eval)

            if is_test and j == 2:
                return

    nni.report_final_result(val_eval['mape'])


if __name__ == "__main__":
    epochs = 3
    ws = 'xxx'
    model_name = 'model1'
    output_path = ws + f'/results/trained_model_{model_name}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    is_test = False
    main(ws, epochs, is_test)
