import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device), torch.zeros(1, 1, self.hidden_layer_size).to(device))
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# 训练模型
def train_model(train_seq, model, loss_function, optimizer, epochs=150):
    for i in range(epochs):
        for seq, labels in train_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq.cuda())
            single_loss = loss_function(y_pred, labels.cuda())
            single_loss.backward()
            optimizer.step()
        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# 测试模型
def test_model(test_inputs, model):
    preds = len(test_inputs) * [None]
    for i in range(len(test_inputs)):
        seq = torch.FloatTensor(test_inputs[i]).cuda()
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
            preds[i] = model(seq).item()
    return preds



if __name__ == '__main__':
    # Load data
    all_data = pd.read_csv("data/CPM/cpm.csv")

    # Plot data
    plt.rcParams["figure.figsize"] = (15, 5)  # 调整生成的图表最大尺寸
    plt.title('MS Load')
    plt.ylabel('load percentage')
    plt.xlabel('time')
    plt.grid(True)  # 显示网格线
    plt.autoscale(axis='x', tight=True)  # 调整x轴的范围

    xpoints = np.array(all_data['date'].values)
    ypoints = np.array(all_data['recordValue'].values)
    plt.plot(xpoints, ypoints, label='load')
    # plt.show()

    # 测试数据集大小
    test_data_size = 48
    # 切分训练集和测试集
    train_data = ypoints[:-test_data_size]
    test_data = ypoints[-test_data_size:]

    # 使用最小/最大标度器对我们的数据进行标准化处理，最小值和最大值分别为-1和1。
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # reshape(-1, 1)将数据转换为列向量 fit_transform()将数据转换为标准化数据
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    # 将我们的数据集转换成张量，因为PyTorch模型是使用张量进行训练的
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    # 训练时的输入序列长度设置为12
    train_window = 48

    # 生成用来训练的列表和相关的标签
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    model = LSTM().cuda()
    # 定义损失函数和优化器，这里使用的是均方误差损失函数和Adam优化器
    loss_function = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(train_inout_seq, model, loss_function, optimizer, epochs=50)

    fut_pred = 48

    # 从训练集中筛选出最后12个值，这12个项目将被用来对测试集的第一个项目进行预测，然后，预测值将被追加到test_inputs列表中，以便它可以被用来预测下一个值，以此类推。
    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
    print(test_inputs)

    # 截取预测出来的12个实际的预测值
    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
    print(actual_predictions)

    predict_x = xpoints[-train_window:]
    plt.plot(predict_x, actual_predictions, label='predict')
    plt.plot(predict_x, test_data, label='true')
    plt.legend()
    plt.show()