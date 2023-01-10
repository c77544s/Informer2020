import numpy as np
import main_informer
from exp.exp_informer import Exp_Informer
import matplotlib.pyplot as plt
import pandas as pd


args = main_informer.get_args()
setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                     args.data,
                                                                                                     args.features,
                                                                                                     args.seq_len,
                                                                                                     args.label_len,
                                                                                                     args.pred_len,
                                                                                                     args.d_model,
                                                                                                     args.n_heads,
                                                                                                     args.e_layers,
                                                                                                     args.d_layers,
                                                                                                     args.d_ff,
                                                                                                     args.attn,
                                                                                                     args.factor,
                                                                                                     args.embed,
                                                                                                     args.distil,
                                                                                                     args.mix,
                                                                                                     args.des,
                                                                                                     0)
exp = Exp_Informer(args)

# pred_data = pd.read_csv("data/CPM/cpm.csv")
cpm_data = pd.read_csv("data/CPM/cpm.csv")
# 循环5次
for i in range(5):
    # 获取随机数范围在0-1000之间的1个数
    random_num = np.random.randint(0, 900)
    start = random_num * 1
    end = start + args.seq_len
    pred_data = cpm_data[start:end]
    true_data = cpm_data[end:end + args.pred_len]
    # 获取true_data的序号
    true_data_index = np.array(true_data.index)
    true_data1 = np.array(true_data['recordValue'])

    predicts = exp.predict_true(pred_data, setting, True)


    # Plot data
    plt.rcParams["figure.figsize"] = (15, 5)  # 调整生成的图表最大尺寸
    plt.title('cpm Load')
    plt.ylabel('recordValue')
    plt.xlabel('time')
    plt.grid(True)  # 显示网格线
    plt.autoscale(axis='x', tight=True)  # 调整x轴的范围
    plt.plot(true_data_index, true_data1, label='true')
    plt.plot(true_data_index, predicts[0,:, 0], label='predict')
    # for i in range(real_prediction.shape[1]):
        # plt.plot(true[i, :, 0], label='true')
    # plt.plot(np.array(range(0, real_prediction.shape[1])), np.array(real_prediction[0][:,0]), label='real_prediction')
        # plt.plot(pred[i, :, 0], label='pred')
    plt.legend()
    plt.show()
