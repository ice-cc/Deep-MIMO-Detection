# 生成数据, 这里设定 K = 4 ; N = K = 4
# Hty:shape(k,1)  HtH:shape(k,k)  x:shape(k,1)

import numpy as np

DataSet_x = []
DataSet_F1 = []
DataSet_F2 = []
Sigma2 = 1.0
SNR_min_dB = 0.0
SNR_max_dB = 20.0
# 生成的数据维度 K，N,可修改
K = 4
N = K
number = 1  # ！！！ 下边文件存放路径要改。。生成的数据个数
# 初始化数据集，x, H.T * y, H.T * H 分别存放于三个list：DataSet_x， DataSet_F1， DataSet_F2
for _ in range(0, number):
    SNR_dB = SNR_min_dB + (SNR_max_dB - SNR_min_dB) * np.random.rand()
    # rand 生成均匀分布的伪随机数。分布在（0~1）之间
    # SNR_dB = (SNR_min_dB + SNR_max_dB) / 2
    SNR = 10.0 ** (SNR_dB / 10.0)
    H = ((0.5 * float(SNR) / float(K)) ** (1 / 2.0)) * np.random.randn(N, K)
    x = 2 * np.round(np.random.rand(K, 1)) - 1  # x为正负1
    h_prime = np.transpose(H)  # H.T
    noise = (Sigma2 / 2.0) ** (1 / 2.0) * np.random.randn(N, 1)  # 均值为0,方差1...1/100~1
    y = np.matmul(H, x) + noise
    DataSet_x.append(x)
    DataSet_F1.append(np.matmul(h_prime, y))
    DataSet_F2.append(np.matmul(h_prime, H))


# 这里下边的两行，分别是生成train数据和val数据，用哪个就注掉另外一行。这个val是随机SNR生成的，只用于训练的时候评估
file = open('../../data/x.txt', "a")  # shape(k, 1)
# file = open('../../data/val_x.txt', "a")  # shape(k, 1)
for x in DataSet_x:
    for i in range(K):
        file.write(str(x[i][0]) + ' ')
    file.write('\n')
file.close()

file = open('../../data/f1_Hty.txt', "a")  # shape(k, 1)
# file = open('../../data/val_f1_Hty.txt', "a")  # shape(k, 1)
for x in DataSet_F1:
    for i in range(K):
        file.write(str(x[i][0]) + ' ')
    file.write('\n')
file.close()

file = open('../../data/f2_HtH.txt', "a")  # shape(k, k)
# file = open('../../data/val_f2_HtH.txt', "a")  # shape(k, k)
for x in DataSet_F2:
    for i in range(K):
        for j in range(K):
            file.write(str(x[i][j]) + ' ')
    file.write('\n')
file.close()
