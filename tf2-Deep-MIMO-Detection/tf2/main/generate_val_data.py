# 生成数据, 这里设定 K = 4 ; N = K = 4
# Hty:shape(k,1)  HtH:shape(k,k)  x:shape(k,1)

import numpy as np

Sigma2 = 1.0
SNR_min_dB = 0.0
SNR_max_dB = 20.0
SNR_dB_range = np.arange(0, 22, 2)  # [ 0  2  4  6  8 10 12 14 16 18 20] shape:(11,)
# 生成的数据维度 K，N
K = 4
N = K
number = 1  # 生成的数据个数

for SNR_dB in SNR_dB_range:
    DataSet_x = []
    DataSet_F1 = []
    DataSet_F2 = []
    # 初始化数据集，x, H.T * y, H.T * H 分别存放于三个list：DataSet_x， DataSet_F1， DataSet_F2
    for _ in range(0, number):

        SNR = 10.0 ** (SNR_dB / 10.0)
        H = ((0.5 * float(SNR) / float(K)) ** (1 / 2.0)) * np.random.randn(N, K)
        x = 2 * np.round(np.random.rand(K, 1)) - 1
        h_prime = np.transpose(H)  # H.T
        noise = (Sigma2 / 2.0) ** (1 / 2.0) * np.random.randn(N, 1)
        y = np.matmul(H, x) + noise
        DataSet_x.append(x)
        DataSet_F1.append(np.matmul(h_prime, y))
        DataSet_F2.append(np.matmul(h_prime, H))

    file_name_x = '../../data/val/SNR={}dB_x.txt'.format(SNR_dB)
    file = open(file_name_x, "a")
    for x in DataSet_x:
        for i in range(K):
            file.write(str(x[i][0]) + ' ')
        file.write('\n')
    file.close()

    file_name_f1_Hty = '../../data/val/SNR={}dB_f1_Hty.txt'.format(SNR_dB)
    file = open(file_name_f1_Hty, "a")  # shape(k, 1)
    for x in DataSet_F1:
        for i in range(K):
            file.write(str(x[i][0]) + ' ')
        file.write('\n')
    file.close()

    file_name_f2_HtH = '../../data/val/SNR={}dB_f2_HtH.txt'.format(SNR_dB)
    file = open(file_name_f2_HtH, "a")  # shape(k, k)
    for x in DataSet_F2:
        for i in range(K):
            for j in range(K):
                file.write(str(x[i][j]) + ' ')
        file.write('\n')
    file.close()
