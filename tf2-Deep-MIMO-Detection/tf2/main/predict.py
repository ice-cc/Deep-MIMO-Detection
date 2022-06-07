import tensorflow as tf
import numpy as np
import get_model

k = 4
val_sample_total = 5000  # val样本总数大小
val_number = 5000  # 预测个数
BER = []

model = get_model.get_model()

# 在这里修改checkpoint_save_path可以加载其他的模型参数
checkpoint_save_path = "checkpoint-k=4-L=2k/model.ckpt"
model.load_weights(checkpoint_save_path)

# # # # # # # # # # # # # #  载入数据
SNR_dB_range = np.arange(0, 22, 2)  # [ 0  2  4  6  8 10 12 14 16 18 20] shape:(11,)
for SNR_dB in SNR_dB_range:
    file_name_f1 = '../../data/val/SNR={}dB_f1_Hty.txt'.format(SNR_dB)
    f1_data = get_model.read_data(file_name_f1, k)
    f1_data = np.array(f1_data).reshape((val_sample_total, k, 1))

    file_name_f2 = '../../data/val/SNR={}dB_f2_HtH.txt'.format(SNR_dB)
    f2_data = get_model.read_data(file_name_f2, k * k)
    f2_data = np.array(f2_data).reshape((val_sample_total, k, k))

    file_name_x = '../../data/val/SNR={}dB_x.txt'.format(SNR_dB)
    xk2_label = get_model.read_data(file_name_x, k)
    xk2_label = np.array(xk2_label).reshape((val_sample_total, k, 1))

    f1_pre = f1_data[0:val_number]
    f2_pre = f2_data[0:val_number]
    vk_pre = np.zeros(shape=(val_number, 2 * k, 1), dtype=np.float32)
    xk_pre = np.zeros(shape=(val_number, k, 1), dtype=np.float32)
    xk2_pre = xk2_label[0:val_number]

    # # # # # # # # # # # # # #  预测
    result = model.predict(
        ({'f1': f1_pre, 'vk': vk_pre, 'xk': xk_pre, 'f2': f2_pre})
    )

    # 处理模型的初始输出数据，round函数四舍五入
    # 比如将0.9999999或者1.000000001的输出变成1; -0.99999999999或者-1.000000001变成-1
    result1 = tf.math.round(result)

    # # 检查结果,对照着查看
    # print('---------------模型初始输出---------------------------')
    # print(result)
    # print('---------------round处理后的模型输出-------------------')
    # print(result1)
    # print('---------------预测与真实对比--------------------------')
    # print(result1 == xk2_pre)

    # 输出预测的BER
    print('------------------SNR =', SNR_dB, 'dB------------------------')
    BER.append(get_model.predict(result1, xk2_pre))

print('\nSNR_dB_range:', SNR_dB_range)
print('BER_range:', BER)

get_model.plot_SNR_BER(SNR_dB_range, BER)  # 画SNR-BER图

# SNR_dB_range: [ 0  2  4  6  8 10 12 14 16 18 20]
# BER_range: [0.4936, 0.4028, 0.30685, 0.2233, 0.1471, 0.09005, 0.05425, 0.034, 0.0245, 0.01835, 0.01345]
