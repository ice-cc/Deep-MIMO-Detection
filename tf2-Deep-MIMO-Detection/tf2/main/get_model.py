# 把一些方法整在了一个文件里，便于调用。也可以改成类实现。

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

tf.keras.backend.clear_session()

k = 4
L = 2 * k  # 修改L后，训练时也需要修改其他东西，见train_model.py文件中的注释


# 从文件中读取数据，file_name：文件名；number：每行的数据个数
def read_data(file_name, number):
    x = []
    in_file = open(file_name, 'r')
    for line in in_file:
        x_temp = []
        data_txt = line.split(' ')
        for i in range(0, number):
            x_temp.append(float(data_txt[i]))
        x.append(x_temp)
    return x


# 画SNR-BER图
def plot_SNR_BER(SNR_dB_range, BER):
    SNR_dB_range_new = np.linspace(SNR_dB_range.min(), SNR_dB_range.max())
    BER_smooth = make_interp_spline(SNR_dB_range, BER)(SNR_dB_range_new)
    plt.plot(SNR_dB_range_new, BER_smooth, color='orange', label='test')
    plt.scatter(SNR_dB_range, BER, color='orange', marker='*')
    for x, y in zip(SNR_dB_range, BER):
        plt.text(x, y, '({}, {})'.format(x, y))
    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('SNR(dB)---BER')
    x = MultipleLocator(2)  # x轴每2一个刻度
    ax = plt.gca()
    ax.xaxis.set_major_locator(x)
    ax.set_yscale('log')
    plt.legend()
    plt.show()


# 自定义激活函数,fai_t(x) =relu(x + t) / abs(t) - 1 - relu(x - t) / abs(t)
def psi_activation(x, tt=0.5):
    t = tt * tf.ones_like(x)
    relu1 = tf.nn.relu((x + t))
    relu2 = tf.nn.relu((x - t))
    ab = tf.abs(t)
    out1 = tf.divide(relu1, ab)
    out2 = tf.divide(relu2, ab)
    one = tf.ones_like(out1, dtype=tf.float32)
    temp = tf.add(one, out2)
    return tf.subtract(out1, temp)


# 预测正确的个数
def predict(my_predict, label):
    cmp_two = (np.squeeze(my_predict) == np.squeeze(label))
    predict_right_bit = sum(sum(cmp_two))
    all_bit = (len(label) * k)
    predict_wrong_bit = all_bit - predict_right_bit
    BER = predict_wrong_bit / all_bit
    print('预测错误的码数:', predict_wrong_bit)
    print('误码率BER:', BER)
    return BER


def model_block(f1_input, vk_input, xk_input, f2_input):
    # 输入层  在参数里

    # 2层
    f2xk_feat = tf.matmul(f2_input, xk_input)

    # 3层
    concat_features = layers.concatenate(inputs=[f1_input, vk_input, xk_input, f2xk_feat], axis=-2)

    # 4层
    concat_features_flatten = layers.Flatten()(concat_features)
    zk_features = layers.Dense(8 * k, activation='relu')(concat_features_flatten)

    # 5层
    vk2_features = layers.Dense(2 * k)(zk_features)
    xk2_features = layers.Dense(k, activation=psi_activation)(zk_features)
    vk2_features = tf.expand_dims(input=vk2_features, axis=-1)
    xk2_features = tf.expand_dims(input=xk2_features, axis=-1)

    return f1_input, vk2_features, xk2_features, f2_input


def get_model():
    # 输入层
    f1_input = keras.Input(shape=(k, 1), name='f1')
    vk_input = keras.Input(shape=(2 * k, 1), name='vk')
    xk_input = keras.Input(shape=(k, 1), name='xk')
    f2_input = keras.Input(shape=(k, k), name='f2')

    # 第一块
    f1_input, vk2_features, xk2_features, f2_input = model_block(f1_input, vk_input, xk_input, f2_input)
    # 剩余的(L-1)块
    for _ in range(L - 1):
        f1_input, vk2_features, xk2_features, f2_input = model_block(f1_input, vk2_features, xk2_features, f2_input)

    # 构建模型
    my_model = keras.Model(inputs=[f1_input, vk_input, xk_input, f2_input],
                           outputs=[xk2_features])

    return my_model
