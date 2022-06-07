from __future__ import absolute_import, division, print_function
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import get_model

tf.keras.backend.clear_session()

k = 4  # L需要在get_model.py中修改,修改后还需要修改tf_op_layer_ExpandDims_？
sample_total = 100000  # train样本大小
# 这里的val_sample_total=1000不是最后预测的，预测用的5000评估，这里是训练过程中给每个epoch评估一下的，其数据是随机SNR生成的
val_sample_total = 1000
epoch = 1000
batch_size = 1000
lr = 1e-3


# 从文件中读取数据,数据以空格隔开.file_name：文件名；number：每行的数据个数
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


# 自定义评估指标,误码率BER,越低越好
class my_precision_BER(tf.keras.metrics.Metric):
    def __init__(self):
        super(my_precision_BER, self).__init__()
        self.total = self.add_weight(name='total', dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.round(y_pred)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        cmp_two = tf.equal(y_pred, y_true)
        cmp_two = tf.cast(cmp_two, tf.float32)
        predict_right_bit = tf.reduce_sum(cmp_two)
        all_bit = tf.shape(y_true)[0] * tf.shape(y_true)[1]
        all_bit = tf.cast(all_bit, tf.float32)
        predict_wrong_bit = all_bit - predict_right_bit
        predict_wrong_bit = tf.cast(predict_wrong_bit, tf.float32)
        self.total.assign_add(all_bit)
        self.count.assign_add(predict_wrong_bit)

    def result(self):
        return self.count / self.total


# # # # # # # # # # # # # #  搭建模型
model = get_model.get_model()
model.summary()
keras.utils.plot_model(model, '../image/new_model.png', show_shapes=True)  # 画出模型


# # # # # # # # # # # # # #  编译模型
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss={'tf_op_layer_ExpandDims_15': 'mse'},  # 若L更改，这个tf_op_layer_ExpandDims_？也要改
              metrics=[my_precision_BER()]
              )

# # # # # # # # # # # # # #  载入数据并训练,sample_total是train总样本数,val_sample_total是val总样本数
# train
f1_data = read_data('../../data/f1_Hty.txt', k)
f1_data = np.array(f1_data).reshape((sample_total, k, 1))

vk_data = np.zeros(shape=(sample_total, 2 * k, 1), dtype=np.float32)

xk_data = np.zeros(shape=(sample_total, k, 1), dtype=np.float32)

f2_data = read_data('../../data/f2_HtH.txt', k * k)
f2_data = np.array(f2_data).reshape((sample_total, k, k))
# 标签
xk2_label = read_data('../../data/x.txt', k)
xk2_label = np.array(xk2_label).reshape((sample_total, k, 1))

# val
val_f1_data = read_data('../../data/val_f1_Hty.txt', k)
val_f1_data = np.array(val_f1_data).reshape((val_sample_total, k, 1))

val_vk_data = np.zeros(shape=(val_sample_total, 2 * k, 1), dtype=np.float32)

val_xk_data = np.zeros(shape=(val_sample_total, k, 1), dtype=np.float32)

val_f2_data = read_data('../../data/val_f2_HtH.txt', k * k)
val_f2_data = np.array(val_f2_data).reshape((val_sample_total, k, k))
# val_标签
val_xk2_label = read_data('../../data/val_x.txt', k)
val_xk2_label = np.array(val_xk2_label).reshape((val_sample_total, k, 1))

# # # # # # # # # # # # #  断点续训
checkpoint_save_path = "checkpoint/model.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

# # # # # # # # # # # # # # 训练模型
history = model.fit(
    {'f1': f1_data, 'vk': vk_data, 'xk': xk_data, 'f2': f2_data},
    {'tf_op_layer_ExpandDims_15': xk2_label},  # 若L更改，这个tf_op_layer_ExpandDims_？也要改
    batch_size=batch_size,
    epochs=epoch,
    validation_data=({'f1': val_f1_data, 'vk': val_vk_data, 'xk': val_xk_data, 'f2': val_f2_data},
                     {'tf_op_layer_ExpandDims_15': val_xk2_label}),  # 若L更改，这个tf_op_layer_ExpandDims_？也要改
    validation_freq=1,
    callbacks=[cp_callback]
)
print('----------------------------model.trainable_variables----------------------------')
# print(model.trainable_variables)
print('------------------------------------------------------------------------------------')
file = open('checkpoint/weights.txt', 'w')  # 将模型参数写入txt文件
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# # # # # # # # # # # # # #  显示训练和验证的BER和loss曲线
acc = history.history['my_precision_ber']
val_acc = history.history['val_my_precision_ber']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training BER')
plt.plot(val_acc, label='Validation BER')
plt.title('Training and Validation BER')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
