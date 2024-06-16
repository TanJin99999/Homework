import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re

###读取文件夹中的所有文件
def read_files(path):
    files_to_read = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                files_to_read.append(os.path.join(root, file))
    return files_to_read
###去掉语料库中不相关的部分
def is_uchar(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False
###生成训练数据
def data_generator(data, batch_size, time_steps):
    num_batches = len(data) // (batch_size * time_steps)
    data = data[:num_batches * batch_size * time_steps]
    data = np.array(data).reshape((batch_size, -1))
    while True:
        for i in range(0, data.shape[1], time_steps):
            x = data[:, i:i + time_steps]
            y = np.roll(x, -1, axis=1)
            yield x, y
###定义RNN模型
class RNNModel(tf.keras.Model):
    def __init__(self, hidden_size, hidden_layers, vocab_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True) for _ in
                            range(hidden_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs)
        new_states = []
        for i in range(self.hidden_layers):
            x, state_h, state_c = self.lstm_layers[i](x, initial_state=states[i] if states else None, training=training)
            new_states.append([state_h, state_c])
        x = self.dense(x)
        if return_state:
            return x, new_states
        else:
            return x
###回调函数
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
###文本生成函数
def generate_text(model, start_string, num_generate=100):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    states = None
    for i in range(num_generate):
        predictions, states = model(input_eval, states=states, return_state=True)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id2char[predicted_id])
    out_put=''.join(text_generated)
    return start_string + out_put

#########################################################################################
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is NOT available.")

######参数设定：
hidden_size = 256  
hidden_layers = 3 
batch_size = 32  
time_steps = 50  
epochs = 20
learning_rate = 0.01 
#####读取并处理文件
with open (r'D://Resource//射雕英雄传.txt', encoding='gbk', errors='ignore') as f:
    data = f.readlines()
pattern = re.compile(r'//(.*//)')
data = [pattern.sub('', lines) for lines in data]
data = [line.replace('……', '。') for line in data if len(line) > 1]
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)
######构建词表
vocab = list(set(data))
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for i, c in enumerate(vocab)}
vocab_size=len(vocab)
num_data = [char2id[char] for char in data]
######获取训练数据
train_data = data_generator(num_data, batch_size, time_steps)
####定义RNN模型的结构
model = RNNModel(hidden_size, hidden_layers, vocab_size)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
####记录训练过程中的损失值
history = LossHistory()
####训练模型
model.fit(train_data, epochs=epochs, steps_per_epoch=len(num_data) // (batch_size * time_steps), callbacks=[history])
####绘制loss曲线
plt.plot(history.losses)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()

####生成文本示例
print(generate_text(model, start_string="道长喝酒用的是内功，兄弟用的却是外功，乃体外之功。你请瞧吧！"))

