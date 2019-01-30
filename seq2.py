import random
import numpy as np
from keras import layers
from keras.layers import Input, Embedding, Bidirectional, Dense, Concatenate, LSTM
from keras.models import Model, load_model
from keras import callbacks

import matplotlib.pyplot as plt

num_samples = 50000

data_path = 'dataset.txt'
# 读入数据
with open(data_path, 'r') as f:
    lines = f.read().split('\n')

# 打乱顺序
random.shuffle(lines)
# 显示部分数据
for line in lines[:10]:
    print(line)


#---------------------------
input_texts = []  # 输入数据，也就是数字字符串
target_texts = []  # 目标数据，也就是中文大写字符串
input_characters = set()
target_characters = set()
# 分割数据为 input_texts 和 target_texts
# 且统计一些信息
for line in lines[: min(num_samples, len(lines) - 1)]:
    try:
        input_text, target_text = line.split('\t')
    except ValueError:
        print('Error line:', line)
        input_text = ''
        target_text = ''

    # 计算 input_texts 中的 tokens数量
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)

    # 计算 target_texts 中的 tokens数量
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

    # 用 '始'作为开始字符
    # 用 '终'作为结束字符
    target_text = '始' + target_text + '终'

    input_texts.append(input_text)
    target_texts.append(target_text)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Nunmber of samples:', len(input_texts))
print('Max sequence length of input:', max_encoder_seq_length)
print('Max sequence length of outputs:', max_decoder_seq_length)

# 建立字典，用于字符转数字

# '[pad]'表示填充字符
input_token_index = dict( [(char, i) for i, char in enumerate(['pad']+input_characters)])

# '始'表示开始字符，'终'表示终止字符，'空'表示填充字符
special_characters = ['空', '始', '终']
target_token_index = dict([ (char,i ) for i, char in enumerate(special_characters + target_characters)])

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)

def characters2index(text, char_index):
    # 将字符串向量化
    return [char_index.get(char) for char in text]


# 创建向量化的数据
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype=np.int32)
decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length), dtype=np.int32)
# decoder_target_data 需要 one-hot 编码
decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype=np.float32)

# 填充数据
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    input_indexs = characters2index(input_text, input_token_index)
    encoder_input_data[i, :len(input_indexs)] = input_indexs

    target_indexs = characters2index(target_text, target_token_index)
    decoder_input_data[i, :len(target_indexs)] = target_indexs

    # decoder_target_data 做one-hot编码，且偏移一位
    for t, index in enumerate(decoder_input_data[i, 1:]):
        decoder_target_data[i, t, index] = 1.0
    decoder_target_data[i, -1, 0] = 1.0



# 超参数
encoder_embedding_dim = 10
decoder_embedding_dim = 20
latent_dim = 128

batch_size = 64
epochs = 20


def build_basic_model():
    rnn = layers.LSTM
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(num_encoder_tokens, encoder_embedding_dim, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = rnn(latent_dim, return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.5,
                       name='encoder_lstm')
    _, *encoder_states = encoder_lstm(encoder_embedding)

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(num_decoder_tokens, decoder_embedding_dim, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = rnn(latent_dim, return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.5,
                       name='decoder_lstm')
    rnn_outputs, *decoder_states = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(rnn_outputs)

    basic_model = Model([encoder_inputs, decoder_inputs], [decoder_outputs])
    basic_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return basic_model



# 回调函数
callback_list = [callbacks.ModelCheckpoint('basic_model_best.h', save_best_only=True)]
# 获取模型
basic_model = build_basic_model()
# 训练
basic_model_hist = basic_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size, epochs=epochs,
                validation_split=0.2, callbacks=callback_list)


# 建立推理模型
def build_basic_inference_model(model_path):
    model = load_model(model_path)

    # encoder
    encoder_inputs = Input(shape=(None,))
    # encoder_embedding
    encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
    # get encoder states
    _, *encoder_states = model.get_layer('encoder_lstm')(encoder_embedding)
    encoder_model = Model(encoder_inputs, encoder_states)

    # decoder
    # decoder inputs
    decoder_inputs = Input(shape=(None,))
    # decoder input states
    decoder_state_h = Input(shape=(latent_dim,))
    decoder_state_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_h, decoder_state_c]

    # decoder embedding
    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    # get rnn outputs and decoder states
    rnn_outputs, *decoder_states = model.get_layer('decoder_lstm')(decoder_embedding,
                                                                   initial_state=decoder_states_inputs)
    decoder_outputs = model.get_layer('decoder_dense')(rnn_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


reverse_input_char_index = dict([(i, char) for char, i in input_token_index.items()])
reverse_target_word_index = dict([(i, char) for char, i in target_token_index.items()])


def decode_sequence(input_seq, encoder_model, decoder_model):
    # get encoder states
    states_value = encoder_model.predict(input_seq)

    # create a empty sequence
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['始']

    # 进行句子的恢复
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output, *decoder_states = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_word

        if sampled_word == '终' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # update target_seq
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # update states
        states_value = decoder_states

    return decoded_sentence

# 建立一些测试用的样本
test_input_texts = [line.split('\t')[0] for line in lines[-20:]]
test_input_data = np.zeros((len(test_input_texts), max_encoder_seq_length), np.int32)
for i, test_text in enumerate(test_input_texts):
    test_indexs = characters2index(test_text, input_token_index)
    test_input_data[i, :len(test_indexs)] = test_indexs


# 导入模型
model_path = 'basic_model_best.h'
inference_encoder_model, inference_decoder_model = build_basic_inference_model(model_path)

# 测试
for seq_index in range(len(test_input_texts)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = test_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq, inference_encoder_model, inference_decoder_model)
    print('-')
    print('Input sentence:', test_input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)