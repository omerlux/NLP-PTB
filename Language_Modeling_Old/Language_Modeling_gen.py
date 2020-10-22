import os
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
import collections
import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# fix for memory problem...
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print("======= Examples =======")
    print("Train data: ")
    print(train_data[:5])
    print("Vocabulary: ")
    print(vocabulary)
    print("Reversed dictionary: " + " ".join([reversed_dictionary[x] for x in train_data[:10]]))
    print("========================")
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

    def generateX(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x

    def generateY(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield y


def model_create(vocab_size, encode_dim, layer_dim, batch, seq, stateful, n_layers):
    init = keras.initializers.random_uniform(-0.05, 0.05)
    # input is batch_size x steps x integer(vocab_size)
    # Creating RNN model and fit it:
    _model = keras.Sequential()
    # Creating embedding layer -
    # (vocabulary size, dimensions of the encoder, sequence length)
    _model.add(layers.Embedding(input_dim=vocab_size, output_dim=encode_dim, batch_input_shape=(batch, seq),
                                embeddings_initializer=init))

    for i in range(n_layers):
        # Add an LSTM layer with 'dim' internal units.
        _model.add(layers.LSTM(layer_dim, return_sequences=True, stateful=stateful,
                               kernel_initializer=init, recurrent_initializer=init))

    # Add a Dense layer with 'encode_dim' units - 'decoding' layer
    # read https://arxiv.org/pdf/1708.02182v1.pdf 4.5. Independent embedding size and hidden size
    _model.add(layers.Dense(encode_dim, activation='linear',
                            kernel_initializer=init))

    # Add a Dense layer with len(dictionary) units - output is unit vector
    # TODO: Try trainable=False, and copy weights as in the Embedding layer:
    #  https://arxiv.org/pdf/1708.02182v1.pdf 4.4.  Weight tying
    _model.add(layers.Dense(vocab_size, activation='softmax',
                            kernel_initializer=init))

    return _model


def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    # cross_entropy = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # perplexity = np.exp(np.mean(cross_entropy))
    cce = keras.losses.SparseCategoricalCrossentropy()
    perplexity = np.exp(cce(y_true, y_pred))
    return perplexity


if __name__ == "__main__":
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    logging.info('date {}'.format(datetime.datetime.now()))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir("logs/" + timestamp)
    log_dir = "logs/" + timestamp

    # *******************************************************************************
    # -------------- --- hyper parameters -----------------------------------------*
    # *******************************************************************************
    data_path = '../Language_Modeling_Generator/data'
    epochs = 39
    batch_size = 10  # recommendation for validation - 1
    seq_len = 35  # length of input, and output
    # shift = seq_len + 1  # shift = seq_len + 1 for stateful=True
    layers_num = 1
    hidden_layer_dim = 200
    encoder_dim = 200
    # optimizer:
    clip_norm = 5
    # lr_scheduler:
    initial_learning_rate = 1.
    decay_epoch = 6
    decay = 0.833  # was 0.96
    # dropout = 0   TODO: variational dropout

    with open(log_dir + "/log.txt", "w") as file:
        file.write("NLP - Log " + timestamp +
                   "\n non-regularized LSTM: \n" +
                   ">> Epochs: " + str(epochs) + "\n" +
                   ">> Batch Size: " + str(batch_size) + "\n" +
                   ">> Sequence Length: " + str(seq_len) + "\n" +
                   ">> Encoder Dimension: " + str(encoder_dim) + "\n" +
                   ">> Hidden Layer (1st LSTM): " + str(hidden_layer_dim) + "\n" +
                   ">> # LSTM layers: " + str(layers_num) + "\n" +
                   ">> Initial Learning Rate: " + str(initial_learning_rate) + "\n" +
                   ">> Learning Rate # Epoch Decay: " + str(decay_epoch) + "\n" +
                   ">> Learning Rate Decay: x" + str(decay) + "\n" +
                   "\n\n")

    # *******************************************************************************
    # ---------------- data: Python lists of strings ------------------------------*
    # *******************************************************************************
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
    # generate data - (batch_size x seq_len x vocabulary) is actually 1 batch input
    train_data_generator = KerasBatchGenerator(train_data, seq_len, batch_size, vocabulary,
                                               skip_step=seq_len)
    valid_data_generator = KerasBatchGenerator(valid_data, seq_len, batch_size, vocabulary,
                                               skip_step=seq_len)
    test_data_generator = KerasBatchGenerator(test_data, seq_len, batch_size, vocabulary,
                                               skip_step=seq_len)
    train_steps_4epoch = len(train_data) // (batch_size * seq_len)
    valid_steps_4epoch = len(valid_data) // (batch_size * seq_len)
    test_steps_4epoch = len(test_data) // (batch_size * seq_len)
    # *******************************************************************************
    # ----------------- configure TF graph -----------------------------------------*
    # *******************************************************************************
    # Batch size is 1, the batch_size is taking in count in the data itself because we want to use the stateful
    model = model_create(vocab_size=vocabulary, encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                         n_layers=layers_num, batch=batch_size, seq=seq_len, stateful=True)
    model_test = model_create(vocab_size=vocabulary, encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                              n_layers=layers_num, batch=batch_size, seq=seq_len, stateful=True)
    with open(log_dir + "/log.txt", "a") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=(train_steps_4epoch * decay_epoch),  # decay after 6 epochs (was 100000)
        decay_rate=decay,
        staircase=True)
    _loss = tf.keras.losses.CategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, clipnorm=clip_norm)
    model.compile(
        loss=_loss,
        optimizer=_optimizer,
        metrics=['categorical_crossentropy']  # TODO: ask about perplexity metric
    )

    # *******************************************************************************
    # ------------------------- initialize ----------------------------------------*
    # *******************************************************************************
    loss_train = []
    loss_valid = []
    perplexity_train = []
    perplexity_valid = []

    raw_data = {'time': [],
                'epoch': [],
                'train loss': [],
                'valid loss': [],
                'train perplexity': [],
                'valid perplexity': []}
    columns = ['time', 'epoch', 'train loss', 'valid loss', 'train perplexity', 'valid perplexity']

    # *******************************************************************************
    # ---------------------- train and evaluate -----------------------------------*
    # *******************************************************************************
    #  checkpoint = keras.callbacks.ModelCheckpoint('logs/' + timestamp + '/checkpoint',
    #                                             monitor='val_loss', save_best_only=True, mode='min') # TODO: uncomment
    for epoch in range(epochs):
        print("\nEPOCH " + str(epoch + 1) + "/" + str(epochs))
        # train fit:
        hist = model.fit(train_data_generator.generate(), steps_per_epoch=train_steps_4epoch, epochs=1,
                         validation_data=valid_data_generator.generate(), validation_steps=valid_steps_4epoch,
                         shuffle=False)
        # TODO: UNCOMMENT - , callbacks=[checkpoint])
        model.reset_states()

        loss_train += hist.history['loss']
        loss_valid += hist.history['val_loss']
        perplexity_train += [np.exp(loss_train[-1])]

        # valid prediction:
        model_test.set_weights(model.get_weights())
        pred = model.predict(valid_data_generator.generate())
        perplex = perplexity(y_pred=pred, y_true=valid_data_generator.generateY())
        perplexity_valid += [perplex]

        raw_data['time'] += [str(datetime.timedelta(seconds=round(time.time() - start_time)))]
        raw_data['epoch'] += [epoch + 1]
        raw_data['train loss'] += [loss_train[-1]]
        raw_data['valid loss'] += [loss_valid[-1]]
        raw_data['train perplexity'] += [perplexity_train[-1]]
        raw_data['valid perplexity'] += [perplexity_valid[-1].astype(float)]

        df = pd.DataFrame(raw_data, columns=columns)
        df.to_csv(log_dir + '/fit_' + timestamp + '.csv')

    model.save(log_dir + '/model')
    # *******************************************************************************
    # ----------------------- plot ------------------------------------------------*
    # *******************************************************************************
    plt.rcParams['axes.facecolor'] = 'floralwhite'
    plt.plot(range(epochs), perplexity_train, linewidth=1, color='blue', label='training')
    plt.plot(range(epochs), perplexity_valid, linewidth=1, color='red', label='validation')
    plt.grid(True, which='both', axis='both')
    plt.title('Penn Treebank Corpus')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(log_dir + "/graph.png")
    plt.show()
    plt.clf()

    m = min(10, epochs // 2)
    plt.rcParams['axes.facecolor'] = 'floralwhite'
    plt.plot(range(epochs)[m:], perplexity_train[m:], linewidth=1, color='blue', label='training')
    plt.plot(range(epochs)[m:], perplexity_valid[m:], linewidth=1, color='red', label='validation')
    plt.grid(True, which='both', axis='both')
    plt.title('Penn Treebank Corpus - Zoomed')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig(log_dir + "/graph_zoomed.png")
    plt.show()

    # # *******************************************************************************
    # # ------------------------ test -----------------------------------------------*
    # # *******************************************************************************
    # test prediction:
    model_test.set_weights(model.get_weights())
    pred = model.predict(test_data_generator.generate())
    perplex = perplexity(y_pred=pred, y_true=test_data_generator.generateY())

    txt = "======= End of training results =======\n" + \
          "Train perplexity: \t" + str(perplexity_train[-1]) + "\n" + \
          "Valid perplexity: \t" + str(perplexity_valid[-1]) + "\n" + \
          "Test perplexity: \t" + str(perplex.astype(float)) + "\n" + \
          "======================================="
    with open(log_dir + "/log.txt", "a") as file:
        file.write(txt)
    print(txt)

