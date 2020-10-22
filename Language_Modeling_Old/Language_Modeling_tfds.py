import os
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import logging
import collections
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def features_labels(data_array, _batch_size, _seq_len, _shift, train):
    if not train:
        _batch_size = 1
    window_size = _seq_len + 1
    _shift = min(_shift, window_size)
    inputs = np.array([data_array[i:i + _seq_len] for i in range(0, data_array.shape[0] - window_size, _shift)])
    labels = np.array([data_array[i + 1:i + window_size] for i in range(0, data_array.shape[0] - window_size, _shift)])
    batch_steps = (inputs.shape[0]) // _batch_size
    _len = batch_steps * _batch_size
    inputs = inputs[:_len]
    labels = labels[:_len]
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.batch(_batch_size)
    return dataset, batch_steps


def data_loader(_batch_size, _seq_len, _shift):
    # data: Python lists of strings
    with open(dir + '/ptb.train.txt', 'r') as f1, open(dir + '/ptb.valid.txt', 'r') as f2, open(
            dir + '/ptb.test.txt', 'r') as f3:
        seq_train = f1.read().replace('\n', '<eos>').split(' ')
        seq_valid = f2.read().replace('\n', '<eos>').split(' ')
        seq_test = f3.read().replace('\n', '<eos>').split(' ')

    seq_train = list(filter(None, seq_train))
    seq_valid = list(filter(None, seq_valid))
    seq_test = list(filter(None, seq_test))

    logging.info(seq_train[:10])
    logging.info(seq_valid[:10])
    logging.info(seq_test[:10])

    # size
    size_train = len(seq_train)
    size_valid = len(seq_valid)
    size_test = len(seq_test)
    logging.info('size_train {}, size_valid {}, size_test {}'.format(
        size_train, size_valid, size_test))

    # vocabulary
    vocab_train = set(seq_train)
    vocab_valid = set(seq_valid)
    vocab_test = set(seq_test)
    logging.info('vocab_train {}, vocab_valid {}, vocab_test {}'.format(
        len(vocab_train), len(vocab_valid), len(vocab_test)))

    # creating dictionary - id ↔ word
    vocab_train = sorted(vocab_train)  # must have deterministic ordering, so word2id
    # dictionary is reproducible across invocations
    _word2id = {w: i for i, w in enumerate(vocab_train)}
    _id2word = {i: w for i, w in enumerate(vocab_train)}

    # data: np.int64 1-d numpy arrays -> np.int64 2-d numpy arrays of shape (seq_len*steps, batch_size)
    """ Note tf.contrib.cudnn_rnn.CudnnLSTM requires input tensor to be of shape
        (seq_len,batch_size,embedding_dim), where as tf.keras.layers.CuDNNLSTM
        requires input tensor to be of shape (batch_size,seq_len,embedding_dim) """

    ids_train = np.array([_word2id[word] for word in seq_train], copy=False, order='C')
    ids_valid = np.array([_word2id[word] for word in seq_valid], copy=False, order='C')
    ids_test = np.array([_word2id[word] for word in seq_test], copy=False, order='C')

    _data_train, _steps_train = features_labels(
        ids_train, _batch_size, _seq_len, _shift, train=True)
    _data_valid, _steps_valid = features_labels(
        ids_valid, _batch_size, _seq_len, _shift, train=False)
    _data_test, _steps_test = features_labels(
        ids_test, _batch_size, _seq_len, _shift, train=False)

    return _data_train, _steps_train, _data_valid, _steps_valid, _data_test, _steps_test, _word2id, _id2word


def model_create(vocab_size, encode_dim, layer_dim, batch, seq, stateful, n_layers, _dp):
    init = keras.initializers.random_uniform(-0.05, 0.05)
    # input is batch_size x steps x integer(vocab_size)
    # Creating RNN model and fit it:
    _model = keras.Sequential()
    # Creating embedding layer -
    # (vocabulary size, dimensions of the encoder, sequence length)
    _model.add(layers.Embedding(input_dim=vocab_size,
                                output_dim=encode_dim,
                                batch_input_shape=(batch, seq),
                                embeddings_initializer=init))

    for i in range(n_layers):
        # Add an LSTM layer with 'dim' internal units.
        _model.add(layers.LSTM(layer_dim,
                               return_sequences=True,
                               stateful=stateful,
                               dropout=_dp, recurrent_dropout=_dp,
                               # TODO: add regularizers l2
                               kernel_initializer=init, recurrent_initializer=init))

    # Add dropout layer:
    _model.add(layers.Dropout(_dp, noise_shape=(batch, 1, layer_dim)))

    # Add a Dense layer with len(dictionary) units - output is unit vector
    # TODO: read https://arxiv.org/pdf/1708.02182v1.pdf 4.5. Independent embedding size and hidden size
    # TODO: Try trainable=False, and copy weights as in the Embedding layer:
    #  https://arxiv.org/pdf/1708.02182v1.pdf 4.4.  Weight tying
    _model.add(layers.Dense(vocab_size, activation='softmax',
                            kernel_initializer=init))

    return _model


def ce_perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    # cross_entropy = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # perplexity = np.exp(np.mean(cross_entropy))
    y_true = np.array([y for x, y in y_true])         # y_true is a dataset of x,y
    cce = keras.losses.SparseCategoricalCrossentropy()
    ce_loss = cce(y_true, y_pred)
    perplexity_loss = np.exp(ce_loss)
    return np.float32(ce_loss), perplexity_loss


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
    dir = '../Language_Modeling_tf_Dataset/data'
    # General Info:
    epochs = 35
    batch_size = 20  # recommendation for validation - 1
    seq_len = 35  # length of input, and output
    shift = seq_len + 1  # shift = seq_len + 1 for stateful=True
    layers_num = 2
    hidden_layer_dim = 400
    encoder_dim = 400
    # Optimizer:
    clip_norm = 5
    # lr_scheduler:
    initial_learning_rate = 1.
    decay_epoch = 6
    decay = 0.833  # was 0.96
    dropout = 0.5   # TODO: variational dropout

    with open(log_dir + "/log.txt", "w") as file:
        file.write("NLP - Log " + timestamp +
                   "\n non-regularized LSTM: \n" +
                   ">> Epochs: " + str(epochs) + "\n" +
                   ">> Batch Size: " + str(batch_size) + "\n" +
                   ">> Sequence Length: " + str(seq_len) + "\n" +
                   ">> Data Shift Length: " + str(shift) + "\n" +
                   ">> Encoder Dimension: " + str(encoder_dim) + "\n" +
                   ">> Hidden Layer (1st LSTM): " + str(hidden_layer_dim) + "\n" +
                   ">> # LSTM layers: " + str(layers_num) + "\n" +
                   # ">> Dropout Rate: " + str(dropout) + "\n" +
                   ">> Initial Learning Rate: " + str(initial_learning_rate) + "\n" +
                   ">> Learning Rate # Epoch Decay: " + str(decay_epoch) + "\n" +
                   ">> Learning Rate Decay: x" + str(decay) + "\n" +
                   "\n\n")

    # *******************************************************************************
    # ---------------- data: Python lists of strings ------------------------------*
    # *******************************************************************************
    data_train, steps_train, data_valid, steps_valid, data_test, steps_test, word2id, id2word \
        = data_loader(batch_size, seq_len, shift)

    # *******************************************************************************
    # ----------------- configure TF graph -----------------------------------------*
    # *******************************************************************************
    model = model_create(vocab_size=len(word2id), encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                         n_layers=layers_num, batch=batch_size, seq=seq_len, stateful=True, _dp=dropout)
    model_test = model_create(vocab_size=len(word2id), encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                         n_layers=layers_num, batch=1, seq=seq_len, stateful=True, _dp=dropout)

    with open(log_dir + "/log.txt", "a") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=(steps_train * decay_epoch),   # decay after 6 epochs (was 100000)
        decay_rate=decay,
        staircase=True)

    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, clipnorm=clip_norm)

    model.compile(
        loss=_loss,
        optimizer=_optimizer,
        metrics=['sparse_categorical_crossentropy']  # TODO: ask about perplexity metric
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
        # train:
        hist = model.fit(data_train, epochs=1, shuffle=False)     # , callbacks=[checkpoint])
        model.reset_states()

        loss_train += hist.history['loss']
        perplexity_train += [np.exp(loss_train[-1])]

        # valid testing
        model_test.set_weights(model.get_weights())
        pred = model_test.predict(data_valid, batch_size=1)  # Ziv said batch_size=1
        ce_loss, perplex_loss = ce_perplexity(y_true=data_valid, y_pred=pred)
        loss_valid += [ce_loss]
        perplexity_valid += [perplex_loss]

        raw_data['time'] += [str(datetime.timedelta(seconds=round(time.time() - start_time)))]
        raw_data['epoch'] += [epoch + 1]
        raw_data['train loss'] += [loss_train[-1]]
        raw_data['valid loss'] += [loss_valid[-1]]
        raw_data['train perplexity'] += [perplexity_train[-1]]
        raw_data['valid perplexity'] += [perplexity_valid[-1].astype(float)]

        print("Time: " + str(raw_data['time']) + ", Epoch: " + str(epoch+1) + "\t|\tTraining - loss: " + str(loss_train[-1])
              + ", Perplexity: " + str(perplexity_train[-1]) + "\t|\tValidation - loss: " + str(loss_valid[-1])
              + ", Perplexity: " + str(perplexity_valid[-1]))

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
    # test prediction
    model_test.set_weights(model.get_weights())
    pred = model_test.predict(data_test, batch_size=1)  # Ziv said batch_size=1
    _, perplex = ce_perplexity(y_true=data_test, y_pred=pred)

    txt = "======= End of training results =======\n" + \
          "Train perplexity: \t" + str(perplexity_train[-1]) + "\n" + \
          "Valid perplexity: \t" + str(perplexity_valid[-1]) + "\n" + \
          "Test perplexity: \t" + str(perplex.astype(float))
    with open(log_dir + "/log.txt", "a") as file:
        file.write(txt)
    print(txt)
