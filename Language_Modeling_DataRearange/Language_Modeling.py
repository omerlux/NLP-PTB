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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.autograph.set_verbosity(0)


# FIX: allocating only some memory
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def features_labels(data_array, _batch_size, _seq_len):
    batch_steps = (data_array.shape[0] - 1) // (_batch_size * _seq_len)
    _newlength = batch_steps * _batch_size * _seq_len
    x = np.array(data_array[:_newlength])
    y = np.array(data_array[1:_newlength + 1])
    x = x.reshape(_batch_size, -1, _seq_len)
    y = y.reshape(_batch_size, -1, _seq_len)
    inputs = []
    labels = []
    # merge batch at the same epoch steps, so every elment that we insert
    # is from different epoch step - stateful should work fine now
    for i in range(batch_steps):
        for j in range(_batch_size):
            inputs.append(x[j][i])
            labels.append(y[j][i])
    inputs = np.array(inputs).reshape(-1, _seq_len)
    labels = np.array(labels).reshape(-1, _seq_len)
    Data = collections.namedtuple('Data', ['inputs', 'labels'])
    return Data(inputs=inputs, labels=labels), batch_steps


def data_loader(_batch_size, _seq_len):
    dir = 'data'
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

    _data_train_batched, _steps_train = features_labels(ids_train, _batch_size, _seq_len)
    _data_valid_batched, _steps_valid = features_labels(ids_valid, _batch_size, _seq_len)
    _data_valid, _steps_valid = features_labels(ids_valid, 1, _seq_len)
    _data_test, _steps_test = features_labels(ids_test, 1, _seq_len)

    return _data_train_batched, _steps_train, _data_valid_batched, _data_valid, _data_test, _word2id, _id2word


def model_create(vocab_size, encode_dim, layer_dim, batch, seq, stateful, n_layers, _dp, _vari):
    # input is batch_size x steps x integer(vocab_size)
    _layers = []

    # Creating embedding layer -
    # (vocabulary size, dimensions of the encoder, sequence length)
    _layers += [keras.layers.Embedding(input_dim=vocab_size,
                                       output_dim=encode_dim,
                                       batch_input_shape=(batch, seq),
                                       # CAMERON: No weights initialization
                                       # embeddings_initializer=keras.initializers.random_uniform(-0.05, 0.05)
                                       )]

    for i in range(n_layers):
        if _vari:  # NOTE: recurrent dropout will cause fit on CPU!
            _rec_dp = _dp
        else:
            _rec_dp = 0.

        # Dropout mask layer
        _layers += [keras.layers.Dropout(_dp, noise_shape=(batch, 1, layer_dim))]

        # Add an LSTM layer with 'dim' internal units
        _layers += [keras.layers.LSTM(
            layer_dim,  # hidden layer size
            return_sequences=True,
            stateful=stateful,
            recurrent_dropout=_rec_dp,
            kernel_regularizer=keras.regularizers.l1_l2(0, 0),  # 1e-5, 1e-5),
            activity_regularizer=keras.regularizers.l1_l2(0, 0),  # 1e-5, 1e-5),
            recurrent_regularizer=keras.regularizers.l1_l2(0, 0),  # 1e-5, 1e-5),
            # CAMERON: No weights initialization
            # kernel_initializer=keras.initializers.random_uniform(-0.05, 0.05),
            # recurrent_initializer=keras.initializers.random_uniform(-0.05, 0.05)
        )]

    # Add dropout layer:
    _layers += [keras.layers.Dropout(_dp, noise_shape=(batch, 1, layer_dim))]

    # TimeDistributed layer -  This function adds an independent layer for each time step in the recurrent model
    # Add a Dense layer with len(dictionary) units - output is unit vector
    # TODO: Try trainable=False, and copy weights as in the Embedding layer:
    #  https://arxiv.org/pdf/1708.02182v1.pdf 4.4.  Weight tying
    _layers += [keras.layers.TimeDistributed(
        keras.layers.Dense(vocab_size,
                           activity_regularizer=keras.regularizers.l1_l2(0, 0),  # 1e-5, 1e-5),
                           kernel_regularizer=keras.regularizers.l1_l2(0, 0),  # 1e-5, 1e-5),
                           activation='softmax',
                           # CAMERON: No weights initialization
                           # kernel_initializer=keras.initializers.random_uniform(-0.05, 0.05)
                           )
    )]

    # Creating RNN model:
    return keras.Sequential(_layers)


def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!
    https://stackoverflow.com/questions/41881308/how-to-calculate-perplexity-of-rnn-in-tensorflow
    https://github.com/keras-team/keras/issues/8267
    """
    # cross_entropy = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # perplexity = np.exp(np.mean(cross_entropy))
    cce = keras.losses.SparseCategoricalCrossentropy()
    _cce_loss = cce(y_true, y_pred)
    return np.exp(_cce_loss)


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
    # General Info:
    epochs = 100
    batch_size = 40  # recommendation for validation - 1
    seq_len = 70  # length of input, and output
    layers_num = 3
    hidden_layer_dim = 85
    encoder_dim = 850
    dropout = 0.4
    variational = True  # Variational dropout - not working on GPU (no CUDNN support)
    # Optimizer:
    moment = 0.90  # recommended by Cameron0.9
    early_stop = 10  # 15 epochs no progress
    # lr_scheduler:
    initial_learning_rate = 1.1
    curr_lr = initial_learning_rate
    # decay_epoch = 14
    decay = 0.5  # good results with 0.5
    decay_distance = 0.0  # distance between last perplexity and previous

    with open(log_dir + "/log.txt", "w") as file:
        file.write("NLP - Log " + timestamp + "\n"
                                              ">> Epochs: " + str(epochs) + "\n" +
                   ">> Batch Size: " + str(batch_size) + "\n" +
                   ">> Sequence Length: " + str(seq_len) + "\n" +
                   ">> Encoder Dimension: " + str(encoder_dim) + "\n" +
                   ">> Hidden Layer (1st LSTM): " + str(hidden_layer_dim) + "\n" +
                   ">> # LSTM layers: " + str(layers_num) + "\n" +
                   ">> Dropout Rate: " + str(dropout) + "\n" +
                   ">> Dropout Variational: " + str(variational) + "\n" +
                   ">> Momentum: " + str(moment) + "\n" +
                   ">> Early Stopping, # of Epochs: " + str(early_stop) + "\n" +
                   ">> Initial Learning Rate: " + str(initial_learning_rate) + "\n" +
                   ">> Learning Rate Decay by: x" + str(decay) + "\n" +
                   ">> Learning Rate Decay Distance: " + str(decay_distance) + "\n" +
                   "\n\n")

    # *******************************************************************************
    # ---------------- data: Python lists of strings ------------------------------*
    # *******************************************************************************
    data_train_batched, steps_train, data_valid_batched, data_valid, data_test, word2id, id2word \
        = data_loader(batch_size, seq_len)

    # *******************************************************************************
    # ----------------- configure TF graph -----------------------------------------*
    # *******************************************************************************
    model = model_create(vocab_size=len(word2id), encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                         n_layers=layers_num, batch=batch_size, seq=seq_len, stateful=True,
                         _dp=dropout, _vari=variational)
    model_test = model_create(vocab_size=len(word2id), encode_dim=encoder_dim, layer_dim=hidden_layer_dim,
                              n_layers=layers_num, batch=1, seq=seq_len, stateful=True,
                              _dp=dropout, _vari=variational)
    with open(log_dir + "/log.txt", "a") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=(steps_train * decay_epoch),  # decay after x epochs (was 100000)
    #     decay_rate=decay,
    #     staircase=True)
    _loss = tf.keras.losses.SparseCategoricalCrossentropy()
    _optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=moment)  # clipnorm=clip_norm)
    model.compile(
        loss=_loss,
        optimizer=_optimizer,
        metrics=['sparse_categorical_crossentropy']
    )
    model_test.compile(
        loss=_loss,
        metrics=['sparse_categorical_crossentropy']
    )

    # *******************************************************************************
    # ------------------------- initialize ----------------------------------------*
    # *******************************************************************************
    loss_train = []
    loss_valid = []
    perplexity_train = []
    perplexity_valid = []
    decay_occur = []

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
    # Callbacks:
    # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=0, factor=0.5, min_lr=0.001)
    # es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = keras.callbacks.ModelCheckpoint("logs/" + timestamp + '/checkpoint',
                                                 monitor='val_loss', save_best_only=True)  # TODO: uncomment
    for epoch in range(epochs):
        print("\nEPOCH " + str(epoch + 1) + "/" + str(epochs))

        # Learning Rate Decay:
        if epoch > 2 and raw_data['valid perplexity'][-1] + decay_distance > raw_data['valid perplexity'][-2]:
            curr_lr = curr_lr * decay
            print("\tLearning rate decayed to: {0}".format(curr_lr))
            decay_occur += [epoch + 1]
            _optimizer = tf.keras.optimizers.SGD(learning_rate=curr_lr, momentum=moment)
            model.compile(
                loss=_loss,
                optimizer=_optimizer,
                metrics=['sparse_categorical_crossentropy']  # TODO: ask about perplexity metric
            )

        # train:
        hist = model.fit(data_train_batched.inputs, data_train_batched.labels,
                         epochs=1, batch_size=batch_size,
                         validation_data=(data_valid_batched.inputs, data_valid_batched.labels), shuffle=False,
                         verbose=2,
                         callbacks=[checkpoint]
                         )
        model.reset_states()

        loss_train += hist.history['loss']
        perplexity_train += [np.exp(loss_train[-1])]

        # valid testing
        print("\t Epoch {0} validation evaluation...".format(epoch + 1))
        model_test.set_weights(model.get_weights())
        # TODO: OOM Problem...
        # pred = model_test.predict(data_valid.inputs, batch_size=1)      # Ziv said batch_size=1
        # perplex = perplexity(y_pred=pred, y_true=data_valid.labels)
        # perplexity_valid += [perplex]
        validation_ce = model_test.evaluate(data_valid.inputs, data_valid.labels,
                                            batch_size=1, verbose=2)[0]
        loss_valid += [validation_ce]
        perplexity_valid += [np.exp(validation_ce)]

        raw_data['time'] += [str(datetime.timedelta(seconds=round(time.time() - start_time)))]
        raw_data['epoch'] += [epoch + 1]
        raw_data['train loss'] += [loss_train[-1]]
        raw_data['valid loss'] += [loss_valid[-1]]
        raw_data['train perplexity'] += [perplexity_train[-1]]
        raw_data['valid perplexity'] += [perplexity_valid[-1].astype(float)]

        print("Time: " + str(raw_data['time'][-1]) + ", Epoch: " + str(epoch + 1)
              + "\t|\tTraining - loss: %.5f" % loss_train[-1]
              + ", Perplexity: %.3f" % perplexity_train[-1]
              + "\t|\tValidation - loss: %.5f" % loss_valid[-1]
              + ", Perplexity: %.3f" % perplexity_valid[-1])

        df = pd.DataFrame(raw_data, columns=columns)
        df.to_csv(log_dir + '/fit_' + timestamp + '.csv')

        # Early Stopping:
        if epoch > early_stop \
                and raw_data['valid perplexity'][-early_stop] <= min(raw_data['valid perplexity'][-early_stop + 1:]):
            print("\tEarly stopping. There is no progress in the last {0} epochs.".format(early_stop))
            epochs = epoch + 1
            break

    model.save(log_dir + '/model')
    model_test.save(log_dir + '/model_test')
    # *******************************************************************************
    # ----------------------- train & valid results --------------------------------*
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

    # not enough memory... test prediction
    # model_test.set_weights(model.get_weights())
    # pred = model_test.predict(data_test.inputs, batch_size=1)  # Ziv said batch_size=1
    # perplex = perplexity(y_pred=pred, y_true=data_test.labels)

    test_ce = model_test.evaluate(data_test.inputs, data_test.labels,
                                  batch_size=1, verbose=2)[0]
    perplex_test = np.exp(test_ce)

    txt = "========= End of training results =========\n" + \
          "Early stopping at epoch number \t" + str(epoch + 1) + "\n" + \
          "Learning rate decay at epochs: \t" + str(decay_occur) + "\n" + \
          "Train perplexity: \t" + str(perplexity_train[-1]) + "\n" + \
          "Valid perplexity: \t" + str(perplexity_valid[-1]) + "\n" + \
          "Test perplexity: \t" + str(perplex_test.astype(float))
    with open(log_dir + "/log.txt", "a") as file:
        file.write(txt)
    print(txt)
