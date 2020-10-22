import os
import sys
import tensorflow as tf
import Language_Modeling_Old as LM
import numpy as np
from tensorflow import keras

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# FIX: allocating only some memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

if __name__ == "__main__":
    # *******************************************************************************
    # ------------------- hyper parameters -----------------------------------------*
    # *******************************************************************************
    log_dir = "logs/" + sys.argv[1]
    seq_len = int(sys.argv[2])

    # *******************************************************************************
    # ------------------------ test -----------------------------------------------*
    # *******************************************************************************
    # test prediction

    model_test = keras.models.load_model(log_dir + "/model_test")
    _, _, _, _, data_test, _, _ = LM.data_loader(1, seq_len)
    # pred = model_test.predict(data_test.inputs, batch_size=1)  # Ziv said batch_size=1
    # perplex = LM.perplexity(y_pred=pred, y_true=data_test.labels)
    test_ce = model_test.evaluate(data_test.inputs, data_test.labels, batch_size=1)[0]
    test_perplex = np.exp(test_ce)

    txt = "========= Test results =========\n" + \
          "Test perplexity: \t" + str(test_perplex.astype(float))
    with open(log_dir + "/log.txt", "a") as file:
        file.write(txt)
    print(txt)
