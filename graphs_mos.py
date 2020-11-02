import numpy as np
import matplotlib.pyplot as plt
import re


valid_ppl_mc = []
valid_ppl_mc_error = []
valid_ppl = []
train_ppl = []
epoch_mc = []
curr_epoch = 0
for line in open('log.txt'):
    numbers = re.findall("[^a-zA-Z:](\d+[\.]?\d*)", line)
    if "valid ppl avg" in line:
        epoch_mc.append(curr_epoch)
        valid_ppl_mc.append(float(numbers[9]))
        valid_ppl_mc_error.append(float(numbers[11]) - float(numbers[10]))
    if ("valid ppl" in line) and not ("valid ppl avg" in line):
        valid_ppl.append(float(numbers[10]))
    elif "1000/ 1106 batches" in line:
        curr_epoch += 1
        train_ppl.append(float(numbers[13]))
# for line in open('finetune_log.txt'):
#     numbers = re.findall("[^a-zA-Z:](\d+[\.]?\d*)", line)
#     if "valid ppl" in line:
#         valid_ppl.append(float(numbers[3]))
#     elif "1000/ 1106 batches" in line:
#         train_ppl.append(float(numbers[6]))

epochs = range(len(train_ppl))
perplexity_train = train_ppl
perplexity_valid = valid_ppl
plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(epochs, perplexity_train, linewidth=1, color='blue', label='training')
plt.plot(epochs, perplexity_valid, linewidth=1, color='red', label='validation')
plt.errorbar(epoch_mc, valid_ppl_mc, valid_ppl_mc_error, linewidth=1, color='green', label='validation MC')
plt.ylim(0, 180)
plt.grid(True, which='both', axis='both')
plt.title('Penn Treebank - State-of-the-art Model\nwith Variational Weight Dropped LSTM')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig("./sota - variational weight drop.png", dpi=1200)
plt.show()

