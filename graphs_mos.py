import numpy as np
import matplotlib.pyplot as plt
import re

valid_ppl = []
train_ppl = []
for line in open('log.txt'):
    numbers = re.findall("[^a-zA-Z:](\d+[\.]?\d*)", line)
    if "valid ppl" in line:
        valid_ppl.append(float(numbers[10]))
    elif "1000/ 1106 batches" in line:
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
plt.ylim(0, 180)
plt.grid(True, which='both', axis='both')
plt.title('Penn Treebank - State-of-the-art Model')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig("./sota - w finetune.png", dpi=1200)
plt.show()
plt.clf()
