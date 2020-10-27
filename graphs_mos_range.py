import numpy as np
import matplotlib.pyplot as plt
import os
import re


mixtures_path = [path[0] for path in os.walk('./') if path[0].find('mixtures') != -1 and path[0].find('scripts') == -1]
valid_ppl = [[]] * len(mixtures_path)
train_ppl = [[]] * len(mixtures_path)
for i, path in enumerate(mixtures_path):
    valid_ppl[i] = []
    train_ppl[i] = []
    for line in open(os.path.join(path, 'log.txt')):
        numbers = re.findall("[^a-zA-Z:](\d+[\.]?\d*)", line)
        if "valid ppl" in line:
            valid_ppl[i].append(float(numbers[10]))
        elif "1000/ 1106 batches" in line:
            train_ppl[i].append(float(numbers[13]))
# for line in open('finetune_log.txt'):
#     numbers = re.findall("[^a-zA-Z:](\d+[\.]?\d*)", line)
#     if "valid ppl" in line:
#         valid_ppl.append(float(numbers[3]))
#     elif "1000/ 1106 batches" in line:
#         train_ppl.append(float(numbers[6]))

epochs = range(len(train_ppl[0]))

plt.rcParams['axes.facecolor'] = 'whitesmoke'
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/len(mixtures_path)) for i in range(len(mixtures_path))] # plt.cm.Spectral(np.linspace(0, 1, len(train_ppl)))
for j in range(len(mixtures_path)):
    index = (j+1) * 2 -1
    plt.plot(epochs, train_ppl[j], linewidth=1, linestyle='dotted', color=colors[j], label=str(index) + 't')
    plt.plot(epochs, valid_ppl[j], linewidth=1, linestyle='dashed', color=colors[j], label=str(index) + 'v')

plt.ylim(35, 150)
plt.grid(True, which='both', axis='both')
plt.title('Penn Treebank - Different Number of Mixtures')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig("./mixtures.png", dpi=2400)
plt.show()
plt.clf()
