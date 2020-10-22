import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("fit_20200919-175337.csv", delimiter=",", names=["num","time","epoch","trainl", "validl", "trainp", "validp"])
epochs = [int(x) for x in data['epoch'][1:]]
perplexity_train = data['trainp'][1:]
perplexity_valid = data['validp'][1:]
plt.rcParams['axes.facecolor'] = 'floralwhite'
plt.plot(epochs, perplexity_train, linewidth=1, color='blue', label='training')
plt.plot(epochs, perplexity_valid, linewidth=1, color='red', label='validation')
plt.ylim(0, 180)
plt.grid(True, which='both', axis='both')
plt.title('Penn Treebank - Basline Model')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.savefig("./sota - w finetune.png", dpi=1200)
plt.show()
plt.clf()

# m = min(10, epochs // 2)
# plt.rcParams['axes.facecolor'] = 'floralwhite'
# plt.plot(range(epochs)[m:], perplexity_train[m:], linewidth=1, color='blue', label='training')
# plt.plot(range(epochs)[m:], perplexity_valid[m:], linewidth=1, color='red', label='validation')
# plt.grid(True, which='both', axis='both')
# plt.title('Penn Treebank - Basline Model')
# plt.xlabel('Epochs')
# plt.ylabel('Perplexity')
# plt.legend()
# plt.savefig("./graph_zoomed.png")
# plt.show()