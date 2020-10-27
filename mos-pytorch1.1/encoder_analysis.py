import argparse
import os, sys
import pickle
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gc

import data
import model
import time
import hashlib
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

# Gap Statistics function
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        logging.info('| starting {} clusters'.format(k))

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=(ntokens, args.emsize)) #data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': int(k), 'gap': gap}, ignore_index=True)

        logging.info('| cluster count: {} | gap: {}'.format(k, gap))

    return gaps.argmax() + 1, resultsdf  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Language Model')
parser.add_argument('--data', type=str, default='../../data/ptb/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=-1,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--load', action='store_true',
                    help='use word_vectors.pkl instead of creating it')
parser.add_argument('--n_experts', type=int, default=10,
                    help='number of experts')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--gpu_device', type=str, default="0",
                    help='specific use of gpu')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
# --- pytorch 1.2 warning spam ignoring ---
import warnings

warnings.filterwarnings('ignore')

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py'])

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'encoder_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
logging.info('Args: {}'.format(args))

if not args.load:
    if args.continue_train:
        model = torch.load(os.path.join(args.save, 'model.pt'))
    else:
        assert False, 'You must load a model first!'

    if args.cuda:
        if args.single_gpu:
            parallel_model = model.cuda()
        else:
            parallel_model = nn.DataParallel(model, dim=1).cuda()
    else:
        parallel_model = model

    total_params = sum(x.data.nelement() for x in model.parameters())
    logging.info('| Model total parameters: {}'.format(total_params))

try:
    if args.continue_train:
        if not args.load:
            logging.info('| Getting embedding layer from the model {}'.format(args.save))

            encoder_only = model.encoder
            device = 'cuda'
            encoder_only.weight = nn.Parameter(encoder_only.weight.to(device))  # Moving the weights of the embedding layer to the GPU

            logging.info('| vocabulary size: {:3d} | embedding layer size: {:3d} |'.format(ntokens, encoder_only.embedding_dim))
            logging.info('| Getting all vectors...')

            word_vectors = dict()
            for i in range(ntokens):
                input = torch.LongTensor([i])
                input = input.to(device)
                output = encoder_only(input)
                word_vectors[corpus.dictionary.idx2word[i]] = output

            logging.info('| Saving word_vectors.pkl dictionary ...')
            a_file = open(os.path.join(args.save, "word_vectors.pkl"), "wb")

            pickle.dump(word_vectors, a_file)
            a_file.close()
            logging.info('=== run this code without --load to activate Gap statistics ===')
        else:
            logging.info('| Loading embedding vectors dictionary ...')
            logging.info('| vocabulary size: {:3d} | embedding layer size: {:3d} |'.format(ntokens, args.emsize))

            a_file = open(os.path.join(args.save, "word_vectors.pkl"), "rb")
            word_vectors = pickle.load(a_file)

            # Gap Statistics to measure perfect K:
            data_vec = np.array([tensor.cpu().detach().numpy() for tensor in word_vectors.values()])
            data_vec = data_vec.reshape(ntokens, -1)
            k, gapdf = optimalK(data_vec, nrefs=10, maxClusters=100)
            logging.info('| Optimal k is: ' + str(int(k)))
            # Show the results of the calculated gaps, the higher the value, the more optimal it is
            plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
            plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
            plt.grid(True)
            plt.xlabel('Cluster Count')
            plt.ylabel('Gap Value')
            plt.title('Gap Values by Cluster Count\nPTB Embedding Vectors (After Train)')
            plt.savefig(os.path.join(args.save, 'Gap Values.png'), dpi=1200)
            plt.show()


except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting early')

