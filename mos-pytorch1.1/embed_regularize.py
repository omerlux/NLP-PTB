import numpy as np

import torch
import torch.nn.functional as F


def embedded_dropout(embed, words, eval, dropout=0.1, scale=None):
    # 2/11/20 - no scaling for dropout! - in evaluation it will be multiple by (1-dropout)
    if not eval:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight)  # / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:  # if it's evaluation (normal) - just multiple the weights by the dropout rate
        masked_embed_weight = embed.weight * (1 - dropout)
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(words, masked_embed_weight,
                    padding_idx, embed.max_norm, embed.norm_type,
                    embed.scale_grad_by_freq, embed.sparse
                    )
    return X


if __name__ == '__main__':
    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)

    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = torch.LongTensor(words)

    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
