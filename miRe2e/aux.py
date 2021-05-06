import torch
import numpy as np
import random as rn
import torch as tr
import math


def gen_code(seq):
    seq2num = {'n': 0, 'a': 1, 'c': 2, 'g': 3, 'u': 4, 't': 4, 'N': 0,
               'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}
    x = tr.LongTensor([seq2num[c] for c in seq])
    return x


def gen_code2(seq):
    seq2num = {'(': 1, ')': 2, '.': 0}
    x = tr.LongTensor([seq2num[c] for c in seq])
    return x


def make_weights_for_balanced_classes(x):
    count = [0] * 2
    for item in range(len(x)):
        if x[item] == 1:
            count[0] += 1
        else:
            count[1] += 1
    weight_per_class = [0.] * 2
    N = float(sum(count))
    for i in range(2):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(x)
    for i in range(len(x)):
        weight[i] = weight_per_class[1 - int(x[i])]
    return weight


def test_proportion(seq_pos, seq_neg, str_pos, str_neg, test_proportion, im,
                    center, augmentation):
    n_neg = int(len(seq_neg)/im)
    n_pos = len(seq_pos)

    test_neg_idx = [x + n_pos for x in rn.sample(range(n_neg),
                                                 int(test_proportion * n_neg))]
    train_neg_idx = [x for x in list(set(list(range(n_pos, n_pos + n_neg)))
                                     - set(test_neg_idx))]

    if center == False:
        test_pos = rn.sample(range(1762), int(test_proportion * 1762))

        test_pos_idx = []
        for i in test_pos:
            test_pos_idx.append(i * augmentation)
            for k in range(augmentation)[1:]:
                test_pos_idx.append(i * augmentation + k)

        train_pos_idx = list(set(list(range(n_pos))) - set(test_pos_idx))

    else:
        test_pos = rn.sample(range(n_pos), int(test_proportion * n_pos))
        train_pos = list(set(list(range(n_pos))) - set(test_pos))

        test_pos_idx = []
        for i in test_pos:
            test_pos_idx.append(i)

        train_pos_idx = []
        for i in train_pos:
            train_pos_idx.append(i)

    pos_t = torch.ones(n_pos, 1)
    neg_t = torch.zeros(n_neg, 1)
    tag1 = torch.cat((pos_t, neg_t))
    neg_t = torch.ones(n_neg, 1)
    pos_t = torch.zeros(n_pos, 1)
    tag2 = torch.cat((pos_t, neg_t))
    tag = torch.cat((tag1, tag2), 1)

    return ((seq_pos + seq_neg), (str_pos + str_neg), (train_pos_idx +
                                                       train_neg_idx),
            (test_pos_idx + test_neg_idx), tag)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_pe(seq_lens, max_len):
    num_seq = seq_lens.shape[0]
    pos_i_abs = torch.Tensor(np.arange(1,max_len+1)).view(1, 
        -1, 1).expand(num_seq, -1, -1).double()
    pos_i_rel = torch.Tensor(np.arange(1,max_len+1)).view(1, -1).expand(num_seq, -1)
    pos_i_rel = pos_i_rel.double()/seq_lens.view(-1, 1).double()
    pos_i_rel = pos_i_rel.unsqueeze(-1)
    pos = torch.cat([pos_i_abs, pos_i_rel], -1)

    PE_element_list = list()
    # 1/x, 1/x^2
    PE_element_list.append(pos)
    PE_element_list.append(1.0/pos_i_abs)
    PE_element_list.append(1.0/torch.pow(pos_i_abs, 2))

    # sin(nx)
    for n in range(1, 50):
        PE_element_list.append(torch.sin(n*pos))

    # poly
    for i in range(2, 5):
        PE_element_list.append(torch.pow(pos_i_rel, i))

    for i in range(3):
        gaussian_base = torch.exp(-torch.pow(pos, 
            2))*math.sqrt(math.pow(2,i)/math.factorial(i))*torch.pow(pos, i)
        PE_element_list.append(gaussian_base)

    PE = torch.cat(PE_element_list, -1)
    for i in range(num_seq):
        PE[i, seq_lens[i]:, :] = 0
    return PE
