import math as m
from Bio import SeqIO
import random
import torch as tr
import numpy as np

from .aux import gen_code, gen_code2

def data_load(length, step, filename):
    """Read fasta file."""
    seq_record = SeqIO.read(filename, 'fasta')
    n_seq = m.floor((len(seq_record)-length)/step)+1


    data_fow = [seq_record[0:length]]*n_seq
    data_rev = [seq_record[0:length]]*n_seq

    for i in range(n_seq):
        data_fow[i] = seq_record[(i)*(step):(i)*(step)+length]
        data_fow[i].id = seq_record.id + "-" + str(i * step) + '-' + \
                         str(i * step + length)
        data_rev[i] = data_fow[i].reverse_complement()
        data_rev[i].id = seq_record.id + ' rev pos:' + str(i * step) + '-' +\
                         str(i * step + length)

    data = data_fow + data_rev

    del data_fow, data_rev

    return data

def load_seq_struct_mfe(fname, length=100):

    seq_fasta = list(SeqIO.parse(fname, "fasta"))
    seq, struct, mfe = [], [], []
    seq_length = []
    for k, s in enumerate(seq_fasta):

        strcode = str(s.seq).replace("\n", "")
        ind = strcode.rfind("(")
        mfe_norm = np.abs(float(strcode[ind:].strip("()"))) / (ind//2)
        mfe.append(mfe_norm)

        seq_code = gen_code(strcode[:ind//2])
        seq.append(seq_code[:length])

        struct_code = gen_code2(strcode[ind//2:ind])
        struct.append(struct_code[:length])

        seq_length.append(min(len(seq_code), length))

    return seq, struct, tr.tensor(mfe)


def load_fasta(fname, length):

    seq = list(SeqIO.parse(fname, "fasta"))
    seq = _seq2code(seq, length)
    return seq

def load_train_valid_data(pos_fname, neg_fname, valid_pos_fname,
                          valid_neg_fname, length):


    pos = load_fasta(pos_fname, length=length)
    neg = load_fasta(neg_fname, length=length)

    split = .8

    if valid_pos_fname is None:
        # Split train partition.
        random.shuffle(pos)
        L = int(split * len(pos))
        train_pos = pos[:L]
        valid_pos = pos[L:]
    else:
        train_pos = pos
        valid_pos = load_fasta(valid_pos_fname, length=length)

    if valid_neg_fname is None:
        random.shuffle(neg)
        L = int(split * len(neg))
        train_neg = neg[:L]
        valid_neg = neg[L:]
    else:
        train_neg = neg
        valid_neg = load_fasta(valid_neg_fname, length=length)

    train_seq = train_pos + train_neg
    valid_seq = valid_pos + valid_neg

    # Labels in the format [N x 2], where the first column is the positive
    # class
    train_label = tr.zeros((len(train_seq), 2))
    train_label[:len(train_pos), 0] = 1
    train_label[len(train_pos):, 1] = 1

    valid_label = tr.zeros((len(valid_seq), 2))
    valid_label[:len(valid_pos), 0] = 1
    valid_label[len(valid_pos):, 1] = 1

    return train_seq, train_label, valid_seq, valid_label


def _seq2code(seq, length):
    """ Converts sequence to numeric coding"""
    out = []
    for i in range(len(seq)):
        out.append(gen_code(seq[i].seq[:length]))
    return out