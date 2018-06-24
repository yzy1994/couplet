import unicodedata
import string
import re
import random
from nltk.translate.bleu_score import corpus_bleu
letter2idx = {'<PAD>': 0}
all_letters = string.ascii_letters + " .,;'"
for _, letter in enumerate(all_letters):
    letter2idx[letter] = len(letter2idx)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(s) for s in lines]


def line2Tensor(line):
    letters = []
    for _, letter in enumerate(line):
        letters.append(letter2idx[letter])
    return letters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return idxs


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def pad_to_batch(batch, pad_idx):
    x, y = list(zip(*batch))
    max_x = max([len(s) for s in x])
    max_y = max([len(s) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        while len(x[i]) < max_x:
            x[i].append(pad_idx)
        x_p.append(x[i])
        while len(y[i]) < max_y:
            y[i].append(pad_idx)
        y_p.append(y[i])
    return x_p, y_p


def cal_bleu(preds, targets, EOSIDX):
    references_list, hypotheses_list = [], []
    for i in range(targets.size(0)):
        reference = []
        for j in range(targets.size(1)):
            reference.append(str(targets[i][j].item()))
            if targets[i][j].item() == EOSIDX:
                break
        references_list.append([reference])

    for i in range(preds.size(0)):
        hypothesis = []
        for j in range(preds.size(1)):
            hypothesis.append(str(preds[i][j].item()))
            if preds[i][j].item() == EOSIDX:
                break
        hypotheses_list.append(hypothesis)
    bleu = corpus_bleu(references_list, hypotheses_list)
    if bleu>0.1:
        print(references_list)
        print(hypotheses_list)
    return bleu