import torch
import torch.nn as nn
from torch import optim
from model import seq2seq
from utils import *
from torch.autograd import Variable
import argparse
import numpy as np
PADDING = '<pad>'
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
USE_CUDA = torch.cuda.is_available()

VOCAB_PATH = './couplet/vocabs'
TRAIN_X_PATH = './couplet/train/in.txt'
TRAIN_Y_PATH = './couplet/train/out.txt'
TEST_X_PATH = './couplet/test/in.txt'
TEST_Y_PATH = './couplet/test/out.txt'
LR = 0.001
BATCH_SIZE = 64
EPOCH = 20
EMB_SIZE = 300
HIDDEN_SIZE = 1024
RNN_LAYERS = 3
TEACHER_FORCING_RATIO = 0.5
LOSSES_NUM = 100
MIN_LENGTH = 4


def load_vocab():
    char2idx = {PADDING: 0, UNK: 1, SOS: 2, EOS: 3}
    with open(VOCAB_PATH, 'r', encoding='utf-8') as fopen:
        for line in fopen.readlines():
            char = line.strip()
            if char not in char2idx.keys():
                char2idx[char] = len(char2idx)
    return char2idx


def load_dataset(char2idx, min_len = 4):
    test_x, test_y, train_x, train_y = [], [], [], []
    with open(TRAIN_X_PATH, 'r', encoding='utf-8') as fx, open(TRAIN_Y_PATH, 'r', encoding='utf-8') as fy:
        for line in fx.readlines():
            char_seq = line.strip().split()
            x = [char2idx[char] if char2idx.get(char) is not None else char2idx[UNK] for char in char_seq]
            x.append(char2idx[EOS])
            train_x.append(x)

        for line in fy.readlines():
            char_seq = line.strip().split()
            x = [char2idx[char] if char2idx.get(char) is not None else char2idx[UNK] for char in char_seq]
            x.append(char2idx[EOS])
            train_y.append(x)

    with open(TEST_X_PATH, 'r', encoding='utf-8') as fx, open(TEST_Y_PATH, 'r', encoding='utf-8') as fy:
        for line in fx.readlines():
            char_seq = line.strip().split()
            x = [char2idx[char] if char2idx.get(char) is not None else char2idx[UNK] for char in char_seq]
            x.append(char2idx[EOS])
            test_x.append(x)

        for line in fy.readlines():
            char_seq = line.strip().split()
            x = [char2idx[char] if char2idx.get(char) is not None else char2idx[UNK] for char in char_seq]
            x.append(char2idx[EOS])
            test_y.append(x)
    train_data = list(zip(train_x, train_y))
    #test_data = train_data
    test_data = list(zip(test_x, test_y))
    return train_data, test_data

def main():
    char2idx = load_vocab()
    train_data, test_data = load_dataset(char2idx)

    model = seq2seq(len(char2idx), EMB_SIZE, HIDDEN_SIZE, 0.1, RNN_LAYERS, TEACHER_FORCING_RATIO)
    #model = torch.load('./model20.pkl')
    if USE_CUDA:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    losses = []
    for epoch in range(EPOCH):
        for _, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
            batch_x, batch_y = pad_to_batch(batch, char2idx[PADDING])
            batch_x, batch_y = Variable(torch.LongTensor(batch_x)), Variable(torch.LongTensor(batch_y))
            start_decode = Variable(torch.LongTensor([[char2idx[SOS]]]*batch_x.size(0)))

            if USE_CUDA:
                batch_x, batch_y, start_decode = batch_x.cuda(), batch_y.cuda(), start_decode.cuda()

            preds = model(batch_x, start_decode, batch_y.size(1), batch_y)

            loss = loss_function(preds, batch_y.view(-1))
            losses.append(loss.data.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if len(losses) == LOSSES_NUM:
                print('loss:', np.mean(losses))
                losses = []


        # Eval
        bleu_scores = []
        for _, batch in enumerate(getBatch(BATCH_SIZE, test_data)):
            test_x, test_y = pad_to_batch(batch, char2idx[PADDING])
            test_x, test_y = Variable(torch.LongTensor(test_x)), Variable(torch.LongTensor(test_y))
            start_decode = Variable(torch.LongTensor([[char2idx[SOS]]] * test_x.size(0)))
            if USE_CUDA:
                test_x, test_y, start_decode = test_x.cuda(), test_y.cuda(), start_decode.cuda()

            #output, hidden = encoder(test_x)
            #preds = decoder(start_decode, hidden, test_y.size(1), output)
            preds = model(test_x, start_decode, test_y.size(1), None, False)
            preds = torch.max(preds, 1)[1].view(test_y.size(0), test_y.size(1))
            bleu_scores.append(cal_bleu(preds, test_y, char2idx[EOS]))

        print('Epoch.', epoch, ':mean_bleu_score:', np.mean(bleu_scores))
        print(bleu_scores)

        torch.save(model, './model/model'+str(epoch+1)+'.pkl')

    torch.save(model, './model/model.pkl')


if __name__ == '__main__':
    main()
