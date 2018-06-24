import torch.nn as nn
import torch.nn.functional as F
import torch
import random


class seq2seq(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, dropout_rate, num_layers, tf_ratio,bidirectional=False):
        super(seq2seq, self).__init__()
        self.hidden_size = hidden_size
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1

        self.embedding_layer = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.encoder = nn.GRU(emb_size, hidden_size,
                              batch_first=True,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        self.decoder = nn.GRU(emb_size + hidden_size * self.n_directions, hidden_size,
                              batch_first=True,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = nn.Linear(self.hidden_size * self.n_directions, self.hidden_size * self.n_directions)
        self.n_layers = num_layers
        self.tf_ratio = tf_ratio

        self.init_weight()

    def forward(self, x, start_decode, max_len, y, is_training=True):
        if y is not None:
            y_t = y.transpose(0, 1)
        embedded_x = self.embedding_layer(x)
        encoder_outputs, hidden = self.encoder(embedded_x)
        hidden_layer = hidden[-self.n_directions:]
        context = torch.cat([h for h in hidden_layer], 1).unsqueeze(1) # context:(B,1,D)
        decoded_x = self.embedding_layer(start_decode)
        decode = []
        for i in range(max_len):
            rnn_input = torch.cat((decoded_x, context), 2)
            ########
            rnn_output, hidden = self.decoder(rnn_input, hidden)

            concated = torch.cat((rnn_output, context), 2)
            score = self.fc(concated.squeeze(1))
            softmaxed = F.log_softmax(score, 1)
            decode.append(score)
            decoded = softmaxed.max(1)[1]
            decoded_x = self.embedding_layer(decoded).unsqueeze(1)
            USE_TF = random.random() < self.tf_ratio
            if is_training and USE_TF:
                decoded_x = self.embedding_layer(y_t[i]).unsqueeze(1)
            context, alpha = self.Attention(rnn_output, encoder_outputs)
        scores = torch.cat(decode, 1)
        return scores.view(x.size(0) * max_len, -1)

    def Attention(self, hidden, encoder_outputs):
        # hidden : B,1,D
        # encoder_outputs : B,L,D
        # B:batch_size, L:Seq_Length(max_len in batch), D:hidden_dim

        # output context:(B,1,D) alpha:(B,1,L)
        hidden = hidden.squeeze(1).unsqueeze(2)  # (B,1,D) -> (B,D,1)
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        temp = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        temp = temp.view(batch_size, max_len, -1)  # (B*L,D)->(B,L,D)
        alpha = torch.bmm(temp, hidden)  #(B,L,D) * (B,D,1) -> (B,L,1)
        alpha = F.softmax(alpha.squeeze(2), 1)
        alpha = alpha.unsqueeze(1) # alpha: (B,1,L)
        context = torch.bmm(alpha, encoder_outputs) #(B,1,L) * (B,L,D) -> (B,1,D)
        return context, alpha

    def init_weight(self):
        self.embedding_layer.weight = nn.init.xavier_uniform_(self.embedding_layer.weight)
        self.attn.weight = nn.init.xavier_uniform_(self.attn.weight)
        self.encoder.weight_hh_l0 = nn.init.xavier_uniform_(self.encoder.weight_hh_l0)
        self.encoder.weight_ih_l0 = nn.init.xavier_uniform_(self.encoder.weight_ih_l0)
        self.decoder.weight_hh_l0 = nn.init.xavier_uniform_(self.decoder.weight_hh_l0)
        self.decoder.weight_ih_l0 = nn.init.xavier_uniform_(self.decoder.weight_ih_l0)
