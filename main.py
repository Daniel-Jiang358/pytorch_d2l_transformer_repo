import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

class PositionWiseFFn(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **Kwargs):
        super(PositionWiseFFn, self).__init__(**Kwargs)
        self.dense1=nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **Kwargs):
        super(AddNorm, self).__init__(**Kwargs)
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(self.dropout(Y)+X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_inputs, ffn_num_hiddens, num_heads
                 , dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention=d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads,
                                              dropout, use_bias)
        self.addnorm1=AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFn(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm2=AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y=self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_inputs,
                 ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **Kwargs):
        super(TransformerEncoder, self).__init__(**Kwargs)
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding=d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                              ffn_num_inputs, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X=self.pos_embedding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights=[None]*len(self.blks)
        for i, blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights

        return X


class DecoderBlock(nn.Module):
    """解码器第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens
                 , num_heads, dropout, i, **Kwargs):
        super(DecoderBlock, self).__init__(**Kwargs)
        self.i=i
        self.attention1=d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1=AddNorm(norm_shape, dropout)
        self.attention2=d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2=AddNorm(norm_shape, dropout)
        self.ffn=PositionWiseFFn(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3=AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens=state[0], state[1]
        batch_size, num_steps, _=X.shape
        #dec_valid_lens的开头:(batch_size, num_steps),
        #其中每一行时[1,2,..., num_steps]
        dec_valid_lens=torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)

        #zizhuyili
        X2=self.attention1(X, X, X, dec_valid_lens)
        Y=self.addnorm1(X, X2)
        #encoder-decoder self-attention
        #enc_outputs的开头,(batch_size, num-steps, num_hiddens)
        Y2=self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z=self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding=d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks=nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                    ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense=nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        self.seqX=None
        return [enc_outputs, enc_valid_lens]

    def forward(self, X, state):
        if not self.training:
                self.seqX=X if self.seqX is None else torch.cat((self.seqX, X), dim=1)
                X=self.seqX

        X=self.pos_embedding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights=[[None]*len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state=blk(X,state)
            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i]=blk.attention2.attention.attention_weights

        if not self.training:
            return self.dense(X)[:,-1:,:],state

        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


num_hiddens, num_layers, dropout, batch_size, num_steps=32, 2, 0.1, 64, 10
lr, num_epochs, device=0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads=32, 64, 4
key_size, query_size, value_size=32, 32, 32
norm_shape=[32]

train_itr, src_vocab, tgt_vocab=d2l.load_data_nmt(batch_size, num_steps)

encoder=TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout)
decoder=TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    num_heads, num_layers, dropout)
net=d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_itr, lr, num_epochs, tgt_vocab, device)


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')



