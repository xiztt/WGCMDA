import numpy as np
import torch.utils.data as data_utils
from gensim.models import KeyedVectors
from collections import OrderedDict,defaultdict
import torch
import json

vocab_path = './data/privacy_dataset/out4.2_shuffle0_vocab.json'
vocab_file = open(vocab_path, 'r')
vocab = json.load(vocab_file, object_pairs_hook=OrderedDict)
vocab = dict(vocab)

#load_data function is to load the real train and test dataset.
def load_data(batch_size=64):
    X_tst = np.load("./data/privacy_dataset/X_test.npy")
    #print(X_tst.shape)
    X_trn = np.load("./data/privacy_dataset/X_train.npy")
    #print(X_trn.shape)
    Y_trn = np.load("./data/privacy_dataset/y_train.npy")
    #print(Y_trn.shape)
    Y_tst = np.load("./data/privacy_dataset/y_test.npy")
    #print(Y_tst.shape)
    label_embed = np.load("./data/privacy_dataset/label_embed.npy")
    embed=np.load("./data/privacy_dataset/w2v_embed_mr.npy")
    #embed = text.embedding.CustomEmbedding('./data/privacy_dataset/word_embed.txt')
    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader, label_embed, embed, X_tst, vocab, Y_tst, Y_trn

#load_eval_data function is to load train subset and validation subset for parameter tuning.
def load_eval_data(batch_size=64):
    X_trn = np.load("./data/privacy_dataset/X_train.npy")
    Y_trn = np.load("./data/privacy_dataset/y_train.npy")
    X_tst = X_trn[:1136]
    X_trn = X_trn[1136:]
    Y_tst = Y_trn[:1136]
    Y_trn= Y_trn[1136:]
    label_embed = np.load("./data/privacy_dataset/label_embed.npy")
    embed=np.load("./data/privacy_dataset/w2v_embed_mr.npy")
    #embed = text.embedding.CustomEmbedding('./data/privacy_dataset/word_embed.txt')
    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    return train_loader, test_loader, label_embed, embed, X_tst, vocab, Y_tst, Y_trn
