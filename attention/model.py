import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math

class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features)).to(torch.double)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input=input.to(torch.double)
        input=input.to("cpu")
        support = torch.matmul(input, self.weight)
        support=support.cuda()
        adj.cuda()
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path


class StructuredSelfAttention(torch.nn.Module):

    def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, label_embed, embeddings,A_matrix):
        super(StructuredSelfAttention, self).__init__()
        self.n_classes = n_classes
        self.A_matrix = A_matrix
        self.embeddings = self._load_embeddings(embeddings)
        self.label_embed = self.load_labelembedd(label_embed)
        self.lstm = torch.nn.LSTM(300, hidden_size=lstm_hid_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, n_classes)

        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight3 = torch.nn.Linear(n_classes, 1)
        self.weight4 = torch.nn.Linear(n_classes, 1)

        self.output_layer = torch.nn.Linear(lstm_hid_dim*2, n_classes)
        #self.embedding_dropout = torch.nn.Dropout(p=0.3)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim
        self.gcn_linear_0 = torch.nn.Linear(300, 600)
        self.gcn_linear_1=torch.nn.Linear(300,450)
        self.gcn_linear_2=torch.nn.Linear(450,600)
        self.batch_norm1=torch.nn.BatchNorm1d(450)
        self.batch_norm2=torch.nn.BatchNorm1d(600)
        self.batch_norm3 = torch.nn.BatchNorm1d(32)
        self.batch_norm4 = torch.nn.BatchNorm1d(32)
        self.gc1 = GraphConvolution(300, 450)
        self.gc2 = GraphConvolution(450, 600)
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.linear1=torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.att_l1 = torch.nn.Linear(n_classes, 1)
        self.att_l2 = torch.nn.Linear(n_classes, 1)

        #GCN

    def _load_embeddings(self, embeddings):

        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings
    def load_labelembedd(self, label_embed):

        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self):
        return (torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda(),torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda())

    def graph_convolution(self,input1, in_dim, out_dim, A_matrix):
        tmp=torch.mm(A_matrix.cuda(),input1.cuda())
        tmp=tmp.to(torch.float32)
        if(in_dim==300 and out_dim==600):
            mul_result = self.gcn_linear_0(tmp)
        else:
            if(in_dim==300 and out_dim==450):
                mul_result=self.gcn_linear_1(tmp)
            else:
                if(in_dim==450):
                    mul_result=self.gcn_linear_2(tmp)
                else: print("wrong")
        output=F.leaky_relu(mul_result)
        return output

    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)

        # step1 get LSTM outputs
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.lstm(embeddings, hidden_state)

        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)

        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        label = self.label_embed.weight.data
        m1 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)

        # step4 feature fusion assisted by GCN
        x = self.gc1(label.to(torch.double).cuda(), self.A_matrix.cuda())
        x = self.relu1(x)
        gcn_out = self.gc2(x, self.A_matrix.cuda())
        gcn_out=self.relu2(gcn_out)
        gcn_out = gcn_out.expand(self.batch_size, 32, 600)
        gcn_out = gcn_out.to(torch.float)
        re_label_att = torch.bmm(gcn_out, label_att.transpose(1, 2))
        re_label_att = self.att_l1(re_label_att)
        re_label_att = torch.sigmoid(re_label_att)
        re_self_att = torch.bmm(gcn_out, self_att.transpose(1, 2))
        re_self_att = self.att_l2(re_self_att)
        re_self_att = torch.sigmoid(re_self_att)
        re_label_att = re_label_att / (re_self_att + re_label_att)
        re_self_att = 1 - re_label_att
        doc = re_self_att * label_att + re_label_att * self_att
        avg_sentence_embeddings = torch.sum(doc, 1) / self.n_classes

        pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))

        return pred