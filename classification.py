from attention.model import StructuredSelfAttention
from attention.train import train
import torch
import data.utils as utils
import data_got_privacy
import numpy as np
config = utils.read_config("config.yml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#if config.GPU:
    #torch.cuda.set_device(3)
print('loading data...\n')
label_num = 32

#put the training subset and the validation subset here for parameter selecting.
#Or put the training set and the testing set here for the model test.
train_loader, test_loader, label_embed,embed,X_tst,word_to_id,Y_tst,Y_trn = data_got_privacy.load_data(batch_size=64)

label_embed = torch.from_numpy(label_embed).float()

embed = torch.from_numpy(embed).float()
print("load done")

def multilabel_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    train(attention_model, train_loader, test_loader, loss, opt, epochs,GPU)
A_matrix = np.load("attention/A_matrix_1.npy")
A_matrix=torch.from_numpy(A_matrix)
attention_model = StructuredSelfAttention(batch_size=config.batch_size, lstm_hid_dim=config['lstm_hidden_dimension'],
                                          d_a=config["d_a"], n_classes=label_num, label_embed=label_embed,embeddings=embed,A_matrix=A_matrix)
attention_model=attention_model.cuda()
multilabel_classification(attention_model, train_loader, test_loader, epochs=config["epochs"])
