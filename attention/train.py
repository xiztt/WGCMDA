import numpy as np
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_name='./attention/models_saved'
def train(attention_model,train_loader,test_loader,criterion,opt,epochs = 5,GPU=True,testing=False):
    if(testing==True):
        attention_model.load_state_dict(torch.load(path_name+'/model_best.pth'))
        attention_model.eval()
        if torch.cuda.is_available():
            attention_model.cuda()
        test_acc_k = []
        test_loss = []
        test_s_k = []
        test_avgpre_k=[]
        test_one_err_k=[]
        attention_model.eval()
        labelss=[]
        predss=[]
        for batch_idx, test in enumerate(tqdm(test_loader)):
            x, y = test[0].cuda(), test[1].cuda()
            # x,y=test[0],test[1]
            val_y = attention_model(x)
            loss = criterion(val_y, y.float()) / train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            for i in labels_cpu.numpy():
                labelss.append(i)
            pred_cpu = val_y.data.cpu()
            for j in pred_cpu.numpy():
                predss.append(j)
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_acc_k.append(prec)
            ndcg = s_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            test_s_k.append(ndcg)
            test_loss.append(float(loss))
        Y=np.array(labelss)
        FY=np.array(predss)
        train_total = np.size(Y, 0)
        fake_index = np.arange(train_total).reshape(train_total, 1)
        fake_gnd = np.hstack((fake_index, Y))
        fake_pred = np.hstack((fake_index, FY))

        Avg_prec = avgprec(fake_gnd, fake_pred)
        one_err = one_error(fake_gnd, fake_pred)
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_s = np.array(test_s_k).mean(axis=0)

        print("avg_pre : %.4f , one_err : %.4f , precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
            Avg_prec,one_err, test_prec[0], test_prec[2], test_prec[4]))
        print("s@1 : %.4f , s@3 : %.4f , s@5 : %.4f " % (test_s[0], test_s[2], test_s[4]))


        return
    if torch.cuda.is_available():
        attention_model.cuda()
    for i in range(epochs):
        attention_model.train()
        labelss = []
        predss = []
        print("Running EPOCH",i+1)
        train_loss = []
        prec_k = []
        sk = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train[0].cuda(), train[1].cuda()
            #x, y = train[0],train[1]
            y_pred= attention_model(x)
            loss = criterion(y_pred, y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            labels_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            for k in labels_cpu.numpy():
                labelss.append(k)
            pred_cpu = y_pred.data.cpu()
            for j in pred_cpu.numpy():
                predss.append(j)
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = s_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            sk.append(ndcg)
            train_loss.append(float(loss))
        Y = np.array(labelss)
        FY = np.array(predss)
        train_total = np.size(Y, 0)
        fake_index = np.arange(train_total).reshape(train_total, 1)
        fake_gnd = np.hstack((fake_index, Y))
        fake_pred = np.hstack((fake_index, FY))
        Avg_prec = avgprec(fake_gnd, fake_pred)
        one_err = one_error(fake_gnd, fake_pred)
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_s = np.array(sk).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.6f" % (i+1, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("s@1 : %.4f , s@3 : %.4f , s@5 : %.4f " % (epoch_s[0], epoch_s[2], epoch_s[4]))

        torch.save(attention_model.state_dict(),'./attention/models_saved/epoch_'+str(i+1)+'.pth')


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)


        mat = np.multiply(score_mat, true_mat)

        num = np.sum(mat, axis=1)
        #print(num,k)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def s_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)


        mat = np.multiply(score_mat, true_mat)

        num=0
        for j in mat:
            if np.sum(j)!=0:
                num=num+1

        p[k] = np.mean(num / (k + 1))
        p[k]=num/true_mat.shape[0]
    return np.around(p, decimals=4)


def avgprec(gnd, pred):
    gnd_id = gnd[:, 0]
    gnd = gnd[:, 1:]

    pred_id = pred[:, 0]
    pred_id = np.array(pred_id, dtype=int)
    pred = pred[:, 1:]

    num_tweets = np.size(gnd, 0)
    num_properties = np.size(gnd, 1)

    prec = np.zeros(num_tweets)

    for i in range(num_tweets):

        cur_pred_id = i
        ind = np.argsort(-pred[cur_pred_id])
        gnd = np.array(gnd, int)
        Y = np.sum(gnd[i])
        exist_pro_index = np.where(gnd[i] == 1)[0]
        rank = np.zeros(num_properties)
        for j in exist_pro_index:
            rank[j] = np.where(ind == j)[0][0] + 1
        ind_rank = np.where(rank != 0)[0]
        sum_rank = 0
        for j in ind_rank:
            sum_rank += np.size(np.where((rank <= rank[j]) & (rank > 0))[0]) / rank[j]
        if Y == 0:
            prec[i] = 0
        else:
            prec[i] = sum_rank / Y
        if not (prec[i] >= 0):
            print(i, sum_rank, Y, gnd[i])
    tot = 0
    for i, num in enumerate(prec):
        tot += num

    ans = tot / num_tweets

    return ans

def one_error(gnd, pred):
    gnd = gnd[:, 1:]

    pred_id = pred[:, 1]
    pred = pred[:, 1:]

    num_tweets = np.size(gnd, 0)

    sum1 = 0
    for i in range(num_tweets):
        cur_pred_id = i
        ind = np.argmax(pred[cur_pred_id])
        if gnd[i,ind] == 0:
            sum1 += 1
    one_error = float(sum1) / float(num_tweets)

    return one_error

def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)