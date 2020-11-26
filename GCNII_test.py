import os, pickle
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, precision_recall_curve


# path
project_home = "/data2/users/yuanqm/PPI"
Dataset_Path = project_home + "/dataset/"
Feature_Path = project_home + "/feature/"
Model_Path = '/home/chenjw48/yuanqm/PPI/Model/GCNII_ablation/SOTA/'

# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 50
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2 # [not bind, bind]

# GCNII parameters
LAYER = 8
INPUT_DIM = 54
HIDDEN_DIM = 256
DROPOUT = 0.1
ALPHA = 0.5
LAMBDA = 1.5
VARIANT = True
MAP_CUTOFF = 14


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark=True


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def load_features(sequence_name):
    # L * 45
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hhm_feature = np.load(Feature_Path + "hhm/" + sequence_name + '.npy')
    dssp_feature = np.load(Feature_Path + "dssp_ASA/" + sequence_name + '.npy')
    feature_matrix = np.concatenate([pssm_feature, hhm_feature, dssp_feature], axis = 1).astype(np.float32)
    return feature_matrix


def load_graph(sequence_name):
    matrix = np.load(Feature_Path + 'distance_map_{}/'.format(MAP_CUTOFF) + sequence_name + '.npy').astype(np.float32)
    matrix = normalize(matrix)
    return matrix


class ProDataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        # L * 45
        sequence_feature = load_features(sequence_name)
        # L * L
        sequence_graph = load_graph(sequence_name)

        return sequence_name, sequence, label, sequence_feature, sequence_graph

    def __len__(self):
        return len(self.labels)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gcnii = GCNIIppi(nfeat = INPUT_DIM, nlayers = LAYER, nhidden = HIDDEN_DIM,
                              nclass = NUM_CLASSES, dropout = DROPOUT, lamda = LAMBDA,
                              alpha = ALPHA, variant = VARIANT)
        self.criterion = nn.CrossEntropyLoss() # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    def forward(self, x, adj):                      # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        output = self.gcnii(x, adj)                 # output.shape = (seq_len, NUM_CLASSES)
        return output


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    valid_name = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, sequence_features, sequence_graphs = data

            if torch.cuda.is_available():
                features = Variable(sequence_features.cuda())
                graphs = Variable(sequence_graphs.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_features)
                graphs = Variable(sequence_graphs)
                y_true = Variable(labels)

            y_true = torch.squeeze(y_true)
            features = torch.squeeze(features)
            graphs = torch.squeeze(graphs)

            y_pred = model(features, graphs)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            valid_name.append(sequence_names[0])

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, valid_name


def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    test_result = {}

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = Model()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name,map_location='cuda:6'))

        epoch_loss_test_avg, test_true, test_pred, test_name = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        print("\n========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Test sensitivity: ", result_test['sensitivity'])
        print("Test specificity: ", result_test['specificity'])

        test_result[model_name] = [
            epoch_loss_test_avg,
            result_test['binary_acc'],
            result_test['precision'],
            result_test['recall'],
            result_test['f1'],
            result_test['AUC'],
            result_test['AUPRC'],
            result_test['mcc'],
            result_test['sensitivity'],
            result_test['specificity'],
        ]
        
        # export prediction
        with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
            pickle.dump([test_true, test_pred], f)


def analysis(y_true, y_pred):
    best_f1 = 0
    best_threshold = 0
    for threshold in range(0, 100):
        threshold = threshold / 100
        binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
        binary_true = y_true
        f1 = metrics.f1_score(binary_true, binary_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true
    
    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = precision_recall_curve(binary_true, y_pred)
    AUPRC = auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitivity = 1.0 * TP / (TP + FN)
    specificity = 1.0 * TN / (FP + TN)

    result = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    return result


if __name__ == "__main__":
    with open(Dataset_Path + "PPI_data.pkl", "rb") as f:
        PPI_data = pickle.load(f)
    del PPI_data["train"]["1k74B"]

    IDs = []
    sequences = []
    labels = []

    for ID in PPI_data["test"]:
        item = PPI_data["test"][ID]
        IDs.append(ID)
        sequences.append(item[0])
        labels.append(item[1])

    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe)
