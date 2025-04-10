
'''

= = = WORK IN PROGRESS = = = 

TO DO: My probes should have the same class structure as CCS!

'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import copy
import torch as torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import pickle

'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probes as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

random.seed(42)

class LogReg():

    def __init__(self):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()

    def fit(data, gold, self):
        self.probe.fit(data, gold)

    def cross_validation(self, X, y, cv=5, scoring='accuracy'):             # To test
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.fit(), X, y, cv=kf, scoring=scoring)
        return scores

    def predict(data, self):
        self.probe.predict(data)

    def save(llm, layer, self):
        with open(f'logreg_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self, file)

class Mmp():

    def __init__(self, layers):
        self.layers = [x for x in range(layers)]
        self.data = None
        self.gold = None
        self.probe = LogisticRegression()
        self.lda = LDA(n_components=2)

    def cross_validation(self, X, y, cv=5, scoring='accuracy'):             # To test
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.fit(), X, y, cv=kf, scoring=scoring)
        return scores

    def fit(data, gold, self):
        data = self.lda.fit_transform(data, gold)
        self.probe.fit(data, gold)

    def predict(data, self):
        data = self.lda.transform(data)
        self.probe.predict(data)

    def save(llm, layer, self):
        with open(f'mmp_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self.probe, file)

class Neural(nn.Module):

    # Init

    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=128, hidden_dim3=64, output_dim=1, threshold=0.5):
        super(Neural, self).__init__()

        # Architecture
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.criterion = nn.BCELoss()

        # Hyperparameters

        self._initialize_weights()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = Adam()
        self.threshold = threshold
        self.best = None

        # Data
        self.input_dim = input_dim
        self.llm = "Default"

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    # The following two functions will be useful to pickle the probes with meaningful filenames

    def set_layers(new_layers, self):
        self.layers = new_layers

    def set_llm(new_llm, self):
        self.llm = new_llm

    def forward(self, data, train=False):

        stream = nn.Flatten()(data) # Data should be passed through dataloader
        out = self.fc1(stream)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        logits = self.fc3(out)
        probs = self.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        if train:
            return probs    # We want to calculate loss
        else:
            return preds

    def save(self, llm, layer):
        with open(f'neural_{llm}_{layer}.pkl', 'wb') as file:
            pickle.dump(self, file)

''' The following code is from https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb by Burns et al. 2022 '''


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1,
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)


    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]

                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
