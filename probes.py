import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
import torch as torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float, Int
import pickle
import transformer_lens as tlens

'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probe as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

''' Following class adapted from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py '''

class MMP(nn.Module):

    def __init__(self,
                covariance=None, 
                inv=None, 
                acts=None, 
                labels=None, 
                atol=1e-3
    ) -> None:

        super().__init__()

        if acts is not None and labels is not None:
            # Compute direction from data if provided
            pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
            pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
            self.direction = nn.Parameter(pos_mean - neg_mean, requires_grad=False)
            
            # Compute covariance
            centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            covariance = centered_data.t() @ centered_data / acts.shape[0]
        else:
            # Otherwise, you can initialize direction however you want or leave it as None
            self.direction = nn.Parameter(torch.zeros(acts.shape[1]), requires_grad=False)

        if inv is None:
            self.inv = nn.Parameter(torch.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
        else:
            self.inv = nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        if iid:
            return torch.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            return torch.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

class Saplma(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=128, hidden_dim3=64, output_dim=1, threshold=0.5):
        super().__init__()
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

    def forward(self, data):

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

        return probs

''' The following code is adapted from https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb by Burns et al. 2022 '''

class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class Probe(object):

    def __init__(self, 
                 input_dim,
                 nepochs: Int = 1000, 
                 ntries: Int = 10, 
                 lr: Float = 1e-3, 
                 batch_size: Int =-1,
                 verbose: bool = False, 
                 device: torch.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"), 
                 probe_type: str = "linear", 
                 weight_decay: Float = 0.01, 
                 var_normalize: bool = False
                 ):
        # data
        self.var_normalize = var_normalize
        self.input_dim =  input_dim

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.probe_type = probe_type 
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.probe_type == "linear":
            self.probe = nn.Sequential(nn.Linear(self.input_dim, 1), nn.Sigmoid())
        elif self.probe_type == "mmp" and self.supervision_type == "S":
            self.probe = MMP(self.input_dim, self.x, self.gold)                                # Initialize MMP with Labels
        elif self.probe_type == "mmp" and self.supervision_type == "U":
            self.probe = MMP(self.input_dim, self.x0, self.x1)                                # Initialize MMP with Contrast Pairs
        elif self.probe_type == "simple_mlp":
            self.probe = MLPProbe(self.input_dim)
        elif self.probe_type == "saplma":
            self.probe = Saplma(self.input_dim)
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

    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
    
    def save_best_probe(self, 
                        filename: str
    ) -> None:
        """
        Save the best trained probe to a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.best_probe, f)
        print(f"Best probe saved to {filename}")

class SupervisedProbe(Probe):

    def __init__(self, x, input_dim, gold, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.x = self.normalize(x)
        self.gold = gold
        self.d = self.x.shape[-1]
        self.supervision_type = "S"

    def get_tensor_data(self, x):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x = torch.tensor(x, dtype=torch.float, requires_grad=False, device=self.device)
        return x
    
    def get_loss(self, 
                 p: Float[torch.Tensor, "batch"], 
                 gold: Int[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        
        return torch.binary_cross_entropy_with_logits(p, gold)

    def get_acc(self, 
                X_valid: Float[torch.Tensor, "batch"], 
                y_valid: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch"]:
        
        X_valid = torch.tensor(self.normalize(X_valid), dtype=torch.float, requires_grad=False, device=self.device)

        with torch.no_grad():
            probs = self.best_probe(X_valid)
        predictions = (probs.detach().cpu().numpy() >= 0.5).astype(int)[:, 0]
        acc = (predictions == y_valid.cpu().numpy()).mean()

        return acc
    
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x = self.get_tensor_data(self.x)
        gold = self.get_tensor_data(self.gold)

        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = len(x) if self.batch_size == -1 else self.batch_size
        nbatches = len(x) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x_batch = x[j*batch_size:(j+1)*batch_size]
                gold_batch = gold[j*batch_size:(j+1)*batch_size]

                # probe
                p = self.probe(x_batch)

                # get the corresponding loss
                loss = self.get_loss(p, gold_batch)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

class UnsupervisedProbe(Probe):

    def __init__(self, x0, x1, input_dim, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]
        self.supervision_type = "U"

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