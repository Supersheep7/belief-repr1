import numpy as np
from sklearn.model_selection import train_test_split
import copy
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float, Int
from typing import Tuple, List
import pickle
from tqdm import tqdm
import einops
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probe as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

''' MMP class adapted from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py '''

''' Part of the following code is adapted from https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb by Burns et al. 2022 '''

class MMP(nn.Module):

    def __init__(self, direction, covariance, inv=None, atol=1e-3) -> None:
        super().__init__()
        self.direction = direction

        if inv is None:
            self.inv = nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=True)
        else:
            self.inv = nn.Parameter(inv, requires_grad=True)

    def forward(self, x, iid=True):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction).unsqueeze(1)
        else:
            return t.nn.Sigmoid()(x @ self.direction).unsqueeze(1)

    def pred(self, x, iid=True):
        return self(x, iid=iid).round()

class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, d//2)
        self.linear2 = nn.Linear(d//2, 1)

    def forward(self, x):
        h = t.relu(self.linear1(x))
        o = self.linear2(h)
        return t.sigmoid(o)

class Probe(object):

    def __init__(self, probe_config):

        # probe config
        self.var_normalize = probe_config.var_normalize
        self.dropout = probe_config.dropout
        self.with_direction = probe_config.with_direction
        self.seed = probe_config.seed
        self.nepochs = probe_config.nepochs
        self.ntries = probe_config.ntries
        self.lr = probe_config.lr
        self.verbose = probe_config.verbose
        self.device = probe_config.device
        self.batch_size = probe_config.batch_size
        self.weight_decay = probe_config.weight_decay
        self.max_iter = probe_config.max_iter
        self.C = probe_config.C
        self.probe_type = probe_config.probe_type
        self.control = probe_config.control

        # probe
        self.probe = None
        self.best_probe = None
        self.train_loader = None
        self.test_loader = None
        self.direction = None
        self.covariance = None

    def initialize_direction(self):
        if self.supervision_type == 'S':
            # Compute direction from data if supervised
            acts = t.tensor(self.x, dtype=t.float, requires_grad=True, device=self.device)
            labels = t.tensor(self.labels_train, dtype=t.float, requires_grad=True, device=self.device)
            pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        else:
            # Otherwise, direction and covariance will be computed from contrast pairs
            x0 = t.tensor(self.x0, dtype=t.float, requires_grad=True, device=self.device)
            x1 = t.tensor(self.x1, dtype=t.float, requires_grad=True, device=self.device)
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        self.direction = nn.Parameter(pos_mean - neg_mean, requires_grad=True)
        # Compute covariance if we use mmp
        if self.probe_type == 'mmp':
            centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            self.covariance = centered_data.t() @ centered_data / centered_data.shape[0]

    def initialize_probe(self):

        """
        Initializes the probe. If self.with_direction, also initializes the direction and covariance matrix.
        """

        if self.with_direction or self.probe_type == 'mmp':
            self.initialize_direction()

        if self.probe_type == "linear":
            self.x = np.array(self.x.detach().cpu())
            self.X_test = np.array(self.X_test.detach().cpu())
            self.labels_train = np.array(self.labels_train.detach().cpu())
            self.labels_test = np.array(self.labels_test.detach().cpu())
            self.probe = LogisticRegression(max_iter=self.max_iter,
                                            solver="lbfgs",
                                            C=self.C,
                                            random_state=self.seed,
                                            n_jobs=-1)
        else:
            if self.supervision_type == "S":
                x = self.x.clone().detach()
                X_test = self.X_test.clone().detach()
                train_labels = self.labels_train.clone().detach()
                test_labels = self.labels_test.clone().detach()
                dataset = TensorDataset(x, train_labels)
                test_dataset = TensorDataset(X_test, test_labels)
            else:
                self.x0 = t.tensor(self.x0, dtype=t.float, requires_grad=False, device=self.device)
                self.x1 = t.tensor(self.x1, dtype=t.float, requires_grad=False, device=self.device)
                dataset = TensorDataset(self.x0_train, self.x1_train)
                test_dataset = TensorDataset(self.x0_test, self.x1_test)

            self.train_loader = DataLoader(dataset, batch_size=self.batch_size if self.batch_size > 0 else len(dataset), shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            if self.probe_type == "linear_layer":
                self.probe = nn.Sequential(
                    nn.Linear(self.input_dim, 1, bias=False),
                    nn.Sigmoid()
                )
            if self.probe_type == "mmp":
                self.probe = MMP(direction=self.direction, covariance=self.covariance)
            if self.probe_type == "mlp":
                self.probe = MLPProbe(self.input_dim)

            self.probe.to(self.device)

    def normalize(self, x_train, x_test):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        train_mean = x_train.mean(dim=0, keepdim=True)
        train_std = x_train.std(dim=0, keepdim=True)

        normalized_x_train = x_train - train_mean
        if self.var_normalize:
            normalized_x_train /= (train_std + 1e-8)

        normalized_x_test = x_test - train_mean
        if self.var_normalize:
            normalized_x_test /= (train_std + 1e-8)

        return normalized_x_train, normalized_x_test

    def repeated_train(self):

        """
        Trains a bunch of probes and keeps the best one.
        If self.probe_type is linear, we just call sklearn's fit method.
        This is the high-level train method for our probes
        """

        if self.probe_type == "linear":
            self.initialize_probe()
            self.probe.fit(self.x, self.labels_train)
            self.best_probe = copy.deepcopy(self.probe)
            return None

        else:
            best_loss = float('inf')
            for train_num in range(self.ntries):
                self.initialize_probe()
                loss = self.train()
                if loss < best_loss:
                    self.best_probe = copy.deepcopy(self.probe)
                    best_loss = loss
            return best_loss

    def train(self):
        """
        Does a single training run of nepochs epochs
        """

        # set up optimizer
        optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float('inf')
        epoch_losses = []
        accuracies = []

        # Start training
        for epoch in range(self.nepochs):
            epoch_loss = 0.0  # Initialize epoch loss

            for x_batch, labels_batch in self.train_loader:
                # probe
                if self.supervision_type == "S":
                    p = self.probe(x_batch).squeeze(-1)
                    # get the corresponding loss
                    loss = self.get_loss(p.float(), labels_batch.float())
                else:
                    p0, p1 = self.probe(x_batch).squeeze(-1), self.probe(x_batch).squeeze(-1)
                    loss = self.get_loss(p0.float(), p1.float())

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            epoch_losses.append(epoch_loss)  # Track epoch loss for plotting

            self.probe.eval()  # Set the model to evaluation mode
            with t.no_grad():  # Disable gradient computation during validation
                correct = 0
                total = 0
                for x_batch, labels_batch in self.test_loader:  # Assuming you have a test_loader
                  predictions = self.probe(x_batch).squeeze(-1).round()
                  correct += (predictions == labels_batch).sum().item()
                  total += labels_batch.size(0)
                  acc = correct / total
            accuracies.append((correct / total))

        if self.verbose:
            # Plot the training and validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
            plt.plot(range(1, len(accuracies) + 1), accuracies, label='Online Accuracy', linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss Over Epochs Accuracy at the last step: {correct / total}')
            plt.legend()
            plt.grid(True)
            plt.show()

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

    def get_direction(self
        ) -> t.Tensor:
        '''
        For steering. TBD
        '''
        return self.direction

class SupervisedProbe(Probe):

    def __init__(self,
                 x_train: Float[t.Tensor, "n_data d_activation"],
                 x_test: Float[t.Tensor, "n_data d_activation"],
                 labels_train: Float[t.Tensor, "n_data"],
                 labels_test: Float[t.Tensor, "n_data"],
                 probe_config
                 ):
        super().__init__(probe_config=probe_config)
        self.input_dim = x_train.shape[-1]
        self.supervision_type = "S"
        self.x, self.X_test = self.normalize(x_train, x_test)
        self.labels_train = labels_train
        self.labels_test = labels_test

        """
        Shuffle the labels if control is True. This is done to create a control condition for the probe.
        """
        if self.control:
            np.random.shuffle(self.labels)

    def get_loss(self,
                 p: Float[t.Tensor, "batch"],
                 labels: Float[t.Tensor, "batch"]
    ) -> Float[t.Tensor, "batch"]:

        return nn.functional.binary_cross_entropy(p, labels)

    def get_acc(self) -> Float:
        '''
        Returns accuracy for the best probe trained on a specific activation
        '''
        if self.probe_type == "linear":
            # We just call sklearn's predict
            X_test = self.X_test
            labels_test = self.labels_test
            predictions = self.probe.predict(X_test)
            acc = (predictions == labels_test).mean()

        else:
            with t.no_grad():
                correct = 0
                total = 0
                for x_batch, labels_batch in self.test_loader:
                    predictions = self.best_probe(x_batch).squeeze(-1).round()
                    correct += (predictions == labels_batch).sum().item()
                    total += labels_batch.size(0)
                acc = (correct / total)

        return acc

class UnsupervisedProbe(Probe):

    def __init__(self,
                 x0_train: Float[t.Tensor, "n_batch batch_size d_activation"],
                 x0_test: Float[t.Tensor, "n_batch batch_size d_activation"],
                 x1_train: Float[t.Tensor, "n_batch batch_size d_activation"],
                 x1_test: Float[t.Tensor, "n_batch batch_size d_activation"],
                 labels: Float[t.Tensor, "n_batch batch_size"],
                 probe_config
                 ):
        super().__init__(probe_config=probe_config)
        self.input_dim = x0_train.shape[-1]
        self.x0_train, self.x0_test = self.normalize(x0_train, x0_test)
        self.x1_train, self.x1_test = self.normalize(x1_train, x1_test)
        self.labels = labels
        self.supervision_type = "U"

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (t.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self) -> Float:
        '''
        Returns accuracy for the best probe trained on a specific activation
        '''
        x0_test = self.x0_test.clone().detach().to(dtype=t.float, device=self.device)
        x1_test = self.x1_test.clone().detach().to(dtype=t.float, device=self.device)
        y_test = self.y_test.clone().detach().to(dtype=t.float, device=self.device)
        y_test = t.tensor(self.labels, dtype=t.float, requires_grad=False, device=self.device).reshape(-1)
        with t.no_grad():
            p0, p1 = self.best_probe(x0_test), self.best_probe(x1_test)
        '''Test below'''
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int).reshape(-1)
        '''Test above'''
        acc = (predictions == y_test.cpu().numpy()).mean()
        acc = max(acc, 1 - acc)

        # print("\nClassification Report:")
        # print(classification_report(y_test.cpu().numpy(), predictions, target_names=['Class 0', 'Class 1']))

        return acc

def probe_sweep(list_of_datasets: List,
                labels: t.Tensor,
                probe_config,
                ) -> Tuple:
    '''
    Runs a probe sweep on a list of activations

    Takes:
        list: a list of activations (if supervised); a list of tuples (activations0, activations1)
        labels: a tensor of labels of shape = list[0].shape (supervised) or list['pos'][0].shape
        probe_config: config object for the probe

    Returns: a list of accuracies for the list of activations; a list of vectors for steering; a list of best_probes for keeping them
    '''
    accuracies = []
    directions = []
    best_probes = []
    labels = einops.rearrange(labels, 'n b -> (n b)')

    for dataset in list_of_datasets:

        if probe_config.supervision == "S":
            dataset = einops.rearrange(dataset, 'n b d -> (n b) d')
            X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=probe_config.seed)
            probe = SupervisedProbe(x_train=X_train, labels_train=y_train,
                                    x_test=X_test, labels_test=y_test,
                                    probe_config=probe_config)
        else:
            x0, x1 = dataset[0], dataset[1]
            x0 = einops.rearrange(x0, 'n b d -> (n b) d')
            x1 = einops.rearrange(x1, 'n b d -> (n b) d')
            x0_train, x0_test, x1_train, x1_test, _, y_test = train_test_split(x0, x1, labels, test_size=0.2, random_state=probe_config.seed)
            probe = UnsupervisedProbe(x0_train=x0_train, x1_train=x1_train,
                                    x0_test=x0_test, x1_test=x1_test, labels=y_test,
                                    probe_config=probe_config)
        probe.repeated_train()
        accuracies.append(probe.get_acc())
        directions.append(probe.get_direction())
        best_probes.append(probe.best_probe)

    return (accuracies, directions, best_probes)