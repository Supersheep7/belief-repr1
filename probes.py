import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import copy
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Tuple, List
import pickle
from tqdm import tqdm
import einops
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

    def forward(self, x, iid=False):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction).unsqueeze(1)
        else:
            return t.nn.Sigmoid()(x @ self.direction).unsqueeze(1)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = t.relu(self.linear1(x))
        o = self.linear2(h)
        return t.sigmoid(o)

class Probe(object):

    def __init__(self,
                 input_dim,
                 nepochs: Int = 1000,
                 ntries: Int = 10,
                 lr: Float = 1e-3,
                 batch_size: Int =-1,
                 verbose: bool = False,
                 device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
                 probe_type: str = "linear",
                 weight_decay: Float = 0.01,
                 var_normalize: bool = False,
                 dropout: Float = 0.0,
                 with_direction: bool = False,
                 seed: int = 42,
                 max_iter: int = 1000,
                 C = 1e6
                 ):
        # data
        self.var_normalize = var_normalize
        self.input_dim =  input_dim
        self.dropout = dropout
        self.with_direction = with_direction
        self.seed = seed

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.C = C

        # probe
        self.probe_type = probe_type
        self.probe = None

    def initialize_probe(self):

        if self.with_direction or self.probe_type == 'mmp':
            # Compute direction
            if self.supervision_type == 'S':
                # Compute direction from data if supervised
                acts = t.tensor(self.x, dtype=t.float, requires_grad=True, device=self.device)
                labels = t.tensor(self.labels, dtype=t.float, requires_grad=True, device=self.device)
                if self.probe_type != 'linear':
                  acts = einops.rearrange(acts, "n_batches d_batch d_act -> (n_batches d_batch) d_act")
                  labels = einops.rearrange(labels, "n_batches d_batch -> (n_batches d_batch)")
                pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
            else:
                # Otherwise, direction and covariance will be computed from contrast pairs
                x0 = t.tensor(self.x0, dtype=t.float, requires_grad=True, device=self.device)
                x1 = t.tensor(self.x1, dtype=t.float, requires_grad=True, device=self.device)
                if self.probe_type != 'linear':
                  pos_acts = einops.rearrange(x0, "n_batches d_batch d_act -> (n_batches d_batch) d_act")
                  neg_acts = einops.rearrange(x1, "n_batches d_batch d_act -> (n_batches d_batch) d_act")
            pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
            self.direction = nn.Parameter(pos_mean - neg_mean, requires_grad=True)
            # Compute covariance if we use mmp
            if self.probe_type == 'mmp':
                centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
                covariance = centered_data.t() @ centered_data / centered_data.shape[0]

        if self.probe_type == "linear":
            self.probe = LogisticRegression(max_iter=self.max_iter,
                                            solver="lbfgs",
                                            C=self.C,
                                            random_state=self.seed,
                                            n_jobs=-1)
        elif self.probe_type == "mmp":
            if  self.supervision_type == "S":
                self.probe = MMP(acts=self.x, direction=self.direction, covariance=covariance)
            elif self.supervision_type == "U":
                self.probe = MMP(x0=self.x0, x1=self.x1, direction=self.direction, covariance=covariance)
        elif self.probe_type == "mlp":
            self.probe = MLPProbe(self.input_dim)

        self.probe.to(self.device) if self.probe_type != "linear" else None

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=-1, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=-1, keepdims=True)

        return normalized_x

    def repeated_train(self):

        if self.probe_type == "linear":
            # We just need to call sklearn's fit
            self.probe.fit(self.x, self.labels)
            return None

        else:
            best_loss = np.inf
            training = 0
            for train_num in range(self.ntries):
                self.initialize_probe()
                loss = self.train()
                if loss < best_loss:
                    self.best_probe = copy.deepcopy(self.probe)
                    best_loss = loss
                    training = train_num
            # print(f"Best loss: {best_loss} from training iter n {training}")
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
        For steering
        '''
        return self.direction

class SupervisedProbe(Probe):

    def __init__(self,
                 x: Float[t.Tensor, "n_batch batch_size d_activation"],
                 input_dim: Int,
                 labels: Float[t.Tensor, "n_batch batch_size"],
                 control: bool = False,
                 **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.x = self.normalize(x)
        self.labels = labels
        if control:
            np.random.shuffle(self.labels)
        if self.probe == "linear" and len(self.x.shape) > 2:
            self.x = einops.rearrange(self.x, "n_batches d_batch d_act -> (n_batches d_batch) d_act")
            self.labels = einops.rearrange(self.labels, "n_batches d_batch -> (n_batches d_batch)")
        self.supervision_type = "S"
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def get_tensor_data(self):
        """
        Forces x to be a tensor (rather than np array)
        """
        x = t.tensor(self.x, dtype=t.float, requires_grad=False, device=self.device)
        return x

    def get_loss(self,
                 p: Float[t.Tensor, "batch"],
                 labels: Float[t.Tensor, "batch"]
    ) -> Float[t.Tensor, "batch"]:

        return nn.functional.binary_cross_entropy(p, labels)

    def get_acc(self,
                X_test: Float[t.Tensor, "n_batch batch_size d_activation"],
                y_test: Float[t.Tensor, "n_batch batch_size"]
    ) -> Float:
        '''
        Returns accuracy for the best probe trained on a specific activation
        '''
        X_test = t.tensor(self.normalize(X_test), dtype=t.float, requires_grad=False, device=self.device)
        y_test = t.tensor(y_test, dtype=t.float, requires_grad=False, device=self.device)
        if self.probe_type == "linear":
            # We just call sklearn's predict
            probs = self.probe.predict(X_test.cpu().numpy())
            predictions = probs
            acc = (predictions == y_test.cpu().numpy()).mean()

        else:
            with t.no_grad():
                probs = self.best_probe(X_test)
            predictions = (probs.flatten().detach().cpu().numpy() >= 0.5).astype(int)
            acc = (predictions == y_test.cpu().numpy()).mean()

        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x = self.get_tensor_data() # Already batchified, shape (n_batches, batch_size, d_model)
        labels = self.labels

        # set up optimizer
        optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Start training
        for epoch in range(self.nepochs):
            for x_batch, labels_batch in zip(x, labels):  # Iterate directly over batches

                # probe
                '''
                Return the probabilities: I am NOT sure I should squeeze this tensor on the last dimension
                '''
                p: t.Tensor = self.probe(x_batch).squeeze(-1)

                # get the corresponding loss
                loss = self.get_loss(p.float(), labels_batch.float())

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

class UnsupervisedProbe(Probe):

    def __init__(self,
                 x0: Float[t.Tensor, "n_batch batch_size d_activation"],
                 x1: Float[t.Tensor, "n_batch batch_size d_activation"],
                 input_dim: Int,
                 **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.supervision_type = "U"
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = t.tensor(self.x0, dtype=t.float, requires_grad=False, device=self.device)
        x1 = t.tensor(self.x1, dtype=t.float, requires_grad=False, device=self.device)
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (t.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self,
                x0_test: Float[t.Tensor, "n_batch batch_size d_activation"],
                x1_test: Float[t.Tensor, "n_batch batch_size d_activation"],
                y_test: Float[t.Tensor, "n_batch batch_size"]
    ) -> Float:
        '''
        Returns accuracy for the best probe trained on a specific activation
        '''
        x0_test = t.tensor(self.normalize(x0_test), dtype=t.float, requires_grad=False, device=self.device)
        x1_test = t.tensor(self.normalize(x1_test), dtype=t.float, requires_grad=False, device=self.device)
        y_test = t.tensor(y_test, dtype=t.float, requires_grad=False, device=self.device).reshape(-1)
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

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()  # Already batchified, shape (n_batches, batch_size, d_model)
        permutation = t.randperm(len(x0))  # Shuffle the batches
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Start training
        for epoch in tqdm(range(self.nepochs), desc="Training"):
            for x0_batch, x1_batch in zip(x0, x1):  # Iterate directly over batches
                # probe
                p0, p1 = self.probe(x0_batch).squeeze(-1), self.probe(x1_batch).squeeze(-1)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

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
    for dataset in list_of_datasets:

        if probe_config.probe_type == 'linear':
            dataset = einops.rearrange(dataset, 'n b d -> (n b) d')

        if probe_config.supervision == "S":
            X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=probe_config.seed)
            probe = SupervisedProbe(probe_type=probe_config.probe_type,
                                    input_dim=dataset.shape[-1],
                                    x=X_train,
                                    labels=y_train,
                                    with_direction=True,
                                    nepochs=probe_config.nepochs,
                                    control=probe_config.control,
                                    ntries=probe_config.ntries,
                                    seed=probe_config.seed,
                                    max_iter=probe_config.max_iter,
                                    C=probe_config.C)
        else:
            x0, x1 = dataset[0], dataset[1]
            x0_train, x0_test, x1_train, x1_test, _, y_test = train_test_split(x0, x1, labels, test_size=0.2, random_state=probe_config.seed)
            probe = UnsupervisedProbe(probe_type=probe_config.probe_type,
                                    input_dim=x0.shape[-1],
                                    x0=x0_train,
                                    x1=x1_train,
                                    with_direction=True,
                                    nepochs=probe_config.nepochs,
                                    control=probe_config.control,
                                    ntries=probe_config.ntries,
                                    seed=probe_config.seed,
                                    max_iter=probe_config.max_iter,
                                    C=probe_config.C)
        probe.initialize_probe()
        probe.repeated_train()
        accuracies.append(probe.get_acc(X_test, y_test)) if probe_config.supervision == "S" else accuracies.append(probe.get_acc(x0_test=x0_test, x1_test=x1_test, y_test=y_test))
        directions.append(probe.get_direction())
        best_probes.append(probe.best_probe)

    return (accuracies, directions, best_probes)