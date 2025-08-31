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
device = t.device("cuda" if t.cuda.is_available() else "cpu")
'''
Here we have the three probes that we will deploy to test for internal representation of belief
Logreg is a simple logistic regressor
MMP is a mass-mean probe as described in Marks & Tegmark 2023
Neural is a tentative copy of SAPLMA as described in Azaria & Mitchell 2023
'''

''' MMP class adapted from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py '''

''' Part of the following code is adapted from https://github.com/collin-burns/discovering_latent_knowledge/blob/main/CCS.ipynb by Burns et al. 2022 '''

class MMP(nn.Module):

    def __init__(self, direction, covariance, inv=None, atol=1e-3):
        super().__init__()
        self.direction = direction.to(device)

        if inv is None:
            inv_32 = t.linalg.pinv(covariance.float(), hermitian=True, atol=atol)
            self.inv = nn.Parameter(inv_32.to(dtype=t.float16), requires_grad=False).to(device)
        else:
            self.inv = nn.Parameter(inv, requires_grad=False).to(device)

    def forward(self, x, iid=True):
        x = nn.Parameter(t.tensor(x).to(device), requires_grad=False)
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction).unsqueeze(1)
        else:
            return t.nn.Sigmoid()(x @ self.direction).unsqueeze(1)
        
class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, d//2)
        self.linear2 = nn.Linear(d//2, d//4)
        self.linear3 = nn.Linear(d//4, d//8)
        self.linear4 = nn.Linear(d//8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = t.relu(x)
        x = self.linear2(x)
        x = t.relu(x)
        x = self.linear3(x)
        x = t.relu(x)
        x = self.linear4(x)
        return t.sigmoid(x)

class Probe(object):

    def __init__(self, probe_cfg):

        # probe config
        self.var_normalize = probe_cfg.var_normalize
        self.dropout = probe_cfg.dropout
        self.direction_type = probe_cfg.direction_type
        self.seed = probe_cfg.seed
        self.nepochs = probe_cfg.nepochs
        self.ntries = probe_cfg.ntries
        self.lr = probe_cfg.lr
        self.verbose = probe_cfg.verbose
        self.device = probe_cfg.device
        self.batch_size = probe_cfg.batch_size
        self.weight_decay = probe_cfg.weight_decay
        self.max_iter = probe_cfg.max_iter
        self.C = probe_cfg.C
        self.probe_type = probe_cfg.probe_type
        self.control = probe_cfg.control

        # probe
        self.probe = None
        self.best_probe = None
        self.train_loader = None
        self.test_loader = None
        self.direction = None
        self.covariance = None
        self.std = None

    def initialize_direction(self, direction_type, full_dataset=True):
        
        if direction_type == 'mmp':
            """
            We take the mass mean of positive and negative activations and subtract them to get the direction.
            """
            if not isinstance(self.x, t.Tensor):
                x_train = t.tensor(self.x, dtype=t.float, device=self.device) if not isinstance(self.x, t.Tensor) else self.x
                x_test = t.tensor(self.X_test, dtype=t.float, device=self.device) if not isinstance(self.x, t.Tensor) else self.X_test
                y_train = t.tensor(self.labels_train, dtype=t.float, device=self.device) if not isinstance(self.x, t.Tensor) else self.labels_train
                y_test = t.tensor(self.labels_test, dtype=t.float, device=self.device) if not isinstance(self.x, t.Tensor) else self.labels_test
            else:
                x_train, x_test, y_train, y_test = self.x, self.X_test, self.labels_train, self.labels_test
            if full_dataset:
                data_for_mm = t.cat([x_train, x_test], dim=0).to(device)
                labels_for_mm = t.cat([y_train, y_test], dim=0)
            else:
                data_for_mm = t.tensor(x_train, device=device)
                labels_for_mm = t.tensor(y_train, device=device)
            pos_acts, neg_acts = data_for_mm[labels_for_mm == 1], data_for_mm[labels_for_mm == 0]
            pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
            self.direction = nn.Parameter(pos_mean - neg_mean, requires_grad=True)
            centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            self.covariance = centered_data.t() @ centered_data / centered_data.shape[0]
        elif direction_type == None: 
            self.direction = None
        else:
            with t.no_grad():
              theta = self.best_probe.coef_[0] if direction_type == 'logistic' else self.best_probe[0].weight[0].cpu().numpy()
              self.direction = nn.Parameter(t.tensor(theta, dtype=t.float, requires_grad=True, device=self.device).squeeze(0))

    def initialize_probe(self):

        if self.probe_type == "mmp":
            # We need the direction and covariance in advance for the MMP probe 
            self.initialize_direction('mmp', full_dataset=False)
            self.x = np.array(self.x)
            self.X_test = np.array(self.X_test)
            self.labels_train = np.array(self.labels_train.detach().cpu())
            self.labels_test = np.array(self.labels_test.detach().cpu())
            self.probe = MMP(direction=self.direction, covariance=self.covariance)

        elif self.probe_type == "linear":
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
                x_train = self.x.clone().float().to(device)
                y_train = self.labels_train.clone().float().to(device)
                dataset = TensorDataset(x_train, y_train)
                X_test = self.X_test.clone().float().to(device)
                y_test = self.labels_test.clone().float().to(device)
                test_dataset = TensorDataset(X_test, y_test)
            else:
                x0_train = self.x0_train.clone().detach()
                x1_train = self.x1_train.clone().detach()
                dataset = TensorDataset(x0_train, x1_train)
                x0_test = self.x0_test.clone().detach()
                y_test = self.labels.clone().detach()
                test_dataset = TensorDataset(x0_test, y_test)

            self.train_loader = DataLoader(dataset, batch_size=self.batch_size if self.batch_size > 0 else len(dataset), shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            if self.probe_type == "linear_layer":
                self.probe = nn.Sequential(
                    nn.Linear(self.input_dim, 1, bias=True),
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
        
        elif self.probe_type == "mmp":
            self.initialize_probe()

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
                    p0, p1 = self.probe(x_batch).squeeze(-1), self.probe(labels_batch).squeeze(-1)
                    loss = self.get_loss(p0.float(), p1.float())

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            epoch_losses.append(epoch_loss)  # Track epoch loss for plotting
            self.best_probe = copy.deepcopy(self.probe)
            self.probe.eval()  # Set the model to evaluation mode
            current_acc = self.get_acc()
            accuracies.append(current_acc)

        if self.verbose:
            # Plot the training and validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
            plt.plot(range(1, len(accuracies) + 1), accuracies, label='Online Accuracy', linestyle='--')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss Over Epochs Accuracy at the last step: {current_acc}')
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

    def get_direction(self,
                      std: bool = False
        ) -> t.Tensor:
        '''
        Gets the chosen direction for steering. 
        '''
        self.initialize_direction(self.direction_type)
        direction = self.direction.clone().detach()

        if std: 
            std = self.get_std()
            return std * direction

        return direction
    
    def get_std(self) -> t.Tensor:
        '''
        For steering.
        '''
        if self.supervision_type == "S":
            # If data is npt, convert to torch tensor
            if not isinstance(self.x, t.Tensor):
                self.x = t.tensor(self.x, dtype=t.float, device=self.device)
                self.X_test = t.tensor(self.X_test, dtype=t.float, device=self.device)
            full_dataset = t.cat([self.x, self.X_test], dim=0)
        else:
            if not isinstance(self.x0_train, t.Tensor):
                self.x0_train = t.tensor(self.x0_train, dtype=t.float, device=self.device)
                self.x1_train = t.tensor(self.x1_train, dtype=t.float, device=self.device)
                self.x0_test = t.tensor(self.x0_test, dtype=t.float, device=self.device)
                self.x1_test = t.tensor(self.x1_test, dtype=t.float, device=self.device)
            full_dataset = t.cat([self.x0_train, self.x1_train, self.x0_test, self.x1_test], dim=0)
        project_on_direction = full_dataset @ self.direction
        self.std = project_on_direction.std(dim=0, keepdim=True)

        return self.std

class SupervisedProbe(Probe):

    def __init__(self,
                 x_train: Float[t.Tensor, "n_data d_activation"],
                 x_test: Float[t.Tensor, "n_data d_activation"],
                 labels_train: Float[t.Tensor, "n_data"],
                 labels_test: Float[t.Tensor, "n_data"],
                 probe_cfg
                 ):
        super().__init__(probe_cfg=probe_cfg)
        self.input_dim = x_train.shape[-1]
        self.supervision_type = "S"
        self.x, self.X_test = self.normalize(x_train, x_test) if self.var_normalize else (x_train, x_test)
        self.labels_train = labels_train
        self.labels_test = labels_test

        """
        Shuffle the labels if control is True. This is done to create a control condition for the probe.
        """
        if self.control:
            np.random.shuffle(self.labels_train)
            if self.probe_type == "mmp":
                np.random.shuffle(self.labels_test)

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
        elif self.probe_type == 'mmp':
            # We just call a forward
            X_test = self.X_test
            labels_test = self.labels_test
            predictions = self.probe(X_test, iid=True).squeeze(-1).detach().cpu().numpy().round() # Only one probe
            acc = (predictions == labels_test).mean()
            print(acc)

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
                 probe_cfg
                 ):
        super().__init__(probe_cfg=probe_cfg)
        self.input_dim = x0_train.shape[-1]
        self.x0_train, self.x0_test = self.normalize(x0_train, x0_test)
        self.x1_train, self.x1_test = self.normalize(x1_train, x1_test)
        self.labels = labels
        self.supervision_type = "U"
        assert probe_cfg.probe_type not in ["linear", "mmp"], ("You should call linear_layer or mlp for an Unsupervised Probe")

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
        x0_test = self.x0_test.clone().detach()
        x1_test = self.x1_test.clone().detach()
        y_test = self.labels.clone().detach()
        with t.no_grad():
            p0, p1 = self.best_probe(x0_test), self.best_probe(x1_test)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test.cpu().numpy()).mean()
        acc = max(acc, 1 - acc)

        return acc

def probe_sweep(list_of_datasets: List,
                labels: t.Tensor,
                probe_cfg
                ) -> Tuple:
    '''
    Runs a probe sweep on a list of activations

    Takes:
        list: a list of activations (if supervised); a list of tuples (activations0, activations1)
        labels: a tensor of labels of shape = list[0].shape (supervised) or list['pos'][0].shape
        probe_cfg: config object for the probe

    Returns: a list of accuracies for the list of activations; a list of vectors for steering; a list of best_probes for keeping them
    '''
    accuracies = []
    directions = []
    best_probes = []
    labels = einops.rearrange(labels, 'n b -> (n b)')

    for dataset in tqdm(list_of_datasets, desc="heads", disable=not probe_cfg.verbose):

        if probe_cfg.supervision == "S":
            dataset = einops.rearrange(dataset, 'n b d -> (n b) d')
            X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=probe_cfg.test_size, random_state=probe_cfg.seed)
            probe = SupervisedProbe(x_train=X_train, labels_train=y_train,
                                    x_test=X_test, labels_test=y_test,
                                    probe_cfg=probe_cfg)
        else:
        
            x0, x1 = dataset[0], dataset[1]
            x0 = einops.rearrange(x0, 'n b d -> (n b) d').to(device)
            x1 = einops.rearrange(x1, 'n b d -> (n b) d').to(device)
            x0_train, x0_test, x1_train, x1_test, _, y_test = train_test_split(x0, x1, labels, test_size=probe_cfg.test_size, random_state=probe_cfg.seed)
            probe = UnsupervisedProbe(x0_train=x0_train, x1_train=x1_train,
                                    x0_test=x0_test, x1_test=x1_test, labels=y_test,
                                    probe_cfg=probe_cfg)
        probe.repeated_train()
        accuracies.append(probe.get_acc())
        if probe_cfg.direction_type != None:
            directions.append(probe.get_direction(std=probe_cfg.with_std))
        best_probes.append(probe.best_probe)

    return (accuracies, directions, best_probes)
