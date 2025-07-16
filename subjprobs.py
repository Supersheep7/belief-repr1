import torch as t
import tqdm.auto as tqdm
from tqdm import tqdm
import transformer_lens as tlens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
import copy
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from typing import List, Tuple, Dict
import numpy as np
import functools
from intervention import generate
import re
from sklearn.linear_model import LogisticRegression
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


device = t.device("cuda" if t.cuda.is_available() else "cpu")

def circular_mean(thetas, performances):

    thetas = np.array(thetas)
    performances = np.array(performances)
    sum_weights = np.sum(performances)
    x = np.sum(np.sqrt(performances) * np.cos(thetas)) / sum_weights
    y = np.sum(np.sqrt(performances) * np.sin(thetas)) / sum_weights
    theta = np.arctan2(y, x)

    return theta

class FixedLinear(nn.Module):
    def __init__(self, weight, bias=None):
        super(FixedLinear, self).__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return t.matmul(x, self.weight.T) + self.bias
        else:
            return t.matmul(x, self.weight.T)

class LinearProbe():

    def __init__(self, X_train, y_train, X_val, y_val,
                 input_dim: int, output_dim: int = 1,
                 probe_config=None, random_state: int = 42):

        X_train = t.tensor(X_train, dtype=t.float32, device=device)
        y_train = t.tensor(y_train, dtype=t.float32, device=device)
        X_val = t.tensor(X_val, dtype=t.float32, device=device)
        y_val = t.tensor(y_val, dtype=t.float32, device=device)
        t.manual_seed(random_state)
        if t.cuda.is_available():
          t.cuda.manual_seed_all(random_state)
        linear_layer = nn.Linear(input_dim, 1, bias=False).to(device)
        nn.init.kaiming_uniform_(linear_layer.weight, a=5**0.5)
        if linear_layer.bias is not None:
            fan_in = input_dim
            bound = 1 / fan_in**0.5
            nn.init.uniform_(linear_layer.bias, -bound, bound)
        self.lr = probe_config.lr
        self.weight_decay = probe_config.weight_decay
        self.batch_size = probe_config.batch_size
        self.probe = nn.Sequential(
                    linear_layer,
                    nn.Sigmoid()
                )
        t.manual_seed(random_state)
        if probe_config.control:
          y_train = y_train[t.randperm(y_train.size()[0])]
        dataset = TensorDataset(t.tensor(X_train, dtype=t.float32), t.tensor(y_train, dtype=t.float32))
        val_dataset = TensorDataset(t.tensor(X_val, dtype=t.float32), t.tensor(y_val, dtype=t.float32))
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size if self.batch_size > 0 else len(dataset), shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        self.verbose = probe_config.verbose if probe_config else False
        self.nepochs = probe_config.nepochs
        self.best_probe = None
        self.patience = probe_config.patience

    def get_loss(self, p, labels):
        return t.nn.functional.binary_cross_entropy(p, labels)

    def get_direction(self):
        """
        Returns the direction of the probe in the input space.
        """
        with t.no_grad():
            direction = self.best_probe[0].weight.squeeze(0)
            return direction.cpu().numpy()

    def fit(self):
        optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float('inf')
        epoch_losses = []
        accuracies = []

        patience = self.patience  # You can expose this as a class param if desired
        epochs_no_improve = 0

        # Start training
        for epoch in tqdm(range(self.nepochs), desc='Epochs'):
            epoch_loss = 0.0

            for x_batch, labels_batch in self.train_loader:
                p = self.probe(x_batch).squeeze(-1)
                loss = self.get_loss(p.float(), labels_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_losses.append(epoch_loss)  # Track epoch loss for plotting
            self.probe.eval()

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.best_probe = copy.deepcopy(self.probe)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            current_acc = self.get_acc()
            accuracies.append(current_acc)

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if self.verbose:
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


    def predict(self, X_test, y_test) -> t.Tensor:

        X_test = t.tensor(X_test, dtype=t.float32, device=device)
        y_test = t.tensor(y_test, dtype=t.float32, device=device)
        with t.no_grad():
            dataset = TensorDataset(t.tensor(X_test, dtype=t.float32), t.tensor(y_test, dtype=t.float32))
            test_loader = DataLoader(dataset, batch_size=self.batch_size if self.batch_size > 0 else len(dataset), shuffle=True)
            correct = 0
            total = 0
            all_probs = []

            for x_batch, labels_batch in test_loader:
                predicted_proba = self.best_probe(x_batch).squeeze(-1)
                predictions = (predicted_proba > 0.5).float()

                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

                all_probs.append(predicted_proba)

            acc = correct / total
            all_probs = t.cat(all_probs, dim=0)  # Concatenate all batches into one tensor

        return acc, all_probs.cpu().numpy()

    def get_acc(self) -> t.Tensor:

        with t.no_grad():
            correct = 0
            total = 0
            for x_batch, labels_batch in self.val_loader:
                predicted_proba = self.best_probe(x_batch).squeeze(-1)
                predictions = (predicted_proba > 0.5).float()
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)
            acc = (correct / total)

            return acc

def self_reporting_confidence(model: HookedTransformer,
                              prompt: str,
                              context: str = 'You are a helpful assistant.\n You will receive a statement. You will be asked whether the answer is correct. Answer very carefully whether is True or False, and then provide a confidence score between 0 and 1. Don\'t be overconfident! I will provide three examples.\n',
                              shots: List[str] = ['Paris is the capital of France. This statement is: True. Confidence: 0.95',
                                                  'The largest bear in the world is currently in Italy. This statement is: False. Confidence: 0.70',
                                                  'Milan is the capital of Italy. This statement is: False. Confidence: 0.85'],
                              device: str = "cuda") -> Float:
    """
    Computes the self-reporting probabilities for a batch of tokens using a pre-trained model.
    """

    with t.amp.autocast('cuda'):
      context = context
      shots = '\n'.join(shots)
      full_prompt = context + shots + '\n' + prompt
      answer = generate(model=model, prompt=full_prompt, temperature=0, max_length=10)
      truth_value = answer.split()[0]
      match = re.search(r'\d+\.\d+', answer)
      if match:
          confidence = float(match.group())
      else:
          confidence = 'NaN'

    return truth_value, max(confidence, 1-confidence)

def logit_confidence(model: HookedTransformer,
                    prompt: str,
                    context: str = 'You are a helpful assistant. You will receive a statement. You will be asked whether the answer is correct. ONLY answer "True" or "False". I will provide three examples. \n',
                    shots: List[str] = ['Paris is the capital of France. This statement is: True.\n',
                                        'The richest person in the earth is George Clooney. This statement is: False.\n',
                                        'Milan is the capital of Italy. This statement is: False.\n'],
                    device: str = "cuda") -> Float:
    with t.amp.autocast('cuda'):
      true_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in ['true', 'True', 'TRUE', 'Ġtrue', 'ĠTrue', 'ĠTRUE']]
      false_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in ['false', 'False', 'FALSE', 'Ġfalse', 'ĠFalse', 'ĠFALSE']]
      full_prompt = context + ''.join(shots) + prompt
      tokens = model.to_tokens(full_prompt)
      logits = model(tokens)
      log_probs = t.nn.functional.log_softmax(logits, dim=-1)
      selected_token_ids = true_token_ids + false_token_ids
      restricted_log_probs = log_probs[0, -1, selected_token_ids]
      restricted_probs = t.exp(restricted_log_probs)
      p_true = restricted_probs[:len(true_token_ids)].sum().item()
      p_false = restricted_probs[len(true_token_ids):].sum().item()
      normalized_p_true = p_true / (p_true + p_false)
      normalized_p_false = p_false / (p_true + p_false)
      truth_value = 'True' if normalized_p_true > normalized_p_false else 'False'

    return truth_value, max(normalized_p_true, normalized_p_false)

# On latent representations

def probe_confidence(activations: t.Tensor, trained_probe, probe_type='logistic') -> Float:
    """
    Computes the confidence score using a trained probe or a set thereof on an activation batch.
    """
    activations = activations.cpu().numpy()

    if isinstance(trained_probe, list):
        confidence = 0
        for probe in trained_probe:
            confidence += probe.predict_proba(activations)[:, 1] if probe_type == 'logistic' else probe(activations)
        confidence /= len(trained_probe)
    else:
        confidence = trained_probe.predict_proba(activations)[:, 1] if probe_type == 'logistic' else trained_probe(activations)

    return confidence

def orthogonal_probing(probe_config, X_train, y_train, X_test, y_test, fix_direction=None, n=100):

    probabilities = np.zeros((X_test.shape[0], n))
    directions = []
    if fix_direction is not None:
        directions.append(fix_direction)
    accuracies = []
    for i in range(n):
        for direction in directions:
            # Orthogonal constraint
            if type(direction) == t.Tensor:
                direction = direction.cpu().numpy()
            elif type(X_train) == t.Tensor:
                direction = direction.cpu().numpy()
            projected = np.dot(X_train, direction)
            X_train = X_train - np.outer(projected, direction) / np.dot(direction, direction)
        probe = LinearProbe(X_train, y_train, X_test, y_test,
                                input_dim=X_train.shape[1],
                                output_dim=1,
                                probe_config=probe_config,
                                random_state=i)
        probe.fit()
        acc, run_probs = probe.predict(X_test, y_test)
        directions.append(probe.get_direction())
        if probe_config.log_accuracy_on_recursive:
                print(f"Run {i+1}/{n} accuracy: {acc:.4f}")
                accuracies.append(acc)

    return accuracies

def recursive_probing(probe_config, X_train, y_train, X_test, y_test, n=100):

        probabilities = np.zeros((X_test.shape[0]))
        tot_acc = 0
        for i in range(n):
            model = LinearProbe(X_train, y_train, X_test, y_test,
                                input_dim=X_train.shape[1],
                                output_dim=1,
                                probe_config=probe_config,
                                random_state=i)
            model.fit()
            acc, predicted_proba = model.predict(X_test, y_test)

            if probe_config.log_accuracy_on_recursive:
                tot_acc += acc
                print(f"Run {i+1}/{n} accuracy: {acc:.4f}")
            probabilities += predicted_proba

        probabilities /= n

        if probe_config.log_accuracy_on_recursive:
            print(f"Average accuracy over {n} runs: {tot_acc / n:.4f}")

        return probabilities

def form_master_probe(probe_config, X_train, y_train, X_test, y_test, n=100, circular=False, pca=False):

        directions = []
        performances = []
        for i in range(n):
            model = LinearProbe(X_train, y_train, X_test, y_test,
                                input_dim=X_train.shape[1],
                                output_dim=1,
                                probe_config=probe_config,
                                random_state=i)
            model.fit()
            acc, run_probs = model.predict(X_test, y_test)
            direction = model.get_direction()
            directions.append(direction)

            if probe_config.log_accuracy_on_recursive:
                print(f"Run {i+1}/{n} accuracy: {acc:.4f}")
                performances.append(acc)
        if pca:
          directions /= np.linalg.norm(directions, axis=1, keepdims=True)
          my_pca = PCA(n_components=1)
          pc0 = my_pca.fit(directions).components_[0]
          avg_direction = t.tensor(pc0, dtype=t.float32, device=device)
        else:
          avg_direction = circular_mean(thetas=directions, performances=performances) if circular else np.average(directions, weights=performances, axis=0)
          avg_direction = t.tensor(avg_direction, dtype=t.float32, device=device)

        return nn.Sequential(
                    FixedLinear(avg_direction),
                    nn.Sigmoid()
                ), directions

class JudgeCoherence():

    def __init__(self, logic: str = None):

        self.logic = None
        self.metric = None

    def set_logic(self, logic: str):

        self.logic = logic

    def cosine_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Computes the cosine distance between two probability distributions.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.nn.functional.cosine_similarity(proba1, proba2, dim=-1)

    def kl_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Computes the KL divergence between two probability distributions.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.nn.functional.kl_div(t.log_softmax(proba1, dim=-1), t.softmax(proba2, dim=-1), reduction='batchmean')

    def rmse_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Computes the RMSE between two probability distributions.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.sqrt(t.mean((proba1 - proba2) ** 2))

    def aggregate_euclidean_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Computes the Euclidean distance between two probability distributions.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.norm(proba1 - proba2, p=2)
    
    def less_than_perc(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Computes the percentage of elements in proba1 that are less than the corresponding elements in proba2.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return (proba1 < proba2).float().mean().item()

    def set_metric(self, metric: callable):
        '''
        A metric is a function that takes in a list/tensor of probabilities and returns a single float value.
        This can be performed in terms of distance metrics wrt an ideal agent
        Watch out: depending on the logic, the metric may take in different number of arguments.
        '''
        self.metric = metric

    def judge(self, proba: list) -> Float:

        if self.logic == 'neg':
            return self.metric(proba[0] + proba[1], t.ones_like(proba[0]))
        elif self.logic in ['disj', 'conj', 'datasets']:
            return self.metric(proba[0], proba[1])
        elif self.logic == 'infe':
            return self.metric(proba[0], proba[1], proba[2])