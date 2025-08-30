import torch as t
import tqdm.auto as tqdm
from tqdm import tqdm
from transformer_lens.hook_points import (
    HookPoint,
)
import copy
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from typing import List, Tuple, Dict
import numpy as np
from intervention import generate
import re
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import openai
import pandas as pd 
import gc 
from sklearn.model_selection import train_test_split
device = t.device("cuda" if t.cuda.is_available() else "cpu")

''' Coherence Study '''

def stratified_fixed_sample(df, stratify_col, n, random_state=42):
    
    proportions = df[stratify_col].value_counts(normalize=True)

    per_class_n = (proportions * n).round().astype(int)

    diff = n - per_class_n.sum()
    if diff != 0:
        fractional = (proportions * n) - (proportions * n).round()
        adjust_classes = fractional.abs().sort_values(ascending=False).index
        for i in range(abs(diff)):
            per_class_n[adjust_classes[i % len(adjust_classes)]] += int(diff / abs(diff))

    sampled = []
    for label, count in per_class_n.items():
        sampled.append(df[df[stratify_col] == label].sample(n=count, random_state=random_state))

    return pd.concat(sampled)


def get_data_split(task: str, curated_dataset, probe_config, other_dataset=None, cutoff=1500):

    ''' 
    Provides a train/test split for the different datasets and tasks.
    Crucially, the test split is already provided in the different versions;
    Outputs: 
            train, labels, dataframe
    '''

    if task == 'cross':
        train_set, test_set_0 = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        test_df = (test_set_0.dropna(), other_dataset.dropna())
    elif task == 'cross_mulan':
        train_set, _ = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        mutable = other_dataset[other_dataset['type'] == 'mutable']
        immutable = other_dataset[other_dataset['type'] == 'immutable']
        test_df = (mutable.dropna(), immutable.dropna())
    elif task in ['negation', 'disjunction', 'conjunction']:
        # In this case curated dataset is the logical dataset + the remainder
        remainder = curated_dataset[~curated_dataset['filename'].isin(['common_claim_true_false.csv', 'companies_true_false.csv', 'counterfact_true_false.csv'])]
        curated_dataset = pd.concat([remainder, other_dataset])
        train_set, test_set = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        test_df = (test_set.dropna(), test_set.dropna())
        
    elif task == 'inference':
        remainder = curated_dataset[~curated_dataset['filename'].isin(['common_claim_true_false.csv', 'companies_true_false.csv', 'counterfact_true_false.csv',
                                                                      'cities.csv', 'cities_cities_conj.csv', 'cities_cities_disj.csv', 'neg_cities.csv'
                                                                      ])]
        curated_dataset = pd.concat([remainder, other_dataset])
        train_set, test_set = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        test_df = (test_set.dropna(), test_set.dropna())
        

    # Trim for batch size
    train_set = train_set.iloc[:-(len(train_set) % probe_config.batch_size_extractor), :]
    X_clean_train = list(train_set['statement'])
    y_clean_train = list(train_set['label'])
    
    return (X_clean_train, y_clean_train, test_df)


class FixedLinear(nn.Module):

    ''' Linear probe with a fixed theta '''

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

    ''' Linear probe trained through BCE & AdamW. For random init '''

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
        with t.no_grad():
            direction = self.best_probe[0].weight.squeeze(0)
            return direction.cpu().numpy()

    def fit(self):
        optimizer = t.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float('inf')
        epoch_losses = []
        accuracies = []

        patience = self.patience  # Early stopping
        epochs_no_improve = 0

        for epoch in tqdm(range(self.nepochs), desc='Epochs'):
            epoch_loss = 0.0

            for x_batch, labels_batch in self.train_loader:
                p = self.probe(x_batch).squeeze(-1)
                loss = self.get_loss(p.float(), labels_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_losses.append(epoch_loss)  
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

def self_evaluate_row(prompt, model):
    truth, conf = self_reporting_confidence(model, prompt)
    return pd.Series({'truth': truth, 'confidence': conf})

def self_reporting_confidence(model: HookedTransformer,
                              prompt: str,
                              context: str,
                              shots: List[str],
                              device: str = "cuda") -> Float:
    """
    Computes the self-reporting probabilities for a batch of tokens using a pre-trained model.
    Cleans up CUDA memory after use to prevent OOM when called repeatedly.
    """

    try:
        context = context + '\n'
        shots = '\n'.join(shots)
        full_prompt = context + shots + '\n' + f'{prompt}\nAnswer:'

        with t.no_grad(), t.autocast(device):
            answer = generate(model=model, prompt=full_prompt, temperature=0, max_length=10)
        match = re.search(r'\d+\.\d+', answer)
        if match:
            truth_value = answer.split()[0]
            confidence = float(match.group())
        else:
            print(f"Warning: No confidence score found in the answer: {answer}")
            truth_value = "Unknown"
            confidence = float('nan')

        confidence = max(confidence, 1 - confidence)

        if re.fullmatch(r'(Ġ)?(False|false|FALSE)(\.)?', str(truth_value)):
            confidence = 1 - confidence

        return truth_value, confidence

    finally:
        # Force cleanup of unused memory
        gc.collect()
        t.cuda.empty_cache()



def logit_confidence(model: HookedTransformer,
                    prompt: str,
                    context: str,
                    shots: List[str],
                    device: str = "cuda") -> Float:
    with t.amp.autocast('cuda'):
      #Ġ
      #▁
      true_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in ['true', 'True', 'TRUE', 'Ġtrue', 'ĠTrue', 'ĠTRUE']]
      false_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in ['false', 'False', 'FALSE', 'Ġfalse', 'ĠFalse', 'ĠFALSE']]
      full_prompt = context + '\n'.join(shots) + '\n' + f'{prompt}\nAnswer:'
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
      if p_true + p_false < 0.05:
          # If the model is not confident, we set the truth value to 0.5
          print("Warning: model underconfident")
          normalized_p_true = 0.5
    return truth_value, normalized_p_true

def logit_evaluate_row(prompt):
    with t.no_grad():
        truth, conf = logit_confidence(model, prompt, context='', shots=shots)
    return pd.Series({'truth': truth, 'confidence': conf})


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
          avg_direction = np.average(directions, weights=performances, axis=0)
          avg_direction = t.tensor(avg_direction, dtype=t.float32, device=device)

        return nn.Sequential(
                    FixedLinear(avg_direction),
                    nn.Sigmoid()
                ), directions

class JudgeCoherence():

    def __init__(self, logic: str = None):

        self.logic = logic
        self.metric = None

    def set_logic(self, logic: str):

        self.logic = logic

    
    def to_confidence(probs):
        probs = t.tensor(probs, dtype=t.float32, device=device)
        pointfives = t.full_like(probs, 0.5, device=device)
        probs = t.abs(probs - pointfives) * 2
        return probs 

    def cosine_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:

        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.nn.functional.cosine_similarity(proba1, proba2, dim=-1)

    def kl_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:

        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.nn.functional.kl_div(t.log_softmax(proba1, dim=-1), t.softmax(proba2, dim=-1), reduction='batchmean')

    def rmse_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        ''' 
        Standard for neg logic.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return 1/(1+t.mean((proba1 - proba2) ** 2))

    def mae_metric_clamp(self, proba1: t.Tensor, proba2: t.Tensor, proba3: t.Tensor, easy=False) -> Float:
        ''' 
        Standard for impl logic.
        '''
        proba1 = t.tensor(proba1).clip(lower=0.01)                                  # P(psi and phi)
        proba2 = t.tensor(proba1).clip(lower=0.01)                                  # P(psi)
        y =  t.ones_like(proba3) if easy else t.tensor(proba3).clip(lower=0.01)     # P(phi|psi)            
        y_hat = proba1/proba2
        mask = y_hat < 2
        filtered_y_hat = y_hat[mask]
        filtered_y = y[mask]

        return 1/(1+t.abs(t.mean((filtered_y_hat - filtered_y))))

    def aggregate_euclidean_metric(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:

        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return t.norm(proba1 - proba2, p=2)
    
    def less_than_perc(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Standard for conj/disj logic.
        '''
        proba1 = t.tensor(proba1)
        proba2 = t.tensor(proba2)
        return (proba1 < proba2).float().mean().item()

    def avg_conf_diff(self, proba1: t.Tensor, proba2: t.Tensor) -> Float:
        '''
        Standard for cross-dataset logic.
        '''
        proba1 = self.to_confidence(t.tensor(proba1))
        proba2 = self.to_confidence(t.tensor(proba2))
        return (proba1 - proba2).abs().mean().item()

    def set_metric(self, metric: callable):

        self.metric = metric

    def judge(self, proba: list) -> Float:

        if self.logic == 'neg':
            return self.metric(proba[0] + proba[1], t.ones_like(proba[0]))
        elif self.logic in ['disj', 'conj', 'datasets']:
            return self.metric(proba[0], proba[1])
        # Disj case: proba[0] < proba[1]
        # Conj case: proba[0] > proba[1]
        # Dataset case: mean(proba[0]) < mean(proba[1]) where proba[0] is the target dataset and proba[1] is the source dataset
        elif self.logic == 'ent':
            return self.metric(proba[0], proba[1], proba[2])     
        elif self.logic == 'ent*':
            return self.metric(proba[0], proba[1], proba[2], easy=True)     
        # TRUE case: proba[0] == P(phi); proba[1] == P(phi -> psi); proba[2] == P(psi)
        # FALSE case: proba[0] == P(psi); proba[1] == P(phi -> psi); proba[2] == P(psi)

''' Self-Consistency Study '''

def bin_proba(probs):
    '''
    bins tensor in 10 different bins
    '''
    probs = t.tensor(probs, dtype=t.float32, device=device)
    bins = t.linspace(0, 1, steps=11, device=device)
    binned_probs = t.bucketize(probs, bins) - 1  # -1 to make it zero-indexed
    return binned_probs

def check_calibration(probs, labels):

    probs = t.tensor(probs, dtype=t.float32, device=device)

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Consistency Error (ECE)"""
    probs = np.array(probs)
    labels = np.array(labels)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_edges[i], bin_edges[i+1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_probs = probs[in_bin]
            bin_labels = labels[in_bin]
            avg_conf = np.mean(bin_probs)
            avg_acc = np.mean(bin_labels)
            ece += (bin_size / total) * np.abs(avg_conf - avg_acc)
    return ece