import torch as t
import tqdm.auto as tqdm
from tqdm import tqdm
import transformer_lens as tlens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from typing import List, Tuple, Dict
import numpy as np
import functools
import openai
from shots import get_shots
from intervention import generate
import re
from sklearn.linear_model import LogisticRegression

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

    return truth_value, confidence

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

def orthogonal_probing(probe_config, X_train, y_train, X_test, y_test, n=100):

    probabilities = np.zeros((X_test.shape[0], n))
    directions = []
    for i in range(n):  
        logreg = LogisticRegression(max_iter=probe_config.max_iter,
                                    solver="lbfgs",
                                    C=probe_config.C,
                                    random_state=42,
                                    n_jobs=-1,
                                    fit_intercept=False)
        for direction in directions:
            # Orthogonal constraint
            projected = np.dot(X_train, direction)
            X_train = X_train - np.outer(projected, direction) / np.dot(direction, direction)
        logreg.fit(X_train, y_train)
        directions.append(logreg.coef_.flatten())
        predicted_proba = logreg.predict_proba(X_test)[:, 1]
        probabilities[:, i] = predicted_proba
        if probe_config.log_accuracy_on_recursive:
            predictions = logreg.predict(X_test)
            acc = (predictions == y_test).mean()
            print(f"Run {i+1}/{n} accuracy: {acc:.4f}")

    return probabilities

def recursive_probing(probe_config, model, X_train, y_train, X_test, y_test, n=100):

        probabilities = np.zeros((X_test.shape[0], 1))
        tot_acc = 0
        for i in range(n):

            model.fit(X_train, y_train)
            predicted_proba = model.predict(X_test)[:, 1]
            
            if probe_config.log_accuracy_on_recursive:
                predictions = predicted_proba > 0.5
                acc = (predictions == y_test).mean()
                tot_acc += acc
            probabilities += predicted_proba.reshape(-1, 1)

        probabilities /= n
        if probe_config.log_accuracy_on_recursive:
            print(f"Average accuracy over {n} runs: {tot_acc / n:.4f}")

        return probabilities
    
class JudgeCoherence():

    def __init__(self, logic: str = None, metric: callable = None):
        
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

    def set_metric(self, metric: callable):
        '''
        A metric is a function that takes in a list/tensor of probabilities and returns a single float value.
        This can be performed in terms of distance metrics wrt an ideal agent
        Watch out: depending on the logic, the metric may take in different number of arguments.
        '''
        self.metric = metric

    def judge(self, proba: list) -> Float:

        if self.logic == 'neg':
            return self.metric(proba[0] + proba[1])
        elif self.logic in ['disj', 'conj', 'datasets']:
            return self.metric(proba[0], proba[1])
        elif self.logic == 'infe':
            return self.metric(proba[0], proba[1], proba[2])
