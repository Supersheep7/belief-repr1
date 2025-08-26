import torch as t
import tqdm.auto as tqdm
from tqdm import tqdm
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from typing import List, Tuple, Dict
import numpy as np
import functools
import openai

'''
= = = = = = = = = = = = = = = = Intervention = = = = = = = = = = = = = = = =
'''

def generate(model, prompt, max_length=50, temperature=0.0, top_k=None):

    with t.no_grad():
        tokens = model.to_tokens(prompt)
        generated_tokens = tokens.clone()
        for _ in range(max_length):
            logits = model(generated_tokens)
            next_token_logits = logits[0, -1, :]
            if temperature > 0:
                next_token_logits /= temperature

            if top_k is not None:
                top_k_values, _ = t.topk(next_token_logits, top_k)
                threshold = top_k_values[-1]
                next_token_logits[next_token_logits < threshold] = -float('inf')

            probabilities = t.nn.functional.softmax(next_token_logits, dim=-1)
            if temperature != 0:
              next_token = t.multinomial(probabilities, num_samples=1)
              append = next_token.unsqueeze(0)
            else:
              next_token = t.argmax(probabilities)
              append = next_token.unsqueeze(0).unsqueeze(0)  
            # print("Next token:", model.tokenizer.decode(next_token))
            generated_tokens = t.cat([generated_tokens, append], dim=1)
            if next_token.item() == model.tokenizer.eos_token_id:
                break
        generated_text = model.tokenizer.decode(generated_tokens[0, len(tokens[0]):])
    return generated_text


def mask_top_k(activation_accuracies: np.array,
                activation_directions: np.array,
                K: int = 1
                ) -> Tuple[List[Tuple], List[np.array]]:

    """
    Takes a tensor of head accuracies and a tensor of head directions,
    returns a list of the top K heads in coordinate form and their corresponding directions.

    E.g. with K = 1
    returns top_k_indices = [(10, 12)]
    returns top_k_directions = [activation_directions[10][12]] = [direction vector of head 12 in layer 10]
    """

    assert activation_accuracies.shape == activation_directions.shape[:2], "Shape mismatch between activation_accuracies and activation_directions"
    assert K <= activation_accuracies.numel(), "K is larger than the number of available heads"

    # Get the indices of the top K heads
    head_accuracies_flattened = activation_accuracies.flatten()
    flat_indices = np.argsort(head_accuracies_flattened)[-K:]
    row_indices, col_indices = np.unravel_index(flat_indices, activation_accuracies.shape)
    top_k_indices = list(zip(row_indices, col_indices)) # (list of tuples of (layer, head) indices)

    # Get the corresponding directions
    top_k_directions = []
    top_k_directions = [activation_directions[layer, head] for layer, head in top_k_indices]

    return top_k_indices, top_k_directions

def set_intervention_hooks(model: HookedTransformer,
                           top_k_indices: List[Tuple],
                           top_k_directions: List[t.Tensor],
                           alpha: float = 1,
                           verbose: bool = False
                           ) -> List:

    """
    Sets the intervention hooks for the top K heads.
    """

    def steering_hook(z: Float[t.Tensor, "d_batch seq_len n_head d_head"],
                      hook: HookPoint,
                      head_idx: int,
                      head_direction: t.Tensor,
                      alpha: float = 1):

        """
        Steers the model by returning a modified activations tensor,
        with some multiple of the steering vector added to the top K heads.
        """
        assert head_direction.shape == z.shape[-1:], f"Shape mismatch: {head_direction.shape} vs {z.shape[-1:]}"
        # Steer only the d_head corresponding to the given head_index

        head_direction = head_direction / head_direction.norm()

        z[:, -1, head_idx, :] += alpha * head_direction

        return z

    model.reset_hooks()
    half = True if next(model.parameters(), None).dtype == t.float16 else False

    if half:
        # Set half precision for the steering
        model.add_hook("hook_embed", lambda tensor, hook: tensor.half())
    if verbose:
        print(f"Setting hooks for top {len(top_k_indices)} heads:")
    for (layer, head), direction in zip(top_k_indices, top_k_directions):
        if verbose:
            print(f"Layer {layer}, Head {head}, Direction Norm: {direction.norm().item()}")
        if half:
            direction = direction.clone().detach().half()
        steering = functools.partial(steering_hook, head_idx=head, head_direction=direction, alpha=alpha)
        print("adding hook for layer", layer, "head", head)
        model.add_hook(f"blocks.{layer}.attn.hook_z", steering)

    return model

def full_intervention(model: HookedTransformer,
                      activation_accuracies: np.array,
                      activation_directions: np.array,
                      K: int = 1,
                      alpha: int = 1,
                      verbose: bool = False) -> HookedTransformer:

    """
    Full intervention function that sets the hooks for the top K heads and returns the model with the hooks set.
    """
    # Force everything into tensors
    activation_accuracies = t.as_tensor(activation_accuracies)
    activation_directions = t.as_tensor(activation_directions)

    # Get the top K heads and their directions
    top_k_indices, top_k_directions = mask_top_k(activation_accuracies, activation_directions, K)

    # Set the intervention hooks for the top K heads
    model = set_intervention_hooks(model, top_k_indices, top_k_directions, alpha, verbose)

    return model

def intervention_on_residual(
                            model: HookedTransformer,
                            activation_accuracies: np.array,
                            activation_directions: np.array,
                            k: int = -1,
                            alpha: int = 1,
                            top_features: np.array = None,
                            verbose: bool = False
                            ) -> HookedTransformer:

    top_k_indices = np.argsort(activation_accuracies)[-k:][::-1]
    top_k_directions = [t.as_tensor(activation_directions[i]) for i in top_k_indices]

    def steering_residual_hook(
                      resid: Float[t.Tensor, "n_batch n_seq d_model"],
                      hook: HookPoint,
                      direction: t.Tensor,
                      alpha: float = 1,
                      top_features: int = None
                      ):

        assert direction.shape == resid.shape[-1:], f"Shape mismatch: {direction.shape} vs {resid.shape[-1:]}"

        direction = direction / direction.norm()
        resid[:, -1, :] += alpha * direction
        return resid

    model.reset_hooks()
    half = True if next(model.parameters(), None).dtype == t.float16 else False

    if half:
        # Set half precision for the steering
        model.add_hook("hook_embed", lambda tensor, hook: tensor.half())
    for layer, direction in zip(top_k_indices, top_k_directions):        
        if half:
            direction = direction.clone().detach().half()
        steering = functools.partial(steering_residual_hook, direction=direction, alpha=alpha, top_features=top_features)
        model.add_hook(f"blocks.{layer}.hook_resid_post ", steering)

    return model


'''
= = = = = = = = = = = = = = = = Evaluation = = = = = = = = = = = = = = = =
'''

''' *** Parameter Sweep ***'''

def parameter_sweep(model_baseline: HookedTransformer,
                    prompts: List[str],
                    activation_accuracies,
                    activation_directions,
                    ks : List = [1, 2, 3, 4, 5],
                    alphas: List = [1, 2, 3, 4, 5],
                    metric: str = 'cosine',
                    verbose: bool = False,
                    shots: List = None,
                    secret = None,
                    dataset_name: str = 'truefalse',
                    labels: List[int] = None,
                    attn: bool = True
                    ) -> np.array:

    with t.no_grad():

        metrics = np.zeros((len(ks), len(alphas)))

        if metric == 'judge':
            client = openai.OpenAI(api_key=secret)
            informative = np.zeros((len(ks), len(alphas)))
        if metric == 'boolprobs':
            prob_diff = np.zeros((len(ks), len(alphas)))

        model_baseline.reset_hooks()

        if next(model_baseline.parameters(), None).dtype == t.float16:
            model_baseline.add_hook("hook_embed", lambda tensor, hook: tensor.half())
        if metric in ['kl', 'ce', 'cosine']:
           baseline_probs = get_mass_probs(model_baseline, prompts)

        for num_k, k in tqdm(enumerate(ks)):
                tqdm.write(f"Steering top {k} activations")
                for num_alpha, alpha in tqdm(enumerate(alphas)):
                    tqdm.write(f"With strength {alpha}")
                    model_baseline.reset_hooks()
                    if attn:
                        model_to_evaluate = full_intervention(model_baseline, activation_accuracies, activation_directions, K=k, alpha=alpha, verbose=verbose)
                    else:
                        model_to_evaluate = intervention_on_residual(model=model_baseline, activation_accuracies=activation_accuracies, activation_directions=activation_directions, k=k, alpha=alpha, verbose=verbose)
                    if metric in ['kl', 'ce', 'cosine']:
                        eval_probs = get_mass_probs(model_to_evaluate, prompts)
                        metrics[num_k, num_alpha] = probs_mass_eval(baseline_probs, eval_probs, metric=metric)
                    elif metric == 'judge':
                        pass # We dropped this metric
                    elif metric == 'boolprobs':
                        if labels is None:
                            raise ValueError("Labels must be provided for truth assignment evaluation.")
                        metrics[num_k, num_alpha], prob_diff[num_k, num_alpha] = mass_truth_assignment_eval(model_to_evaluate, prompts, shots=shots, labels=labels)
        if metric == 'judge':
            return metrics, informative, metrics * informative
        elif metric == 'boolprobs':
            return metrics, prob_diff
        else:
            return metrics

''' *** P(True)-P(False) ***'''

def truth_assignment_single_eval(
              model: HookedTransformer,
              prompt: str,
              label: int,
              true_tokens,
              false_tokens
              ):

    true_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in true_tokens]
    false_token_ids = [model.tokenizer.convert_tokens_to_ids(token) for token in false_tokens]
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    log_probs = t.nn.functional.log_softmax(logits, dim=-1)
    log_p_true = t.logsumexp(log_probs[0, -1, true_token_ids], dim=0).item()
    log_p_false = t.logsumexp(log_probs[0, -1, false_token_ids], dim=0).item()
    most_probable_token_id = t.argmax(log_probs[0, -1]).item()
    most_probable_token = model.tokenizer.convert_ids_to_tokens([most_probable_token_id])[0]
    # print(f"Prompt: {prompt}")
    # print(f"P(True): {np.exp(log_p_true)}, P(False): {np.exp(log_p_false)}")
    successful = int(int(log_p_true >= log_p_false) != label)
    if np.exp(log_p_true) + np.exp(log_p_false) < 0.1:
        # print("Broken!")
        print(f"Answer: {most_probable_token}")
        successful = 0
    # print(f"Correct: {correct}")

    prob_diff = np.exp(log_p_true) - np.exp(log_p_false) if label == 1 else np.exp(log_p_false) - np.exp(log_p_true)
    # print(prob_diff)

    return successful, prob_diff

def mass_truth_assignment_eval(
              model: HookedTransformer,
              statements: List[str],
              labels: List[int],
              true_tokens: List[str] = ['true',
                                        'Ġtrue',
                                        'True',
                                        'ĠTrue'
                                        ],
              false_tokens: List[str] = [
                                        'false',
                                        'Ġfalse',
                                        'False',
                                        'ĠFalse'
                                        ],
              shots: List[str] = None,
              ) -> float:

    """
    Evaluates the model on a set of prompts and labels, returning the average P(True) - P(False) score.
    """

    assert len(statements) == len(labels), "Number of prompts and labels must match"

    total_metric = 0.0
    total_prob_diff = 0.0
    considered_statements = len(statements)
    for statement, label in zip(statements, labels):
        # Rembember to check if the newlines lead to correct answers
        prompt = f"The sky is blue. This sentence is: True \n The earth is flat. This sentence is: False \n {statement}"
        correct, prob_diff = truth_assignment_single_eval(model, prompt, label, true_tokens, false_tokens)
        total_metric += correct
        total_prob_diff += prob_diff

    return total_metric / considered_statements, total_prob_diff / considered_statements

''' *** Distances *** '''

def get_mass_probs(model: HookedTransformer,
                    prompts: List[str]
                    ) -> t.Tensor:

    '''
    Takes: a list of prompts
    Returns: a tensor of (n_prompts, n_vocab) logprobs
    '''
    tot_probs = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        logits = model(tokens).squeeze()[-1]
        probs = t.nn.functional.softmax(logits, dim=-1)
        tot_probs.append(probs)

    return t.stack(tot_probs, dim=0)

def probs_mass_eval(baseline_probs: t.Tensor,
                    eval_probs: t.Tensor,
                    metric: str = 'kl') -> t.Tensor:

    '''
    Takes: two tensors of (n_prompts, n_vocab) logprobs + a metric
    Returns: a scalar value for the metric
    '''

    if metric == 'ce':
        return -t.sum(baseline_probs * t.log(eval_probs), dim=-1).mean()
    elif metric == 'kl':
        return t.sum(baseline_probs * (t.log(baseline_probs) - t.log(eval_probs)), dim=-1).mean()
    elif metric == 'cosine':
        return t.nn.functional.cosine_similarity(baseline_probs, eval_probs, dim=-1).mean()
    
def get_strength(k, alpha, model, attn=True):
    if attn:
        return np.abs(k*alpha/(model.cfg.n_heads*model.cfg.n_layers))
    else:
        return np.abs(k*alpha/model.cfg.n_layers)