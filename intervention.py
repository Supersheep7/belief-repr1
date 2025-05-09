import torch as t
import torch.nn as nn
import tqdm.auto as tqdm
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

def mask_top_k_heads(head_accuracies: np.array, 
                     head_directions: np.array, 
                     K: int = 1
                     ) -> Tuple[List[Tuple], List[np.array]]:

    """
    Takes a tensor of head accuracies and a tensor of head directions, 
    returns a list of the top K heads in coordinate form and their corresponding directions.
    
    E.g. with K = 1
    returns top_k_head_indices = [(10, 12)]
    returns top_k_directions = [head_directions[10][12]] = [direction vector of head 12 in layer 10]
    """

    assert head_accuracies.shape == head_directions.shape[:2], "Shape mismatch between head_accuracies and head_directions"
    assert K <= head_accuracies.numel(), "K is larger than the number of available heads"

    # Get the indices of the top K heads
    head_accuracies_flattened = head_accuracies.flatten()
    flat_indices = np.argsort(head_accuracies_flattened)[-K:]
    row_indices, col_indices = np.unravel_index(flat_indices, head_accuracies.shape)
    top_k_head_indices = list(zip(row_indices, col_indices)) # (list of tuples of (layer, head) indices)

    # Get the corresponding directions 
    top_k_directions = []
    top_k_directions = [head_directions[layer, head] for layer, head in top_k_head_indices]

    return top_k_head_indices, top_k_directions

def set_intervention_hooks(model: HookedTransformer,
                           top_k_head_indices: List[Tuple], 
                           top_k_directions: List[t.Tensor], 
                           alpha: float = 1,
                           verbose: bool = False
                           ) -> List:

    """
    Sets the intervention hooks for the top K heads.
    """
    
    def steering_hook(z: Float[t.Tensor, "n_batch d_batch n_head d_head"],
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
        
        z[:, :, head_idx, :] += alpha * head_direction

        return z

    model.reset_hooks()
    half = True if next(model.parameters(), None).dtype == t.float16 else False

    if half:
        # Set half precision for the steering
        model.add_hook(("hook_embed", lambda tensor, hook: tensor.half()))
    if verbose:
        print(f"Setting hooks for top {len(top_k_head_indices)} heads:")
    for (layer, head), direction in zip(top_k_head_indices, top_k_directions):
        if verbose:
            print(f"Layer {layer}, Head {head}, Direction Norm: {direction.norm().item()}")
        if half:
            direction = direction.clone().detach().half()
        steering = functools.partial(steering_hook, head_idx=head, head_direction=direction, alpha=alpha)
        model.add_hook(f"blocks.{layer}.attn.hook_z", steering)

    return model

def full_intervention(model: HookedTransformer, 
                      head_accuracies: np.array, 
                      head_directions: np.array,
                      K: int = 1,
                      alpha: int = 1,
                      verbose: bool = False) -> HookedTransformer:    

    """
    Full intervention function that sets the hooks for the top K heads and returns the model with the hooks set.
    """
    # Force everything into tensors
    head_accuracies = t.as_tensor(head_accuracies)
    head_directions = t.as_tensor(head_directions)

    # Get the top K heads and their directions
    top_k_head_indices, top_k_directions = mask_top_k_heads(head_accuracies, head_directions, K)

    # Set the intervention hooks for the top K heads
    model = set_intervention_hooks(model, top_k_head_indices, top_k_directions, alpha, verbose)

    return model

def parameter_sweep(model: HookedTransformer,
                    tokens: t.Tensor,
                    head_accuracies,
                    head_directions,
                    ks : List = [1, 2, 3, 4, 5],
                    alphas: List = [0, 0.5, 1, 2, 5], 
                    metric: str = 'cosine',
                    verbose: bool = False
                    ) -> np.array:
    
    metrics = np.zeros((len(ks), len(alphas)))

    model.reset_hooks()
    clean_logits = model(tokens).squeeze()[-1]
    clean_logprobs = t.nn.functional.softmax(clean_logits, dim=-1)

    for num_k, k in enumerate(ks):
        for num_alpha, alpha in enumerate(alphas):
            model_with_steering_hooks = full_intervention(model, head_accuracies, head_directions, K=k, alpha=alpha, verbose=verbose)
            with t.no_grad():
                if metric != 'judge':
                    intervened_logits = model_with_steering_hooks(tokens).squeeze()[-1]
                    intervened_logprobs = t.nn.functional.softmax(intervened_logits, dim=-1)
                    if metric == 'cosine':
                        cosine_similarity = t.nn.functional.cosine_similarity(clean_logits, intervened_logits, dim=-1).cpu().numpy()
                        metrics[num_k, num_alpha] = cosine_similarity
                    elif metric == 'kl':
                        kl_divergence = t.nn.functional.kl_div(clean_logprobs, intervened_logprobs, reduction='batchmean').cpu().numpy()
                        metrics[num_k, num_alpha] = kl_divergence
                    elif metric == 'ce':
                        ce = t.nn.functional.cross_entropy(clean_logprobs, intervened_logprobs, reduction='mean').cpu().numpy()
                        metrics[num_k, num_alpha] = ce
    
    return metrics