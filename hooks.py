import torch as t
import torch.nn as nn
import einops
import tqdm.auto as tqdm
import plotly.express as px
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
from functools import partial
from sae_lens import SAE

''' 
Utilities for intervening 
'''

def head_ablation_hook(
    value: Float[t.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index_to_ablate: int = 0
) -> Float[t.Tensor, "batch pos head_index d_head"]:
    """
    Ablates a specific attention head by setting its values to zero. 
    The ablation is applied to the head specified by `head_index_to_ablate`.
    """
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

def patching_hook(
    resid_pre: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    clean_cache = ActivationCache
) -> Float[t.Tensor, "batch pos d_model"]:
    """
    Patches the residual stream at a specific position using clean activations. 
    Replaces activations at `position` with values from `clean_cache`.
    """
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

def steering_SAE_hook(
    activations: Float[t.Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> t.Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    
    return activations + steering_coefficient * sae.W_dec[latent_idx]

def steering_hook(
    activations: Float[t.Tensor, "batch pos d_in"],
    hook: HookPoint,
    direction: Float[t.Tensor, "d_in"],
    steering_coefficient: float,
    last_token: True
) -> t.Tensor:
    
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """

    if last_token:
        # Modify the last position across the batch
        activations[:, -1, :] += steering_coefficient * direction
    else:
        # Apply steering across all positions
        activations += steering_coefficient * direction
    
    return activations