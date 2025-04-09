import torch
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

''' 
Utilities for intervening 
'''

def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index_to_ablate: int = 0
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

def patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    clean_cache = ActivationCache
) -> Float[torch.Tensor, "batch pos d_model"]:
    
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre

def steering_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    position: int,
    steer: None
) -> Float[torch.Tensor, "batch pos d_model"]:
    
    # Fill
    return