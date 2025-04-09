
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
from plotly.express import imshow, line
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float
import numpy as np 


class LogitAttribution:
    def __init__(self, model, device=None):
        self.model: HookedTransformer = model
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_lens_logit_diffs = None 
        self.logit_lens_labels = None  
        self.per_layer_logit_diffs = None 
        self.per_layer_labels = None 

    def residual_stack_to_logit_diff(self, 
                                    residual_stack: torch.Tensor, 
                                    cache: ActivationCache, 
                                    logit_diff_directions: float, 
                                    prompts: list
    ) -> float:
        
        scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
        return einsum(
            "... batch d_model, batch d_model -> ...",
            scaled_residual_stack,
            logit_diff_directions,
        ) / len(prompts)

    def plot_logit_diffs(self, data, x=None, hover_name=None, title=None):
        
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy() 
        
        line(
            data,
            x=x,
            hover_name=hover_name,
            title=title,
        ).show()

    def compute(self, 
                prompts: list, 
                answers: list
    ) -> None:
        
        # Tokenize answers, tokenize prompts, get cache on a fwd pass
        answer_tokens = torch.tensor(answers).to(self.device)
        tokens = self.model.to_tokens(prompts, prepend_bos=True)
        _, cache = self.model.run_with_cache(tokens)

        # Get logit difference directions
        answer_residual_directions = self.model.tokens_to_residual_directions(answer_tokens)
        logit_diff_directions = (answer_residual_directions[:, 0] - answer_residual_directions[:, 1])
        
        # Get cumulative difference
        accumulated_residual, self.logit_lens_labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
        self.logit_lens_logit_diffs = self.residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_directions, prompts)

        # Get per layer difference
        per_layer_residual, self.per_layer_labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
        self.per_layer_logit_diffs = self.residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions, prompts)

        return 
    
    def plot_all(self) -> None:

        self.plot_logit_diffs(
            self.logit_lens_logit_diffs,
            x=np.arange(self.model.cfg.n_layers * 2 + 1) / 2,
            hover_name=self.logit_lens_labels,
            title="Logit Difference From Accumulate Residual Stream",
        )
        
        self.plot_logit_diffs(
            self.per_layer_logit_diffs,
            hover_name=self.per_layer_labels,
            title="Logit Difference From Each Layer",
        )

        return
    
    def get_attribution(self):

        print("Logit lens diffs and labels:", 
              self.logit_lens_logit_diffs,
              self.logit_lens_labels)

        print("Per layer diffs and labels:", 
              self.per_layer_logit_diffs,
              self.per_layer_labels)

        return self.logit_lens_logit_diffs, self.logit_lens_labels, self.per_layer_logit_diffs, self.logit_lens_logit_diffs