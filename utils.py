
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
from typing import Callable
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

    def plot_logit_diffs(self, 
                         data, 
                         x=None, 
                         hover_name=None, 
                         title=None
    ) -> None:
        
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

class PatchingMetrics():

    '''
    Takes: model type, logits, clean and corrupted input, torch.device
    Returns: None
    Usecase: Initialize PatchingMetrics object with your parameter, then call the relevant metric and pass it to patching
    E.g. 
    mymetric = PatchingMetrics(gpt2, logits, prompt, corrupted_prompt, cuda)
    ... call patching from transformer lens ...
    act_patch_resid_pre = patching.get_act_patch_resid_pre(
    **args, patching_metric=mymetric
    )
    '''

    def __init__(self, 
                 model: HookedTransformer,
                 clean: Float[torch.Tensor, "batch seq"], 
                 corrupted: Float[torch.Tensor, "batch seq"], 
                 answers: Float[torch.Tensor, "batch 2"],
                 device: torch.device = None
    ) -> None:
        
        self.model: HookedTransformer = model
        self.device = device
        self.clean = clean
        self.corrupted = corrupted
        self.clean_logits = None 
        self.corrupted_logits = None
        self.answers = answers

    def logits_to_ave_logit_diff(self,
                                logits: Float[torch.Tensor, "batch seq d_vocab"],
                                answer_tokens: Float[torch.Tensor, "batch 2"],
                                per_prompt: bool = False,
    ) -> Float[torch.Tensor, "*batch"]:
        """
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        """
        final_logits: Float[torch.Tensor, "batch d_vocab"] = logits[:, -1, :]
        answer_logits: Float[torch.Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    def run(self) -> None:
        self.clean_logits, _ = self.model.run_with_cache(self.clean)
        self.corrupted_logits, _ = self.model.run_with_cache(self.corrupted)
        return
    
    def reset(self) -> None:
        self.clean_logits = None
        self.corrupted_logits = None
        return

    def logit_diff(self,
                   logits: Float[torch.Tensor, "batch seq d_vocab"]
    ) -> Float[torch.Tensor, ""]:

        if self.clean_logits is None or self.corrupted_logits is None:
            raise ValueError("Logits not found. Please call the `run` method first.")


        patched_logit_diff = self.logits_to_ave_logit_diff(logits, self.answers)
        clean_logit_diff = self.logits_to_ave_logit_diff(self.clean_logits, self.answers)
        corrupted_logit_diff = self.logits_to_ave_logit_diff(self.corrupted_logits, self.answers)

        return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
  
    def logit_diff_variation(self,
                     logits: Float[torch.Tensor, "batch seq d_vocab"]
    ) -> Float[torch.Tensor, ""]:
        
        if self.clean_logits is None or self.corrupted_logits is None:
            raise ValueError("Logits not found. Please call the `run` method first.")

        patched_logit_diff = self.logits_to_ave_logit_diff(logits, self.answers)
        clean_logit_diff = self.logits_to_ave_logit_diff(self.clean_logits, self.answers)
        corrupted_logit_diff = self.logits_to_ave_logit_diff(self.corrupted_logits, self.answers)

        return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

        
        



# Float[Tensor, ""] This is the returntype of functions here
