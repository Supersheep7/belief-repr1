
import torch as t
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
from plotly.express import imshow, line
import transformer_lens as tlens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer, ActivationCache
from jaxtyping import Float, Int
from typing import List, Tuple, Dict
import numpy as np 
from torch.amp import autocast
from tqdm import tqdm
from datasets import Dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import pandas as pd
import gc

def decompose_mha(mha_batch: Float[t.Tensor, "n_batch batch_size n_head d_head"]
                  ) -> List[Float[t.Tensor, "n_batch batch_size d_head"]]:

    """
    Decomposes a multi-head attention tensor into a list of tensors for each head.

    Returns:
        List[Tensor]: A list where each element is a tensor of shape (n_batch, batch_size, d_head),
                      corresponding to each attention head.
    """
    decomposed = einops.rearrange(mha_batch, 'n_batch batch_size n_head d_head -> n_head n_batch batch_size d_head')
    
    return [decomposed[i] for i in range(decomposed.shape[0])]

def rearrange_by_act_type(act_dict):
    """
    Rearranges the activations in the dictionary by their type (e.g., 'resid_pre', 'resid_post').

    Args:
        act_dict (Dict): A dictionary where keys are activation types and values are tensors.

    Returns:
        Dict: A new dictionary with keys as activation types and values as lists of tensors.
    """
    rearranged_dict = {}
    
    for key, value in act_dict.items():
        act_type = key.split('.')[-1]  # Get the activation type from    the key
        if act_type not in rearranged_dict:
            rearranged_dict[act_type] = []
        rearranged_dict[act_type].append(value)
    
    return rearranged_dict

class LogitAttribution:
    def __init__(self, model, device=None):
        self.model: HookedTransformer = model
        self.device = device if device is not None else ("cuda" if t.cuda.is_available() else "cpu")
        self.logit_lens_logit_diffs = None 
        self.logit_lens_labels = None  
        self.per_layer_logit_diffs = None 
        self.per_layer_labels = None 

    def residual_stack_to_logit_diff(self, 
                                    residual_stack: t.Tensor, 
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
        
        if isinstance(data, t.Tensor):
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
        answer_tokens = t.tensor(answers).to(self.device)
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
    Takes: model type, logits, clean and corrupted input, t.device
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
                 clean: Float[t.Tensor, "batch seq"], 
                 corrupted: Float[t.Tensor, "batch seq"], 
                 answers: Float[t.Tensor, "batch 2"],
                 device: t.device = None
    ) -> None:
        
        self.model: HookedTransformer = model
        self.device = device
        self.clean = clean
        self.corrupted = corrupted
        self.clean_logits = None 
        self.corrupted_logits = None
        self.answers = answers

    def get_logits(self):
        if self.clean_logits is None or self.corrupted_logits is None:
            raise ValueError("Logits not found. Please call the `run` method first.")
        print("Returned metrics cache: clean logits, corrupted logits")
        return self.clean_logits, self.corrupted_logits

    def logits_to_ave_logit_diff(self,
                                logits: Float[t.Tensor, "batch seq d_vocab"],
                                answer_tokens: Float[t.Tensor, "batch 2"],
                                per_prompt: bool = False,
    ) -> Float[t.Tensor, "*batch"]:
        """
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        """
        final_logits: Float[t.Tensor, "batch d_vocab"] = logits[:, -1, :]
        answer_logits: Float[t.Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    def run(self) -> None:
        self.clean_logits, _ = self.model.run_with_cache(self.clean)
        self.corrupted_logits, _ = self.model.run_with_cache(self.corrupted)
        return
    
    def reset(self) -> None:
        print("Resetting logits")
        self.clean_logits = None
        self.corrupted_logits = None
        return

    def logit_diff(self,
                   logits: Float[t.Tensor, "batch seq d_vocab"]
    ) -> Float[t.Tensor, ""]:

        if self.clean_logits is None or self.corrupted_logits is None:
            raise ValueError("Logits not found. Please call the `run` method first.")

        patched_logit_diff = self.logits_to_ave_logit_diff(logits, self.answers)
        clean_logit_diff = self.logits_to_ave_logit_diff(self.clean_logits, self.answers)
        corrupted_logit_diff = self.logits_to_ave_logit_diff(self.corrupted_logits, self.answers)

        return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
  
    def logit_diff_variation(self,
                     logits: Float[t.Tensor, "batch seq d_vocab"]
    ) -> Float[t.Tensor, ""]:
        
        if self.clean_logits is None or self.corrupted_logits is None:
            raise ValueError("Logits not found. Please call the `run` method first.")

        patched_logit_diff = self.logits_to_ave_logit_diff(logits, self.answers)
        clean_logit_diff = self.logits_to_ave_logit_diff(self.clean_logits, self.answers)
        corrupted_logit_diff = self.logits_to_ave_logit_diff(self.corrupted_logits, self.answers)

        return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

class ActivationExtractor():

    '''
    A class to extract activations from a transformer model for later probing tasks.

    Args:
        model: The pre-trained transformer model (e.g., GPT-2).
        tokenizer: Tokenizer corresponding to the model.
        dataset: List of sentences to process.
        batch_size: Number of sentences per batch.
    
    Core method:
        process(): returns a list of tensors of shape batch_size n_acts d_model & a list of tensors of shape batch_size labels. 
        The list can be later used by probes for training/testing on that layer and get accuracy on the task
    
    Use case:

    extractor = ActivationExtractor(model, X, y)
        this will batchify everything during init
    extractor.set_hooks([layers], [names])
        example: extractor.set_hooks([0, 1, 2], ['resid_pre', 'resid_post'])
    batched_activations, batched_labels = extractor_truthfulQA_layer0.process()
        returns a dictionary for each relevant hook and the batched labels to test against      
    '''
    
    def __init__(self, 
                 model: tlens.HookedTransformer, 
                 data: List, 
                 labels: List, 
                 device: t.device,
                 half: bool,
                 batch_size=32,
                 pos=-1):
        
        self.model = model.to(device)
        self.X = self.batchify(data, batch_size)
        self.y = t.tensor(self.batchify(labels, batch_size), dtype=t.float32).to(device)
        self.hooks = []
        self.activations = {}
        self.half = half
        self.device = device
        self.pos = pos  # Position to extract activations from, -1 means last token


    def set_hooks(self, layers, names, attn=False):

        if self.half:
            self.hooks.append(("hook_embed", lambda tensor, hook: tensor.half()))
        
        def get_act_hook(tensor, hook):
            
            last_token = tensor[:, self.pos, :, :].unsqueeze(0) if attn else tensor[..., self.pos, :].unsqueeze(0)  
            last_token = last_token.to(dtype=t.float16, device=t.device('cpu'))

            if hook.name in self.activations:
                self.activations[hook.name] = t.cat([self.activations[hook.name], last_token], dim=0)
            else:
                self.activations[hook.name] = last_token

            return tensor

        for layer in layers:
            for name in names:
                self.hooks.append((f"blocks.{layer}.{name}", get_act_hook))
          
                
    def extract_activations_batch(self, 
                                  sentences: t.Tensor, 
                                  model: tlens.HookedTransformer, 
                                  ) -> None:
        """
        Extract activations and sets them in self dictionary
        """

        '''sentences.shape == (batch_size 1)'''

        tokens = model.to_tokens(sentences)

        '''tokens.shape == (batch_size seq_len)'''

        # Forward pass to get activations
        with autocast('cuda'):

          with t.no_grad():

              model.reset_hooks()
              
              # Forward pass running with hooks
              model.run_with_hooks(
                  tokens,
                  return_type=None,
                  fwd_hooks=self.hooks
              )

        return

    def batchify(self, data, batch_size):
        """
        Split data into batches. We need to add padding or something like that
        """
        result = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        assert len(result[-1]) % batch_size == 0, "Data length must be divisible by batch_size"
        return result

    def process(self
    ) -> Tuple[List[Float[t.Tensor, "batch_size n_acts d_model"]], Int[t.Tensor, "batch_size"]]:
        # Process
        for batch in tqdm(self.X, "Processing"):
            self.extract_activations_batch(batch, self.model)
        return self.activations, self.y

class Seq2SeqDataset(Dataset):
    def __init__(self, df, input_col, target_col, tokenizer, max_len=128):
        self.input_texts = df[input_col].tolist()
        self.target_texts = df[target_col].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_tokens = self.tokenizer(self.input_texts[idx], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        target_tokens = self.tokenizer(self.target_texts[idx], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return input_tokens['input_ids'].squeeze(0), target_tokens['input_ids'].squeeze(0)

def finetune_model(model, df, input_col, target_col, epochs=3, batch_size=16, lr=1e-5, max_len=128):
    dataset = Seq2SeqDataset(df, input_col, target_col, model.tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = t.optim.AdamW(model.parameters(), lr=lr)
    criterion = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.cuda(), target_ids.cuda()

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)

            # Shift target tokens for teacher forcing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    print("Fine-tuning complete.")
    return model

def get_top_heads(accuracies, n=5):
    
    '''
    For intervention analysis: given a matrix of accuracies (layers x heads), return the top n heads and their corresponding accuracies
    '''

    flat_indices = np.argpartition(accuracies.flatten(), -n)[-n:]
    coordinates = np.array(np.unravel_index(flat_indices, accuracies.shape)).T
    top_heads = coordinates[np.argsort(-accuracies[tuple(coordinates.T)])]
    top_values = accuracies[tuple(top_heads.T)]

    return top_heads, top_values

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

def stratified_sample(df, stratify_col, cutoff, random_state=None):
    '''
    For dataset reduction: given a dataframe, a column to stratify on, and a cutoff number of samples
    '''
    groups = df[stratify_col].unique()
    n_groups = len(groups)

    base_sample = cutoff // n_groups
    extra = cutoff % n_groups
    sample_counts = {group: base_sample for group in groups}
    for group in np.random.choice(groups, extra, replace=False):
        sample_counts[group] += 1
    actual_counts = {}
    remaining = 0
    eligible_for_redistribution = []

    for group in groups:
        group_df = df[df[stratify_col] == group]
        available = len(group_df)
        desired = sample_counts[group]

        if available < desired:
            actual_counts[group] = available
            remaining += desired - available
        else:
            actual_counts[group] = desired
            eligible_for_redistribution.append(group)

    while remaining > 0 and eligible_for_redistribution:
        np.random.shuffle(eligible_for_redistribution)
        for group in eligible_for_redistribution:
            group_df = df[df[stratify_col] == group]
            if actual_counts[group] < len(group_df):
                actual_counts[group] += 1
                remaining -= 1
                if remaining == 0:
                    break

    samples = []
    for group, n in actual_counts.items():
        group_df = df[df[stratify_col] == group]
        samples.append(group_df.sample(n=n, random_state=random_state))

    return pd.concat(samples).reset_index(drop=True)

def extract(model, data, labels, device, batch_size, attn, half=True):
    model.to(device)
    model.reset_hooks()
    extractor = ActivationExtractor(model=model, data=data, labels=labels, device=device, half=half,
                                      batch_size=batch_size)
    if attn:
        extractor.set_hooks([i for i in range(model.cfg.n_layers)],
                            [tlens.utils.get_act_name('z')], attn=True)
    else:
        extractor.set_hooks(
                            [i for i in range(model.cfg.n_layers)],
                            [tlens.utils.get_act_name('resid_post')], attn=False) # for instance
    activations, labels = extractor.process() # Get
    model.to(t.device('cpu'))
    gc.collect()
    t.cuda.empty_cache()
    return activations, labels