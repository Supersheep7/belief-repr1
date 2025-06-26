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