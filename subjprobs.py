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
from typing import List, Tuple, Dict, Tensor
import numpy as np
import functools
import openai
from shots import get_shots
from intervention import generate
import re

def self_reporting_probs(model: HookedTransformer,
                         prompt: str,
                         context: str = 'You will receive a statement. You will be asked whether the answer is correct. Answer whether is True or False, and then provide a confidence score between 0 and 1. I will provide three examples. \n',
                         shots: List[str] = ['Paris is the capital of France. This statement is: True. Confidence: 0.95\n', 
                                             'The largest bear in the world is currently in Italy. This statement is: False. Confidence: 0.70\n', 
                                             'Milan is the capital of Italy. This statement is: False. Confidence: 0.85\n'],
                         device: str = "cuda") -> Float:
    """
    Computes the self-reporting probabilities for a batch of tokens using a pre-trained model.
    """

    context = context
    shots = shots
    full_prompt = context + shots + prompt
    answer = generate(full_prompt, model, device=device, max_length=75)
    match = re.search(r'\d+\.\d+', answer)
    if match:
        confidence = float(match.group())
    else:
        confidence = 'NaN'

    return confidence