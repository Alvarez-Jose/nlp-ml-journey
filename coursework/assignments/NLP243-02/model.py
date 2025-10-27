from torch import nn
import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :]
    
    model_name = "gpt2" # decoder only model
    tokeinzer = AutoTokenizer.from_pretrained(model_name)
    tokeinzer.pad_token = tokeinzer.eos_token # ensure pad token is defoned
    model = AutoModelForCausalLM.form_pretrained(model_name)
    #loading the dataset
    dataset = load_dataset('data/train')

    # Begin Tokenize
    '''def tokenize_fn(ex)'''

"""
TODO define your transformer model here. 
this will include: 
    - embed tokens (nn.Embedding)
    - add position encoding (provided)
    - n repetitions of 
        - *masked* self attention (can be single or multi-headed)
        - feedforward (MLP)
        - remember that the layer outputs are added to a residual connection
    - final linear layer with out_features equal to your vocabulary size
"""
def generate_square_subsequent_mask(size: int):
    '''
    function creates a sqaure, uppter - triangular mask typically filed float('-inf-)
    in the upper triangle and 0.0 in the lower triangle (or True and False for boolean mask)
    '''
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask






def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    return

