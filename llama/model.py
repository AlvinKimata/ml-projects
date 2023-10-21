import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim:int = 4096 #ctx size
    n_layers: int = 32
    n_heads: int = 32 #Number of attention heads.
    n_kv_heads: Optional[int] = None #Number of heads for k and v
    vocab_size:int = -1 #Set when we load the tokenizer.
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    #Needed for k v cache.
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class Transformer(nn.Module):
	def __init(self, args = ModelArgs):
		super().__init__()
		assert args.vocab_size != 1, "Vocabulary size must be set."
		self.args = args
		self.vocab_size = args.vocab_size
		self.n_layers = args.n_layers
		self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
		self.layers = nn.ModuleList()

		for _ in args.n_layers:
			self.layers.append(EncoderBlock(args))

		self.norm = RMSNorm(args.dim, eps = args.norm_eps)
		self.output = nn.Linear(args.dim, self.vocab_size, bias = False)
		