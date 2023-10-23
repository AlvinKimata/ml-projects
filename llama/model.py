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

def precompute_theta_pos_frequencies(head_dim:int, seq_len: int, device:str, theta: float = 10000.0):
	assert head_dim % 2 == 0, "Dimension must be divisible by 2."
	theta_numerator = torch.arange(0, head_dim, 2).float()
	theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
	m = torch.arange(seq_len, device = device)
	freqs = torch.outer(m, theta).float()
	freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str):
	x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
	freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
	x_rotated = x_complex * freqs_complex
	x_out = torch.view_as_real(x_rotated)
	x_out = x_out.reshape(*x.shape)
	return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-4):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x: torch.Tensor):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
	
	def forward(self, x: torch.Tensor):
		return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()

		#Number of heads for keys and values.
		self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

		#Number of heads for queries.
		self.n_heads_q = args.n_heads
		self.n_rep = self.n_heads_q // self.n_kv_heads
		self.head_dim = args.dim // args.h_heads

		self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
		self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)
		self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)
		self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

		self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
		args.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

	def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
		batch_size, seq_len, _ = x.shape #(B, 1, Dim)

		#Apply linear projections.
		xq = self.wq(x)
		xk = self.wk(x)
		xv = self.wv(x)

		xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
		xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
		xv = xv.view(batch_size, self.n_kv_heads, self.head_dim)

		




class EncoderBlock(nn.Module):
	def __init__(self , args: ModelArgs):
		super().__init__()

		self.n_heads = args.n_heads
		self.dim = args.dim
		self.head_dim = args.dim // args.h_heads

		self.attention = SelfAttention(args)
		self.feed_forward = FeedForward(args)

		#Normalization before the self attention.
		self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
		
		#Normalization before the feed-forward block.
		self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

	def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
		h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
		out = h + self.feed_forward.forward(self.ffn_norm(h))

		return out

		
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

		self.freqs_complex = precompute_theta_pos_frequencies(self.args_dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)


	def forward(self, tokens: torch.Tensor, start_pos: int):
		batch_size, seq_len = tokens.shape

		assert seq_len == 1, "Only one token at a time can be processed"

		#Compute embeddings of input.
		h = self.tok_embeddings(tokens)

		#Retrieve the pairs corresponding to the positions [start_pos, start_pos + seq_len]
		freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

		#Pass embeddings to the model layers.
		for layer in self.layers:
			h = layer(h, start_pos, freqs_complex)

		#Apply normalization.
		h = self.norm(h)
		
		#Apply grads to a linear layer and softmax to get output.
		output = self.output(h).float()
		return output