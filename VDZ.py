import torch
import math
from torch import nn
from transformers.activations import ACT2FN

# ImageWidth, ImageHeight = 32, 32
ImageEdge = 32
ScaleRatio = 2


class FakeConfig:
    '''
        Here we use FakeConfig to simulate the actual Config class used in different models (Gemmaconfig, MistralConfig, BridgeTowerConfig).
    '''
    hidden_size         = 768
    num_attention_heads = 12
    attention_dropout   = 0.1
    hidden_act          = 'gelu'


class VDZ(nn.Module):
    def __init__(self, config: FakeConfig, layer_idx):
        super(VDZ, self).__init__()
        
        self.layer_idx = layer_idx
        
        self.vision_tokens = ImageEdge * ImageEdge
        self.scale_ratio = ScaleRatio
        self.image_edge = ImageEdge
        self.pool_kernel_size = ImageEdge // 2
        self.pooled_edge = 1 + ImageEdge - self.pool_kernel_size
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        self.dropout = nn.Dropout(self.attention_dropout)
        
        self.pool = nn.AvgPool2d(
            kernel_size=self.pool_kernel_size,
            stride=1,
            padding=0
        )
        
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_act = ACT2FN[config.hidden_act]
        
        self.up_proj = nn.Linear(self.head_dim, int(self.head_dim * self.scale_ratio * self.scale_ratio), bias=False)

    def forward(
        self,
        image_hidden_states, 
        text_hidden_states,
        image_mask = None,
    ):
        # prepare for residual connection
        residual = image_hidden_states
        
        # <start> calculate attention scores for visual embeddings
        batch_size, seq_len, hidden_size = image_hidden_states.shape
        _, q_seq_len, _ = text_hidden_states.shape
        q_states = text_hidden_states.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = image_hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q_states, k_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn_weights = torch.cdist(q_states, k_states) 
        
        if image_mask is not None:
            attn_weights += image_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # <end> calculate attention scores for visual embeddings

        # <start> find the most important area
        image_attn = torch.sum(attn_weights, dim=2)
        image_states = image_hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        image_attn = image_attn.reshape(batch_size, self.num_heads, self.image_edge, self.image_edge)
        image_attn = self.pool(image_attn)
        image_attn = image_attn.reshape(batch_size, self.num_heads, -1)
        image_attn_max_index = torch.argmax(image_attn, dim=-1)
        
        mask = []
        for i in range(batch_size):
            _mask = []
            for j in range(self.num_heads):
                n = image_attn_max_index[i, j]
                row, col = n // self.pooled_edge, n % self.pooled_edge
                temp = torch.zeros(1, 1, self.image_edge, self.image_edge, dtype=torch.bool).to(image_hidden_states.device)
                temp[:, :, row: row + self.pool_kernel_size, col: col + self.pool_kernel_size] = True
                _mask.append(temp)
            _mask = torch.cat(_mask, dim=1)
            mask.append(_mask)
        mask = torch.cat(mask, dim=0)
        # <end> mask denotes the most important area
        
        # <start> downsample and upsample the most important area
        image_states = image_states.reshape(batch_size, self.num_heads, self.image_edge, self.image_edge, self.head_dim)
        mask = mask.unsqueeze(-1).expand_as(image_states)
        image_states = image_states[mask].reshape(batch_size, self.num_heads, self.pool_kernel_size, self.pool_kernel_size, self.head_dim)
        image_states = self.up_proj(image_states) # [batch_size, num_heads, pooled_width, pooled_height, head_dim * ratio * ratio]
        image_states = image_states.reshape(batch_size, self.num_heads, self.pool_kernel_size, self.pool_kernel_size, 2, 2, self.head_dim)
        image_states = image_states.transpose(3, 4).reshape(batch_size, self.num_heads, self.image_edge * self.image_edge, self.head_dim)
        
        image_states = image_states.transpose(1, 2).reshape(batch_size, self.image_edge * self.image_edge, hidden_size)
        
        image_hidden_states = image_states
        # <end> downsample and upsample the most important area
        
        # <start> ffn and residual connection
        image_hidden_states = self.input_proj(image_hidden_states)
        image_hidden_states = self.input_act(image_hidden_states)

        image_hidden_states = residual + image_hidden_states
        # <end> ffn and residual connection
        
        return image_hidden_states
