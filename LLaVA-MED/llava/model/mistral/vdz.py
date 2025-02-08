import torch
from torch import nn
from transformers.activations import ACT2FN
from .configuration_mistral import MistralConfig

def if_add_vdz_module(layer_index):
    return True and layer_index % 6 == 0

class ImageRearrangement(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx):
        super(ImageRearrangement, self).__init__()
        
        self.layer_idx = layer_idx
        
        self.vision_tokens = 24 * 24
        self.scale_ratio = 2
        self.image_edge = 24
        self.pool_kernel_size = 24 // 2
        self.pooled_edge = 1 + 24 - self.pool_kernel_size
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        
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
        hidden_states: torch.Tensor
    ):
        residual = hidden_states
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        q_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        attn_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        attn_mask[:, 5: 5 + 24 * 24] = torch.finfo(attn_weights.dtype).min
        attn_weights += attn_mask.reshape(batch_size, 1, seq_len, 1).expand_as(attn_weights)
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # [batch_size, num_heads, seq_len, seq_len]
        
        attn_weights = torch.sum(attn_weights, dim=2)
        image_attn = attn_weights[:, :, 5: 5 + 24 * 24] # [batch_size, num_heads, image_token_len]
        
        image_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        image_states = image_states[:, :, 5: 5 + 24 * 24, :]
        
        image_attn_softmax = nn.functional.softmax(image_attn, dim=-1)
        image_attn_softmax = image_attn_softmax.reshape(batch_size, self.num_heads, 24, 24)
        
        image_attn = image_attn.reshape(batch_size, self.num_heads, 24, 24)
        image_attn = self.pool(image_attn)
        image_attn = image_attn.reshape(batch_size, self.num_heads, -1)
        image_attn_max_index = torch.argmax(image_attn, dim=-1) # [batch_size, num_heads]
        
        mask = []
        for i in range(batch_size):
            _mask = []
            for j in range(self.num_heads):
                n = image_attn_max_index[i, j]
                row, col = n // self.pooled_edge, n % self.pooled_edge
                temp = torch.zeros(1, 1, 24, 24, dtype=torch.bool)
                temp[:, :, row: row + self.pool_kernel_size, col: col + self.pool_kernel_size] = True
                _mask.append(temp)
            _mask = torch.concat(_mask, dim=1)
            mask.append(_mask)
        mask = torch.concat(mask, dim=0)
        
        image_attn_weight = image_attn_softmax[mask].reshape(batch_size, self.num_heads, -1).sum(-1)
        
        image_states = image_states.reshape(batch_size, self.num_heads, 24, 24, self.head_dim)
        mask = mask.unsqueeze(-1).expand_as(image_states)
        image_states = image_states[mask].reshape(batch_size, self.num_heads, self.pool_kernel_size, self.pool_kernel_size, self.head_dim)
        image_states = self.up_proj(image_states)
        image_states = image_states.reshape(batch_size, self.num_heads, self.pool_kernel_size, self.pool_kernel_size, self.scale_ratio, self.scale_ratio, self.head_dim)
        image_states = image_states.transpose(3, 4).reshape(batch_size, self.num_heads, 24*24, self.head_dim)

        image_states *= image_attn_weight.reshape(batch_size, self.num_heads, 1, 1).expand_as(image_states)
        image_states = image_states.transpose(1, 2).reshape(batch_size, 24*24, self.hidden_size)
        
        hidden_states = torch.concat([
            hidden_states[:, :5],
            image_states,
            hidden_states[:, 5 + 24 * 24:]
        ], dim=1)
        
        hidden_states = self.input_proj(hidden_states)
        hidden_states = self.input_act(hidden_states)
        
        hidden_states = residual + hidden_states
        return hidden_states
