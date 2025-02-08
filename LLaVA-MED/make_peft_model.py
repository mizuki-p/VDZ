import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import torch
from peft import get_peft_model, LoraConfig

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates

torch.set_printoptions(precision=3, threshold=10000, linewidth=225)

conv = conv_templates['mistral_instruct']

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path='downloaded/llava-med-v1.5-mistral-7b',
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b'
)

model = get_peft_model(model, LoraConfig(
    r=128,
    target_modules=r'model\.layers\.\d+\.((self_attn\.(q|k|v|o))|(mlp\.(gate|up|down)))_proj',
    modules_to_save=['mm_projector', 'vdz']
))

model.save_pretrained('outputs/vdz_base')