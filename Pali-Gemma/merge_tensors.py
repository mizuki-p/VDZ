import torch
from safetensors import safe_open
from safetensors.torch import save_file


projs = torch.load('recycle_bin/projs.pth', map_location='cpu')

adapters = {}
with safe_open('outputs/ir_base/adapter_model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        adapters[key] = f.get_tensor(key)

# projs[i].weight ->
# base_model.model.language_model.model.layers.0.ir.input_proj.weight  # [2048, 2048]

# torch.concat([torch.eye() for _ in range(4)], dim=0) -> 
# base_model.model.language_model.model.layers.0.ir.up_proj.weight # [1024, 256]

for key in adapters:
    if 'ir' in key:
        layer_index = int(key.split('.')[-4])
        if 'input_proj' in key:
            adapters[key].weight = projs[layer_index]
        elif 'up_proj' in key:
            adapters[key].weight = torch.concat([torch.eye(256) for _ in range(4)], dim=0)
        else:
            print('error')

save_file(adapters, 'recycle_bin/new_ada.safetensors', metadata={
    'format': 'pt'
}) 