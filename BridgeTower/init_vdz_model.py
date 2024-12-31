import torch
from models.modeling_bridgetower import BridgeTowerForVisualQuestionAnswering

# model : BridgeTowerForVisualQuestionAnswering = BridgeTowerForVisualQuestionAnswering.from_pretrained('pretrained_models/bridgetower-base-itm-mlm')
# del model.classify_head
# torch.save(model.state_dict(), 'pretrained_models/ir_base/pytorch_model.bin')

doc = torch.load('pretrained_models/vdz_base/pytorch_model.bin', map_location='cpu')
keys = list(doc.keys())
# keys.sort()

# with open('./keys_ori.txt', 'w') as f:
#     for key in keys:
#         f.write(f'{key}\n')


# bridgetower.cross_modal_image_layers.0.attention.output.dense.weight -> bridgetower.cross_modal_image_layers.0.ir.input_proj.weight
# bridgetower.cross_modal_image_layers.0.attention.output.dense.bias -> bridgetower.cross_modal_image_layers.0.ir.input_proj.bias
# bridgetower.cross_modal_image_layers.0.ir.up_proj.weight

for key in keys:
    if 'ir' in key:
        if 'input_proj' in key:
            layer_idx = key.split('.')[2]
            if 'weight' in key:
                doc[key] = doc[f'bridgetower.cross_modal_image_layers.{layer_idx}.attention.output.dense.weight']
            else:
                doc[key] = doc[f'bridgetower.cross_modal_image_layers.{layer_idx}.attention.output.dense.bias']
        else:
            new_tensor = torch.concat([torch.eye(64) for _ in range(4)], dim=0)
            assert new_tensor.shape == doc[key].shape
            doc[key] = new_tensor

torch.save(doc, 'pretrained_models/vdz_base/pytorch_model.bin')

