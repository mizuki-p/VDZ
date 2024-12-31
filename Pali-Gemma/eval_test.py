import os
import sys

sys.path[0] = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import datetime
import torch
import json
import re

from dataloader import SlakeDataset, Collator, OVQADataset, VQARadDataset
from modeling.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from peft import PeftModel

torch.set_printoptions(precision=3, threshold=10000, linewidth=225)
device = torch.device('cuda')

# dataset = SlakeDataset('test')
# dataset = OVQADataset('test')
dataset = VQARadDataset('test')
model = PaliGemmaForConditionalGeneration.from_pretrained('models/paligemma-3b-ft-vqav2-448')
model = model.to(torch.bfloat16).to(device)
model = PeftModel.from_pretrained(model, 'outputs/vqarad_vdz', is_trainable=False)
processor = PaliGemmaProcessor.from_pretrained('models/paligemma-3b-ft-vqav2-448')
collate_fn = Collator(processor, with_suffix=False)

dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
model.requires_grad_(False)
model.eval()

out_answer = []
out_answer_file_path = 'recycle_bin/vqarad_vdz.json'

for index, batch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
    for sample in [batch]:
        sample = {k: v.to(device) if hasattr(v, 'to') else v for k, v in sample.items() }
        with torch.no_grad():
            out = model.generate(
                **sample,
                # stopping_criteria = StoppingCriteriaList([
                #     MaxLengthCriteria(sample['input_ids'].shape[1] + 40 + 1024),
                # ]),
                max_new_tokens=30,
                # max_length=1200,
                use_cache=False
            )

            
        # out, logits = out['sequences'], out['logits']
        output_ids = out.to('cpu')
        text = processor.batch_decode(output_ids)
        del output_ids

        text = text[0]
        text = text.replace('<image>'*1024, '')
        _u = re.match(r'<bos>.*?\n(.+)<eos>', text)
        if _u is not None:
            text = _u.groups()[0]
        
        
        print(f'=============={index:0>4}=================')
        print(f'[text] : {text}')
        print(f'[label[0]] : {dataset[index]["suffix"]}')
        
        
        out_answer.append({
            'index': index,
            'question': dataset[index]['text'],
            'response': text,
            'label': dataset[index]['suffix']
        })

with open(out_answer_file_path, 'w', encoding='utf-8') as f:
    json.dump(out_answer, f, indent=2)