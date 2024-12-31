import os
import sys

sys.path[0] = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import datetime
import torch
import json
import re

from dataloader import SlakeDataset, Collator, OVQADataset, VQARadDataset
from models.modeling_bridgetower import BridgeTowerForVisualQuestionAnswering
from transformers.models.bridgetower.processing_bridgetower import BridgeTowerProcessor


from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
# from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from peft import PeftModel
from PIL import Image

torch.set_printoptions(precision=3, threshold=10000, linewidth=225)
device = torch.device('cuda')

# dataset = SlakeDataset('test')
# dataset = OVQADataset('test')
dataset = VQARadDataset('test')
model = BridgeTowerForVisualQuestionAnswering.from_pretrained('outputs/rad_vdz')

processor = BridgeTowerProcessor.from_pretrained('pretrained_models/base')
collate_fn = Collator(processor, with_answer=False)

dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
model.requires_grad_(False)
model = model.to(torch.bfloat16).to('cuda')
model.eval()

out_answer = []
out_answer_file_path = 'recycle_bin/rad_vdz.json'

for index, batch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
    for sample in [batch]:
        sample = {k: v.to(device) if hasattr(v, 'to') else v for k, v in sample.items() }
        with torch.no_grad():
            out = model(**sample).logits
            out = out.argmax(dim=-1)
            out = out.tolist()
        text = list(map(dataset.id_to_answer, out))[0]
        
        print(f'=============={index:0>4}=================')
        print(f'[text] : {text}')
        print(f'[label[0]] : {dataset[index]["answer"]}')
        
        
        out_answer.append({
            'index': index,
            'question': dataset[index]['text'],
            'response': text,
            'label': dataset[index]['answer']
        })

with open(out_answer_file_path, 'w', encoding='utf-8') as f:
    json.dump(out_answer, f, indent=2)