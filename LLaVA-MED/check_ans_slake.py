import json
from rich import print
from dataloader import RadDataset, SlakeDataset
from transformers import AutoTokenizer

output_file = 'recycle_bin/acc/slake_vdz_2l3.txt'

doc = []
with open(output_file, 'r') as f:
    for line in f:
        doc.append(json.loads(line))

dataset = SlakeDataset('test')
tokenizer = AutoTokenizer.from_pretrained('downloaded/llava-med-v1.5-mistral-7b')

to_dict = lambda x: dict(zip(['qid', 'question', 'answer', 'image'], x))
opened_right_count = 0
opened_p_rgt_count = 0
closed_right_count = 0
total_count = 0

open_count = 0
closed_count = 0
for response_item, label_item in zip(doc, dataset):
    label_item = to_dict(label_item)
    assert response_item['qid'] == label_item['qid']
    
    # if dataset.get_answer_type(label_item['qid']) != 'CLOSED':
    #     continue
    
    response = response_item['response']
    label = label_item['answer']
    
    do_clean = lambda x: x.lower().strip()
    
    if dataset.get_answer_type(label_item['qid']) == 'CLOSED':
        if do_clean(response) == do_clean(label):
            closed_right_count += 1
        closed_count += 1
    else:
        if do_clean(response) == do_clean(label):
            opened_right_count += 1
            
        response_ids = tokenizer(response, padding=False, truncation=False, return_attention_mask=False)['input_ids']
        label_ids = tokenizer(label, padding=False, truncation=False, return_attention_mask=False)['input_ids']
        
        response_ids = response_ids[1:]
        label_ids = set(label_ids[1:])
        
        recall = sum([1 for rid in response_ids if rid in label_ids]) / len(response_ids)
        opened_p_rgt_count += recall
        
        open_count += 1
        
    total_count += 1

print(f'[blue]\[acc][/blue]: {(closed_right_count + opened_right_count) / total_count * 100}')
print(f'[blue]\[closed][/blue]: {closed_right_count / closed_count * 100}')
print(f'[blue]\[opened][/blue]: {opened_right_count / open_count * 100}')
print(f'[blue]\[opened_word_recall][/blue]: {opened_p_rgt_count / open_count * 100}')