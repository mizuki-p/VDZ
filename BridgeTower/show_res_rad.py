import json
from rich import print
from dataloader import VQARadDataset


ds = VQARadDataset('test')
with open('recycle_bin/rad_vdz.json', 'r') as f:
    doc = json.load(f)

do_clean = lambda x: str(x).strip().lower()

open_set = []
close_set = []
for item, resp in zip(ds.ds, doc):
    status = do_clean(resp['response']) == do_clean(resp['label'])
    if item['answer_type'] == 'OPEN':
        open_set.append(status)
    else:
        close_set.append(status)


print(f'\[Open Acc]: {sum(open_set) / len(open_set)}')
print(f'\[Closed Acc]: {sum(close_set) / len(close_set)}')
print(f'\[Overall Acc]: {(sum(open_set) + sum(close_set)) / (len(open_set) + len(close_set))}')