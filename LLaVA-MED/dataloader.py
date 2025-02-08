import os
import json
import torch
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import Conversation
from typing import Union, Literal
from torch.utils.data.dataset import Dataset
from llava.mm_utils import tokenizer_image_token, process_images
from torch.nn.utils.rnn import pad_sequence


class RadDataset(Dataset):
    def __init__(self, split: Union[Literal['train'], Literal['test']]):
        super(RadDataset, self).__init__()
        
        with open('dataset/data_RAD/trainset.json', 'r') as f:
            self.trainset = json.load(f)
        
        with open('dataset/data_RAD/testset.json', 'r') as f:
            self.testset = json.load(f)
            
        self.img_folder = 'dataset/data_RAD/images'
        
        self.split = split
    
    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.trainset)
        else:
            return len(self.testset)
        
    
    def __getitem__(self, idx) -> tuple[int, str, str, Image.Image]:
        if self.split == 'train':
            item = self.trainset[idx]
        else:
            item = self.testset[idx]
            
        qid = item['qid']
        question = item['question']
        answer = item['answer']
        
        image_path = item['image_name']
        image = Image.open(os.path.join(self.img_folder, image_path)).convert('RGB')
        
        return qid, question, answer, image
    
    def get_answer_type(self, qid):
        if not hasattr(self, 'qid_to_type'):
            ds = self.trainset + self.testset
            self.qid_to_type = {item['qid']: item['answer_type'] for item in ds}
        
        return self.qid_to_type[qid]
    

class SlakeDataset(Dataset):
    def __init__(self, split: Union[Literal['train'], Literal['val'], Literal['test']]):
        super(SlakeDataset, self).__init__()
        
        self.img_folder = 'dataset/SLAKE1.0/imgs'
        self.split = split
        
        self.paths = {
            'train': 'dataset/SLAKE1.0/train.json',
            'val':   'dataset/SLAKE1.0/validate.json',
            'test':  'dataset/SLAKE1.0/test.json'
        }
        
        with open(self.paths[split], 'r') as f:
            self.dataset = json.load(f)
        self.dataset = [item for item in self.dataset if item['q_lang'] == 'en']
    
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx) -> tuple[int, str, str, Image.Image]:
        item = self.dataset[idx]
            
        qid = item['qid']
        question = item['question']
        answer = item['answer']
        
        image_path = item['img_name']
        image = Image.open(os.path.join(self.img_folder, image_path)).convert('RGB')
        
        return qid, question, answer, image
    
    def get_answer_type(self, qid):
        if not hasattr(self, 'qid_to_type'):
            self.qid_to_type = {item['qid']: item['answer_type'] for item in self.dataset}
        return self.qid_to_type[qid]
    
    
class OvqaDataset(Dataset):
    def __init__(self, split: Union[Literal['train'], Literal['val'], Literal['test']]):
        super(OvqaDataset, self).__init__()
        
        self.img_folder = 'OVQA_publish/img'
        self.split = split
        self.paths = {
            'train': 'OVQA_publish/trainset.json',
            'val':   'OVQA_publish/valset.json',
            'test':  'OVQA_publish/testset.json'
        }
        
        with open(self.paths[split], 'r') as f:
            self.dataset = json.load(f)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[int, str, str, Image.Image]:
        item = self.dataset[index]
        
        qid = item['qid']
        question = item['question']
        answer = item['answer']
        
        image_path = item['image_name']
        image = Image.open(os.path.join(self.img_folder, image_path)).convert('RGB')
        
        return qid, question, answer, image
    
    def get_answer_type(self, qid):
        if not hasattr(self, 'qid_to_type'):
            self.qid_to_type = {item['qid']: item['answer_type'] for item in self.dataset}
        return self.qid_to_type[qid]


def get_dataset_constructor(dataset_name: Literal['rad', 'slake', 'ovqa']):
    dataset_name = dataset_name.lower()
    if dataset_name == 'rad':
        return RadDataset
    elif dataset_name == 'slake':
        return SlakeDataset
    elif dataset_name == 'ovqa':
        return OvqaDataset
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    
class DataCollator:
    def __init__(
        self,
        tokenizer,
        split: Union[Literal['train'], Literal['test']],
        conversation_template: Conversation,
        pad_token_id: int,
        image_processor: object,
        model_config: object,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.conversation_template = conversation_template
        self.pad_token_id = pad_token_id
        self.image_processor = image_processor
        self.model_config = model_config
        
    def __call__(
        self,
        rows
    ):
        if self.split == 'train':
            return self._collate_train(rows)
        elif self.split == 'test':
            return self._collate_test(rows)
        else:
            raise ValueError(f'Invalid split: {self.split}')
        
    def _collate_train(self, rows):
        input_ids_list = []
        labels_list = []
        images = []
        for row in rows:
            qid, question, answer, image = row
            images.append(image)
            
            question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
            
            conv = self.conversation_template.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prefix = conv.get_prompt()
            
            conv = self.conversation_template.copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
            full = conv.get_prompt()
            
            prefix = tokenizer_image_token(prefix, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            full = tokenizer_image_token(full, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            
            prefix_length = prefix.size(0)
            full_length = full.size(0)
            
            input_ids = full
            labels = full.clone()
            labels[:prefix_length] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # make padding and attention mask
        pad_value = -114514
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_value)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=pad_value)
        attention_mask = (input_ids != pad_value).long()
        
        input_ids[input_ids == pad_value] = self.pad_token_id
        labels[labels == pad_value] = self.pad_token_id
        
        images : torch.Tensor = process_images(images, self.image_processor, self.model_config).to(torch.bfloat16)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'images': images
        }
    
    def _collate_test(self, rows):
        pass
        # It is a wrong design, there is no need to implement this function.
        # But I will not remove it because I don't know whether some other functions will call it. perhaps not.
            