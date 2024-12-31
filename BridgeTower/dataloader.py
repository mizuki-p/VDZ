import json
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Dict
import torch


def Collator(processor, with_answer=True):
    def fn(batch):
        out = {}
        temp = {}
        for key in ["text", "answer_id", "images"]:
            temp[key] = [item[key] for item in batch]
        
        out.update(processor(temp['images'], temp['text'], padding=True, return_tensors='pt'))
        
        if with_answer:
            out['labels'] = torch.tensor(temp['answer_id'], dtype=torch.long)
                
        return out

    return fn


class BaseDataset(Dataset):
    def __init__(self, split, name):
        self.split = split
        self.name = name

        self.ds = None
        self.img_root = None
        self.answer_to_id = {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        image_path = item["image_name"]
        image = Image.open(f"{self.img_root}/{image_path}").convert("RGB")

        question = item["question"]
        answer = str(item["answer"])
        if answer.lower().strip() not in self.answer_to_id:
            answer_id = -1
        else:
            answer_id = self.answer_to_id[answer.lower().strip()]
        return {"text": f"Q: {question} A:", "answer": answer, 'answer_id': answer_id, "images": image}
    
    def id_to_answer(self, id):
        if not hasattr(self, "id_to_answer_"):
            self.id_to_answer_ = {v: k for k, v in self.answer_to_id.items()}
        return self.id_to_answer_[id]


class VQARadDataset(BaseDataset):
    def __init__(self, split):
        super(VQARadDataset, self).__init__(split, "vqa_rad")
        split_to_path = {
            "train": "dataset/data_RAD/trainset.json",
            # 'val': '',
            "test": "dataset/data_RAD/testset.json",
        }
        self.img_root = "dataset/data_RAD/images"

        with open(split_to_path[split], "r") as f:
            self.ds = json.load(f)
            
        with open("dataset/data_RAD/answer_to_ids.json", "r") as f:
            self.answer_to_id = json.load(f)


class SlakeDataset(BaseDataset):
    def __init__(self, split):
        super(SlakeDataset, self).__init__(split, "slake")

        split_to_path = {
            "train": "dataset/SLAKE1.0/train.json",
            "val": "dataset/SLAKE1.0/validate.json",
            "test": "dataset/SLAKE1.0/test.json",
        }
        self.img_root = "dataset/SLAKE1.0/imgs"

        with open(split_to_path[split], "r") as f:
            self.ds = json.load(f)

        temp = []
        for item in self.ds:
            if item["q_lang"] == "en":
                item["image_name"] = item["img_name"]
                temp.append(item)
        self.ds = temp
        
        with open("dataset/SLAKE1.0/answer_to_ids.json", "r") as f:
            self.answer_to_id = json.load(f)


class OVQADataset(BaseDataset):
    def __init__(self, split):
        super(OVQADataset, self).__init__(split, "ovqa")
        split_to_path = {
            "train": "dataset/OVQA_publish/trainset.json",
            "val": "dataset/OVQA_publish/valset.json",
            "test": "dataset/OVQA_publish/testset.json",
        }
        self.img_root = "dataset/OVQA_publish/img"

        with open(split_to_path[split], "r") as f:
            self.ds = json.load(f)
            
        with open("dataset/OVQA_publish/answer_to_ids.json", "r") as f:
            self.answer_to_id = json.load(f)


name_to_dataset_cons: Dict[str, BaseDataset] = {
    "vqa_rad": VQARadDataset,
    "slake": SlakeDataset,
    "ovqa": OVQADataset,
}
