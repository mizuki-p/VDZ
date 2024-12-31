import json
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Dict


def Collator(processor, with_suffix=True):
    def fn(batch):
        out = {}
        for key in ["text", "suffix", "images"]:
            out[key] = [item[key] for item in batch]
        if not with_suffix:
            out.pop("suffix")
        out = processor(**out, padding=True, return_tensors="pt")
        return out

    return fn


class BaseDataset(Dataset):
    def __init__(self, split, name):
        self.split = split
        self.name = name

        self.ds = None
        self.img_root = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        image_path = item["image_name"]
        image = Image.open(f"{self.img_root}/{image_path}").convert("RGB")

        question = item["question"]
        answer = str(item["answer"])
        return {"text": f"answer en {question}", "suffix": answer, "images": image}


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


name_to_dataset_cons: Dict[str, BaseDataset] = {
    "vqa_rad": VQARadDataset,
    "slake": SlakeDataset,
    "ovqa": OVQADataset,
}
