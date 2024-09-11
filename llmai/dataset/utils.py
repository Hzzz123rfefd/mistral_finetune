from functools import partial
import os
import sys
import torch
sys.path.append(os.getcwd())
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer
import os
from dataclasses import dataclass, field
from typing import Optional
import datasets



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


class MyDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 train = True,  
                 train_size = 0.7):
        # load data from jsonl
        self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        return self.dataset[item]['messages']

def collate_fn(batch,tokenizer:PreTrainedTokenizer):
    encodeds = tokenizer.apply_chat_template(batch, return_tensors="pt",padding=True,return_dict=True)
    return encodeds

if __name__ == "__main__":
    data_arg = DataArguments(data_path = "data/train.jsonl")
    data = MyDataset(data_path = "data/train.jsonl")
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer:PreTrainedTokenizer
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left"
    collate_fn_ = partial(collate_fn, tokenizer=tokenizer)
    train_dataloader = DataLoader(data, 
                              batch_size=6, 
                              shuffle=True,
                              collate_fn=collate_fn_)
    
    for batch_id, batch_data in enumerate(train_dataloader):
        print(batch_id)
        print(batch_data)