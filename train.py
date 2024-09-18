import argparse
from functools import partial
import math
import os
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import torch.nn as nn
from torch import nn,optim
from torch.utils.data import DataLoader

from llmai.model import *
from llmai.dataset import *
import warnings

warnings.filterwarnings("ignore", message=".*past_key_values as a tuple and this is deprecated.*")
class ModelLoss(nn.Module):
    """model loss"""
    def __init__(self):
        super().__init__()
        self.cross = nn.CrossEntropyLoss(reduction='none')

    def forward(self,logits,labels,mask):
        logits = logits[..., :-1, :]
        targets = labels[..., 1:]
        mask = mask[..., 1:]
        B, T, C = logits.shape
        logits = logits.reshape(B*T, C)
        targets = targets.reshape(-1)
        loss = self.cross(logits, targets)
        mask_flat = mask.reshape(-1).float()
        masked_loss = loss * mask_flat
        final_loss = masked_loss.sum() / mask_flat.sum()
        out = {"loss":final_loss}
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + "_best" + filename[-4:])

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_one_epoch(
    epoch,train_dataloader, model, optimizer,criterion,clip_max_norm,log_path
):
    model.train()
    device = next(model.parameters()).device
    pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
    total_loss = AverageMeter()
    average_hit_rate = AverageMeter()
    """ get data """
    for batch_id, batch_data in enumerate(pbar):
        batch_data = batch_data.to(device)
        """ grad zeroing """
        optimizer.zero_grad()

        """ forward """
        used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
        output = model(batch_data)

        # generated_ids = model.backbone.generate(**batch_data, max_new_tokens=1000, do_sample=False)
        # decoded = tokenizer.batch_decode(generated_ids)
        # print(decoded[0])

        """ calculate loss """
        logits = output["logits"].contiguous()
        out_criterion = criterion(logits,batch_data["input_ids"],batch_data["attention_mask"])
        after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
        out_criterion["loss"].backward()
        total_loss.update(out_criterion["loss"])
        average_hit_rate.update(math.exp(-total_loss.avg))

        """ grad clip """
        if clip_max_norm > 0:
            clip_gradient(optimizer,clip_max_norm)

        """ modify parameters """
        optimizer.step()
        postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
            total_loss.avg, 
            average_hit_rate.avg,
            after_used_memory - used_memory
        )
        pbar.set_postfix_str(postfix_str)
        pbar.update()
    with open(log_path, "a") as file:
        file.write(postfix_str+"\n")

    # text = """Prompt to be observed"""
    # answer = tokenizer.decode(
    #     model.backbone.generate(
    #         **tokenizer(text, return_tensors='pt').to(device),
    #         max_new_tokens=256,
    #         pad_token_id=tokenizer.pad_token_id,
    #     )[0]
    # )
    # print(answer)
    # with open(args.lora_config_dir + '/train.log', "a") as file:
    #     file.write(answer+"\n")


def test_epoch(epoch, test_dataloader, model, criterion,log_path):
    total_loss = AverageMeter()
    average_hit_rate = AverageMeter()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_dataloader):
            batch_data = batch_data.to(device)

            """ forward """
            output = model(batch_data)

            """ calculate loss """
            logits = output["logits"].contiguous()
            out_criterion = criterion(logits,batch_data["input_ids"],batch_data["attention_mask"])
            total_loss.update(out_criterion["loss"])
            average_hit_rate.update(math.exp(-total_loss.avg))
    str = (        
        f"Test epoch {epoch}:"
        f"average_hit_rate:{average_hit_rate.avg:.4f} "
        f"total_loss: {total_loss.avg:.4f}\n"
        )
    print(str)
    with open(log_path, "a") as file:
        file.write(str+"\n")
    return total_loss.avg


def main(args):
    """get train device"""
    device = args.device if torch.cuda.is_available() else "cpu"

    """get net struction"""
    if os.path.isdir(args.lora_config_dir):
        model = Mistral(base_model_name_or_path = args.base_model_name_or_path,
                                    peft_config_dir = args.lora_config_dir)
    else:
        os.makedirs(args.lora_config_dir)
        model = Mistral(base_model_name_or_path = args.base_model_name_or_path)
    model.to(device)

    """get data loader"""
    train_datasets = MyDataset(data_path = args.data_path)
    test_dataset = MyDataset(data_path = args.data_path)
    collate_fn_ = partial(collate_fn, 
                          tokenizer=model.tokenizer)
    train_dataloader = DataLoader(train_datasets, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              collate_fn=collate_fn_)
    
    test_dataloader = DataLoader(test_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              collate_fn=collate_fn_)

    """get model loss criterion"""
    criterion = ModelLoss()

    """get optimizer"""
    optimizer = optim.AdamW(params = model.parameters(),
                            lr = args.lr,
                            weight_decay= args.weight_decay)
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                        mode = "min", 
                                                        factor = args.factor, 
                                                        patience = args.patience)
    checkpoint_path = args.lora_config_dir + '/checkpoint.pth'
    log_path = args.lora_config_dir + '/train.log'

    if not os.path.exists(checkpoint_path):   
        save_checkpoint(
            state = {
                "epoch": -1,
                "loss": float("inf"),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best = True,
            filename = checkpoint_path,
        )

    if not os.path.exists(log_path):   
        with open(log_path, "w") as file:
            pass

    checkpoint = torch.load(checkpoint_path, map_location=device)
    last_epoch = checkpoint["epoch"] + 1
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    """ inference """
    best_loss = checkpoint["loss"]
    for epoch in range(last_epoch,args.total_epoch):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(epoch = epoch,
                        train_dataloader = train_dataloader, 
                        model = model, 
                        optimizer = optimizer,
                        criterion = criterion,
                        clip_max_norm = 0.5,
                        log_path = log_path)
        loss = test_epoch(epoch = epoch,
                                    test_dataloader = test_dataloader,
                                    model = model,
                                    criterion = criterion,
                                    log_path = log_path)
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            state = {
                "epoch": epoch,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            },
            is_best = is_best,
            filename = checkpoint_path,
            )
        if is_best:
            model.save_pretrained(args.lora_config_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path",
                        type=str,
                        default = "mistralai/Mistral-7B-Instruct-v0.3")
    
    parser.add_argument("--lora_config_dir",
                        type=str,
                        default = "./saved_model/test")
    
    parser.add_argument("--data_path",
                        type=str,
                        default = "data/train.jsonl")
    
    parser.add_argument("--lr",
                        type=float,
                        default=0.00001)
    
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1)

    parser.add_argument("--factor",
                        type=float,
                        default=0.3)
    
    parser.add_argument("--patience",
                        type=int,
                        default=10)   
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=6)
    
    parser.add_argument("--total_epoch",
                        type = int,
                        default = 1000)
    
    parser.add_argument("--device",
                        type=str,
                        default = "cuda")
    
    args = parser.parse_args()
    main(args)