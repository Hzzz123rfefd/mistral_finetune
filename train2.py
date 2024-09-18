import argparse
import os
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

from llmai.model import Mistral
def main(args):
    """get train device"""
    device = args.device if torch.cuda.is_available() else "cpu"

    """ get model """
    if os.path.isdir(args.lora_config_dir):
        model = Mistral(base_model_name_or_path = args.base_model_name_or_path,
                                    peft_config_dir = args.lora_config_dir)
    else:
        os.makedirs(args.lora_config_dir)
        model = Mistral(base_model_name_or_path = args.base_model_name_or_path)
    model.to(device)

    """ get data """
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    """ get args """
    trainning_args = TrainingArguments(
    output_dir=args.lora_config_dir, # 要保存的目录和存储库ID
    num_train_epochs = args.total_epoch,                     # 训练周期数
    per_device_train_batch_size = args.batch_size,          # 训练期间每个设备的批量大小
    gradient_accumulation_steps = 2,          # 反向/更新前的步骤数
    gradient_checkpointing = True,            # 使用渐变检查点来节省内存
    optim="adamw_torch_fused",              # 使用融合的adamw优化器
    logging_steps = 1,                       # 每10步记录一次
    save_strategy="epoch",                  # 每个epoch保存检查点
    learning_rate = args.lr,                     # 学习率，基于QLoRA论文
    bf16=True,                              # 使用bfloat16精度
    tf32=True,                              # 使用tf32精度
    max_grad_norm = 0.3,                      # 基于QLoRA论文的最大梯度范数
    warmup_ratio = 0.03,                      # 根据QLoRA论文的预热比例
    lr_scheduler_type="constant",           # 使用恒定学习率调度器
    push_to_hub=False,                       # 将模型推送到Hub
    report_to="tensorboard",                # 将指标报告到Tensorboard
    )


    trainer = SFTTrainer(
        model = model.base_model,
        args = trainning_args,
        train_dataset = dataset,
        peft_config = model.peft_config,
        max_seq_length = args.max_seq_length,
        tokenizer = model.tokenizer,
        packing = True,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        }
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path",
                        type=str,
                        default = "mistralai/Mistral-7B-Instruct-v0.3")
    
    parser.add_argument("--lora_config_dir",
                        type=str,
                        default = "./saved_model/test2")
    
    parser.add_argument("--data_path",
                        type=str,
                        default = "data/train_dataset.json")
    
    parser.add_argument("--lr",
                        type=float,
                        default = 2e-4)
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=6)
    
    parser.add_argument("--total_epoch",
                        type = int,
                        default = 3)
    
    parser.add_argument("--max_seq_length",
                        type = int,
                        default = 3072)
    
    parser.add_argument("--device",
                        type=str,
                        default = "cuda")
    
    args = parser.parse_args()
    main(args)

