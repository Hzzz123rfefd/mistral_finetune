# mistral_finetune
finetune mistral with code
## Installation
Operating System: Linux
```bash
conda create -n mistral_finetune python=3.10
conda activate mistral_finetune
git clone https://github.com/Hzzz123rfefd/mistral_finetune.git
cd mistral_finetune
pip install -r requirements.txt
```
## Usage
### finetune
1、prepare finetune data,using jsonl format
```jsonl
{"prompt": "", "prompt_id": "", "messages": [{"content": "", "role": "user"}, {"content": "", "role": "assistant"}]}
{"prompt": "", "prompt_id": "", "messages": [{"content": "", "role": "user"}, {"content": "", "role": "assistant"}]}
....
{"prompt": "", "prompt_id": "", "messages": [{"content": "", "role": "user"}, {"content": "", "role": "assistant"}]}
```
* sh
```bash
python train.py --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                --lora_config_dir {finetune model dir path} \
                --data_path {finetune data path} \
                --lr {learnning rate} \
                --weight_decay {regularized weight}\
                --factor {learnning rate attenuation coefficient} \
                --patience {learnning rate attenuation threshold} \
                --batch_size 6 \
                --total_epoch 1000 \
                --device "cuda"
```
* example
```bash
python train.py --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                --lora_config_dir "./saved_model/test" \
                --data_path "data/train.jsonl" \
                --lr 1e-4 \
                --weight_decay 0.1\
                --factor 0.3 \
                --patience 10 \
                --batch_size 6 \
                --total_epoch 1000 \
                --device "cuda"
```
### chat
#### chat use prompt
* sh
```bash
python example/chat.py --prompt {prompt} \
                       --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                       --lora_config_dir {finetune model dir path,If you haven't made any finetune, just set it to None} \
                       --max_new_tokens {The maximum number of tokens generated} \
                       --device "cuda"
```
* example
```bash
python example/chat.py --prompt "你是一名数据库专家" \
                       --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                       --lora_config_dir "saved_model/finetune" \
                       --max_new_tokens 256 \
                       --device "cuda"
```
#### chat with context
1、 prepare context data, using json format
```json
{
    "messages":
    [
        {"content": "question1","role": "user"},
        {"content": "answer1","role":"assistant"},
        {"content": "question2","role": "user"},
        .....,
    ]
}
```
* sh
```bash
python example/chat_with_context.py --context_path {context json file path} \
                                    --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                                    --lora_config_dir {finetune model dir path,If you haven't made any finetune, just set it to None} \
                                    --max_new_tokens {The maximum number of tokens generated} \
                                    --device "cuda"
```
* example
```bash
python example/chat_with_context.py --context_path "data/context.json" \
                                    --base_model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" \
                                    --lora_config_dir "saved_model/finetune" \
                                    --max_new_tokens 256 \
                                    --device "cuda"
```