import os
import sys
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
import pandas as pd
from trl import SFTTrainer, SFTConfig
import matplotlib.pyplot as plt
import json

training_data = []
with open("processed_training_dataset.jsonl", encoding="utf-8") as file:
    for line in file:
        features = json.loads(line)
        # Format the entire example as a single string.
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        training_data.append(template.format(**features))

# 載入驗證集
validation_data = []
with open("evaluation_dataset.jsonl", encoding="utf-8") as file:
    for line in file:
        features = json.loads(line)
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        validation_data.append(template.format(**features))


# Define model names
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "lianghsun/Llama-3.2-Taiwan-3B-Instruct"

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
)

# You must disable the cache to prevent issues during training
model.config.use_cache = False
model.to("cuda")

# Load the Gemma tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right" 
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to be the same as the eos token

# 定義 system prompt
system_prompt = "你是一個桌球規則專家。根據桌球規則，提供簡潔且準確的回答。"\

# Define the prompt for inference
prompt = template.format(
    instruction=f"{system_prompt}\n\n桌球發球時，拋球10公分是犯規嗎?",
    response="",
)

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate the response
outputs = model.generate(
    inputs.input_ids,
    max_length=128,
    num_return_sequences=1,
    do_sample=False
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response)


# 將資料轉換為 Hugging Face Dataset 格式
train_dataset = Dataset.from_dict({"text": training_data})
val_dataset = Dataset.from_dict({"text": validation_data})

max_length_in_data = max(len(tokenizer(sample)["input_ids"]) for sample in training_data)
print(f"訓練資料的最大長度為: {max_length_in_data}")

# 定義資料處理函數
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length_in_data,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # 添加 labels
    return tokenized

# 預處理資料
tokenized_dataset = train_dataset.map(preprocess_function, batched=True)
val_tokenized_dataset = val_dataset.map(preprocess_function, batched=True)

# 定義 LoRA 配置
peft_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj"]
)

# 定義微調參數
training_arguments = SFTConfig(
    output_dir=r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\fine_tuned_model_chinese_5_1k",
    overwrite_output_dir=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-5,
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="none",
    seed=42
)

# 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    peft_config=peft_config,
    args=training_arguments,
    # 程式使用的是 PyTorch,因此需要將數據轉換為 PyTorch(pt) 張量格式
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt") 
)

# 開始計時
start_time = time.time()

# 開始微調
trainer.train()

# 結束計時
end_time = time.time()

# 計算總時間
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"微調完成，總耗時: {hours} 小時 {minutes} 分鐘 {seconds} 秒")

# 列印 Log History
print("Log History:")
for log in trainer.state.log_history:
    print(log)

# 提取每個 epoch 的 Loss 日誌
epoch_loss_history = []

# 提取訓練 Loss
for log in trainer.state.log_history:
    if "loss" in log and "epoch" in log:
        epoch_loss_history.append(log["loss"])

# 提取驗證 Loss
epoch_eval_loss_history = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
print("驗證 Loss:", epoch_eval_loss_history)

# 繪製每個 epoch 的 Loss 圖表
epochs = list(range(1, len(epoch_loss_history)+1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, epoch_loss_history, label="Training Loss", color="blue", marker="o")
if epoch_eval_loss_history:
    plt.plot(epochs, epoch_eval_loss_history, label="Validation Loss", color="red", marker="x")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Per Epoch")
plt.legend()
plt.grid()
plt.savefig("_model_loss_curve_epoch_5_1k.png")
print("Loss 圖表已儲存為 model_loss_curve_epoch_5_1k.png")



# # 手動儲存微調後的 LoRA 模型
# trainer.model.to("cpu").save_pretrained(finetuned_model)
# tokenizer.save_pretrained(finetuned_model)
# print(f"微調完成，模型已儲存至: {finetuned_model}")


# dataset = load_dataset("lilacai/glaive-function-calling-v2-sharegpt", split="train[:80%]")

# chat_template = \
#     "{{ bos_token }}"\
#     "{% if messages[0]['from'] == 'system' %}"\
#         "{{'user\n' + messages[0]['value'] | trim + ' ' + messages[1]['value'] | trim + '\n'}}"\
#         "{% set messages = messages[2:] %}"\
#     "{% endif %}"\
#     "{% for message in messages %}"\
#         "{% if message['from'] == 'human' %}"\
#             "{{'user\n' + message['value'] | trim + '\n'}}"\
#         "{% elif message['from'] == 'gpt' %}"\
#             "{{'model\n' + message['value'] | trim + '\n' }}"\
#         "{% endif %}"\
#     "{% endfor %}"\
#     "{% if add_generation_prompt %}"\
#         "{{ 'model\n' }}"\
#     "{% endif %}"

# tokenizer.chat_template = chat_template

# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = [tokenizer.apply_chat_template(convo, tokenize = False,
#                       add_generation_prompt = False) for convo in convos]
#     return { "text" : texts, }

# dataset = dataset.map(formatting_prompts_func, batched = True,)

# df_train = pd.DataFrame(dataset)
# df_train["text"] = df_train["text"].apply(
#     lambda x: x.replace("

