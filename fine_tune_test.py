from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import json
from datasets import load_dataset
import matplotlib.pyplot as plt

# 載入模型和 Tokenizer
model_id = 'MediaTek-Research/Llama-Breeze2-3B-Instruct-v0_1'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

# 設定 LoRA 配置
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 針對 Transformer 模型中的特定模組
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"  # 自回歸語言模型
)

# 將模型轉換為 LoRA 模式
model = get_peft_model(model, lora_config)

# 載入資料集
training_dataset_path = "processed_dataset.jsonl"
train_dataset = load_dataset("json", data_files=training_dataset_path)
print(train_dataset)
evaluation_dataset_path = "evaluation_dataset.jsonl"
eval_dataset = load_dataset("json", data_files=evaluation_dataset_path)

# 計算資料集中最長的 token 數
def compute_max_length(dataset, tokenizer):
    max_length = 0
    for example in train_dataset["train"]:
        # print(example) 
        input_length = len(tokenizer(example["instruction"])["input_ids"])
        target_length = len(tokenizer(example["response"])["input_ids"])
        max_length = max(max_length, input_length, target_length)
    return max_length

max_length = compute_max_length(train_dataset, tokenizer)
print(f"資料集中最長的 token 數為: {max_length}")

# 設定 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 將 pad_token 設為 eos_token

# 資料處理函數
def preprocess_function(examples):
    inputs = examples["instruction"]
    targets = examples["response"]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# 預處理資料集
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./lora_fine_tuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    report_to="none"
)

# 自訂 Callback 來紀錄 loss
class LossRecorderCallback:
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

loss_recorder = LossRecorderCallback()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 移除不支援的參數
        if "inputs_embeds" in inputs:
            del inputs["inputs_embeds"]
        return super().compute_loss(model, inputs, return_outputs=return_outputs,  **kwargs)

# 使用自訂的 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)


# # 初始化 Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset["train"],
#     eval_dataset=tokenized_eval_dataset,
#     tokenizer=tokenizer
# )

# 開始微調
trainer.train()

# 儲存微調後的模型
model.save_pretrained("./lora_fine_tuned_model")
tokenizer.save_pretrained("./lora_fine_tuned_model")


# 繪製 Loss 圖表
plt.plot(loss_recorder.losses, label="Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.savefig("training_loss.png")  # 儲存圖表為檔案
plt.show()