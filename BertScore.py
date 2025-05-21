from bert_score import score
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

original_model_name = "lianghsun/Llama-3.2-Taiwan-3B-Instruct"
finetuned_model_path = r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\models\fine_tuned_model_chinese_5_1k\checkpoint-625"
evaluation_dataset_path = r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\data\evaluation_dataset.jsonl"

# 讀取 evaluation_dataset.jsonl
with open(evaluation_dataset_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

# 定義生成回應的函數
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=100,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 擷取Response的文字
def extract_response(response):
    # 找到 "Response:" 的位置，並提取其後的文字
    response_start = response.find("Response:")
    if response_start != -1:
        return response[response_start + len("Response:"):].strip()
    return response.strip()  # 如果找不到 "Response:"，返回整個回應

# 加載原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.bfloat16,
).to("cuda")
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_tokenizer.pad_token = original_tokenizer.eos_token

# 初始化分數累加器
total_precision_orig, total_recall_orig, total_f1_orig = 0.0, 0.0, 0.0
total_precision_fine, total_recall_fine, total_f1_fine = 0.0, 0.0, 0.0
num_samples = 54


# 計算原始模型的分數
for data in dataset:
    instruction = data["instruction"]
    reference_response = data["response"]

    # 構建 prompt
    prompt = f"Instruction:\n{instruction}\n\nResponse:\n"

    # 生成原始模型的回應
    original_response = generate_response(original_model, original_tokenizer, prompt)
    original_response_text = extract_response(original_response) # 擷取原始模型回應的 response 部分

    # 計算分數
    P_orig, R_orig, F1_orig = score(
        [original_response_text], [reference_response],
        model_type="bert-base-chinese", lang="zh", verbose=False
    )

    # 累加分數
    total_precision_orig += P_orig[0].item()
    total_recall_orig += R_orig[0].item()
    total_f1_orig += F1_orig[0].item()


    # 輸出每筆資料的結果
    print(f"Instruction: {instruction}")
    print(f"Reference: {reference_response}")
    print(f"Original Model Response: {original_response_text}")
    print(f"Original Model Scores - Precision: {P_orig[0]:.4f}, Recall: {R_orig[0]:.4f}, F1: {F1_orig[0]:.4f}")
    print("-" * 50)


# 合併微調後的增量模型
finetuned_model = PeftModel.from_pretrained(original_model, finetuned_model_path)
finetuned_model = finetuned_model.merge_and_unload()  # 合併 LoRA 權重
finetuned_model.to("cuda")
finetuned_tokenizer = original_tokenizer  # 使用相同的 tokenizer

# 計算微調後模型的分數
for data in dataset:
    instruction = data["instruction"]
    reference_response = data["response"]

    # 構建 prompt
    prompt = f"Instruction:\n{instruction}\n\nResponse:\n"

    # 生成微調後模型的回應
    finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
    finetuned_response_text = extract_response(finetuned_response)

    # 計算分數
    P_fine, R_fine, F1_fine = score(
        [finetuned_response_text], [reference_response],
        model_type="bert-base-chinese", lang="zh", verbose=False
    )

    # 累加分數
    total_precision_fine += P_fine[0].item()
    total_recall_fine += R_fine[0].item()
    total_f1_fine += F1_fine[0].item()

    # 輸出每筆資料的結果
    print(f"Instruction: {instruction}")
    print(f"Reference: {reference_response}")
    print(f"Fine-tuned Model Response: {finetuned_response_text}")
    print(f"Fine-tuned Model Scores - Precision: {P_fine[0]:.4f}, Recall: {R_fine[0]:.4f}, F1: {F1_fine[0]:.4f}")
    print("-" * 50)


# 計算平均分數
avg_precision_orig = total_precision_orig / num_samples
avg_recall_orig = total_recall_orig / num_samples
avg_f1_orig = total_f1_orig / num_samples

avg_precision_fine = total_precision_fine / num_samples
avg_recall_fine = total_recall_fine / num_samples
avg_f1_fine = total_f1_fine / num_samples

# 輸出平均分數
print("\nAverage Scores for Original Model:")
print(f"Precision: {avg_precision_orig:.4f}, Recall: {avg_recall_orig:.4f}, F1: {avg_f1_orig:.4f}")

print("\nAverage Scores for Fine-tuned Model:")
print(f"Precision: {avg_precision_fine:.4f}, Recall: {avg_recall_fine:.4f}, F1: {avg_f1_fine:.4f}")
