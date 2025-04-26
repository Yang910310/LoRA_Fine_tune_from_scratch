import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 定義模型名稱
original_model_name = "meta-llama/Llama-3.2-3B-Instruct"
finetuned_model_path = r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\fine_tuned_model_english_10\checkpoint-120"

# 定義 prompt
prompt = "桌球發球有哪些規則?"

# 加載原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.bfloat16,
).to("cuda")
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_tokenizer.pad_token = original_tokenizer.eos_token

# 合併微調後的增量模型
finetuned_model = PeftModel.from_pretrained(original_model, finetuned_model_path)
finetuned_model = finetuned_model.merge_and_unload()  # 合併 LoRA 權重
finetuned_model.to("cuda")
finetuned_tokenizer = original_tokenizer  # 使用相同的 tokenizer

# 定義生成回應的函數
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        max_length=256,
        num_return_sequences=1,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 生成原始模型的回應
original_response = generate_response(original_model, original_tokenizer, prompt)
print("原始模型回應:", original_response)

# 生成微調後模型的回應
finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
print("微調後模型回應:", finetuned_response)
