import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 定義模型名稱
# original_model_name = "meta-llama/Llama-3.2-3B-Instruct"
original_model_name = "lianghsun/Llama-3.2-Taiwan-3B-Instruct"
finetuned_model_path = r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\fine_tuned_model_chinese_5_1k\checkpoint-625"

template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

# # 定義 system prompt
# system_prompt = "You are a table tennis rules expert. Provide concise and accurate answers to questions about table tennis rules."\
# # "You must respond in Traditional Chinese."

# # Define the prompt for inference
# prompt = template.format(
#     instruction=f"{system_prompt}\n\nHow should the table tennis ball be tossed when serving?",
#     response="",
# )

# 定義 system prompt
system_prompt = "你是一個桌球規則專家。根據桌球規則，提供簡潔且準確的回答。"\

# Define the prompt for inference
prompt = template.format(
    instruction=f"{system_prompt}\n\n桌球比賽中拋球高度不足16公分會怎麼判？",
    response="",
)

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


# 加載原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.bfloat16,
).to("cuda")
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_tokenizer.pad_token = original_tokenizer.eos_token

# 生成原始模型的回應
original_response = generate_response(original_model, original_tokenizer, prompt)
print("原始模型回應:", original_response)


# 合併微調後的增量模型
finetuned_model = PeftModel.from_pretrained(original_model, finetuned_model_path)
finetuned_model = finetuned_model.merge_and_unload()  # 合併 LoRA 權重
finetuned_model.to("cuda")
finetuned_tokenizer = original_tokenizer  # 使用相同的 tokenizer

# 生成微調後模型的回應
finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, prompt)
print("微調後模型回應:", finetuned_response)




