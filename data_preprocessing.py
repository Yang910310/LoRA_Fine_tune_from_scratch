import json

# 載入原始 JSON 檔案
input_file = "ittf_qa_dataset_final_100.json"
output_file = "processed_dataset.jsonl"

def preprocess_data(input_file, output_file):
    # 開啟並讀取 JSON 檔案
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 將資料轉換為 JSONL 格式
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            # 選取需要的欄位
            processed_item = {
                "instruction": item["instruction"],
                "response": item["response"]
            }
            # 寫入 JSONL 格式，每行一個 JSON 物件
            f.write(json.dumps(processed_item, ensure_ascii=False) + "\n")

# 執行資料預處理
preprocess_data(input_file, output_file)

print(f"資料已成功預處理並儲存至 {output_file}")