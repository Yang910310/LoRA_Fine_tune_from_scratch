import jieba
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from moverscore_v2 import get_idf_dict, word_mover_score


#目前此檔案還無法正常運作





# 讀取資料集
data_file = r"D:\research_information\LoRA_fine_tuning_test\LoRA_Fine_tune_from_scratch\data\processed_training_dataset.jsonl"
references = []
hypotheses = []

with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        references.append(" ".join(jieba.cut(item["reference"])))
        hypotheses.append(" ".join(jieba.cut(item["hypothesis"])))

# 計算 IDF 字典
idf_hyp = get_idf_dict(hypotheses)
idf_ref = get_idf_dict(references)

# 計算 MoverScore
scores = word_mover_score(
    references, hypotheses,
    idf_ref, idf_hyp,
    stop_words=[], 
    model_type='bert-base-chinese',
    n_gram=1,
    remove_subwords=True,
    batch_size=8
)

# 顯示部分結果
for i in range(min(5, len(scores))):
    print(f"[樣本 {i+1}]")
    print("參考句：", references[i])
    print("生成句：", hypotheses[i])
    print("MoverScore：", scores[i])
    print("-" * 30)

# 可以將分數儲存為檔案（選用）
with open("/mnt/data/mover_scores_output.json", "w", encoding="utf-8") as f_out:
    for i, score in enumerate(scores):
        json.dump({
            "reference": references[i],
            "hypothesis": hypotheses[i],
            "mover_score": score
        }, f_out, ensure_ascii=False)
        f_out.write("\n")
