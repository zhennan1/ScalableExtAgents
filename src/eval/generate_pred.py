import json
import argparse

def generate_dev_fullwiki_pred(
    sampled_hotpot_file: str,
    final_preds_file: str,
    output_file: str
):
    # 1. 读取 sampled_hotpot_questions.json
    with open(sampled_hotpot_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)  # 这是一个列表，每个元素是一个包含 _id、question 等字段的字典
    
    # 2. 逐行读取 final_preds_longbook_qa_eng.jsonl，存储 (id -> prediction)
    id_to_prediction = {}
    with open(final_preds_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line.strip())
            pred_id = line_data["id"]
            prediction = line_data["prediction"]
            id_to_prediction[pred_id] = prediction
    
    # 3. 构造输出格式
    final_dict = {"answer": {}}
    for i, q in enumerate(questions):
        question_id = q["_id"]             # "_id" 是问题在原始 JSON 的唯一标识符
        predicted_ans = id_to_prediction.get(i, "")  # 从 id_to_prediction 中取出预测答案
        final_dict["answer"][question_id] = predicted_ans
    
    # 4. 写入输出文件 dev_fullwiki_pred.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description="Generate dev_fullwiki_pred.json")
    parser.add_argument("--final_preds_file", type=str, required=True, 
                        help="Path to the final_preds_longbook_qa_eng.jsonl file")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to the output JSON file (e.g. dev_fullwiki_pred.json)")
    args = parser.parse_args()
    
    sampled_hotpot_file = "./data/sampled_hotpot_questions.json"  # 如有需要也可改为命令行参数

    generate_dev_fullwiki_pred(
        sampled_hotpot_file=sampled_hotpot_file, 
        final_preds_file=args.final_preds_file, 
        output_file=args.output_file
    )
