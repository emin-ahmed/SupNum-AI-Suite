import yaml
import os
import json

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_faq(input_path, output_path):
    # Lecture ligne par ligne (format JSONL)
    processed_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "question" in item and "answer" in item and item["question"] and item["answer"]:
                    processed_data.append(item)
            except json.JSONDecodeError:
                continue  # Ignore les lignes invalides

    # Sauvegarde au format JSONL aussi
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    params = load_params()
    input_file = params["preprocess"]["input"]
    output_file = params["preprocess"]["output"]
    
    print(f"Preparing data from: {input_file} → {output_file}")
    prepare_faq(input_file, output_file)

