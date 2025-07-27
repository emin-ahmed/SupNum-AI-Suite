import yaml, mlflow, boto3, os, json
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from rouge import Rouge

def download_from_s3(bucket, key, local_path):
    boto3.client("s3").download_file(bucket, key, local_path)

def generate_answer(model, tokenizer, question_ar, question_fr, max_tokens):
    input_text = f"سؤال: {question_ar}  Question_FR: {question_fr}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    # Download test set
    os.makedirs("data", exist_ok=True)
    local_test = "data/test.jsonl"
    download_from_s3(params["bucket"], params["test_key"], local_test)
    with open(local_test) as f:
        test_data = [json.loads(line) for line in f]

    # Load model from MLflow
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    model_uri = f"models:/{params['mlflow_model_name']}/Production"
    model_path = mlflow.pyfunc.download_artifacts(model_uri)

    model = T5ForConditionalGeneration.from_pretrained(params["base_model"])
    model = PeftModel.from_pretrained(model, os.path.join(model_path, "adapter"))
    tokenizer = AutoTokenizer.from_pretrained(params["base_model"])
    model.eval()

    predictions, references = [], []
    for sample in test_data:
        pred = generate_answer(model, tokenizer, sample["question_ar"], sample["question_fr"], params["max_new_tokens"])
        predictions.append(pred)
        references.append(f"{sample['answer_ar']} {sample['answer_fr']}")

    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    metrics = {"rouge_l_f": rouge_scores["rouge-l"]["f"]}

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False)

    print("Evaluation metrics:", metrics)
