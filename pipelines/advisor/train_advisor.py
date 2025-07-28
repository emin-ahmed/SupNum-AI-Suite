import os
import boto3
import yaml
import mlflow
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from rouge import Rouge
from transformers import logging as hf_logging

hf_logging.set_verbosity_info()

# --- GPU CHECK ---
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_from_s3(bucket, key, local_path):
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    s3 = session.client("s3")
    s3.download_file(bucket, key, local_path)

def preprocess_function(examples, tokenizer):
    inputs = [f"سؤال: {q_ar}  Question_FR: {q_fr}" 
              for q_ar, q_fr in zip(examples["question_ar"], examples["question_fr"])]
    targets = [f"جواب: {a_ar}  Réponse: {a_fr}" 
               for a_ar, a_fr in zip(examples["answer_ar"], examples["answer_fr"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    rouge = Rouge()
    decoded_preds = [p.strip() for p in preds]
    decoded_labels = [l.strip() for l in labels]
    rouge_scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    return {"rouge_l_f": rouge_scores["rouge-l"]["f"]}

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["advisor"]

    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    mlflow.set_experiment(params["mlflow_experiment"])

    os.makedirs("data", exist_ok=True)
    local_train = "data/train.jsonl"
    download_from_s3(params["bucket"], params["train_key"], local_train)

    dataset = load_dataset("json", data_files={"train": local_train})
    tokenizer = AutoTokenizer.from_pretrained(params["base_model"])
    model = T5ForConditionalGeneration.from_pretrained(params["base_model"])

    # --- Move model to GPU explicitly ---
    model.to(device)

    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        target_modules=["q", "v"],
        lora_dropout=params["lora_dropout"],
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"],
        learning_rate=params["lr"],
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        save_total_limit=0,
        report_to=["mlflow"],
        fp16=torch.cuda.is_available()  # mixed precision if GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        compute_metrics=compute_metrics
    )

    with mlflow.start_run() as run:
        trainer.train()

        mlflow.log_params({
            "epochs": params["epochs"],
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "lora_r": params["lora_r"],
            "lora_alpha": params["lora_alpha"]
        })

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False)

        final_metrics = trainer.state.log_history
        for record in final_metrics:
            if "loss" in record:
                mlflow.log_metric("loss", record["loss"], step=record.get("step", 0))
            if "eval_loss" in record:
                mlflow.log_metric("eval_loss", record["eval_loss"], step=record.get("step", 0))

        adapter_dir = "/tmp/adapter"
        model.save_pretrained(adapter_dir)
        mlflow.pyfunc.log_model(
            artifact_path="advisor_model",
            python_model=None,
            artifacts={"adapter": adapter_dir}
        )

    print("Training completed and model logged to MLflow.")
