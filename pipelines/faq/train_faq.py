import os
import yaml


from transformers import TrainerCallback

class MlflowLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)




# === Étape 1 : Charger la config params.yaml ===
cfg_path = "params.yaml"
cfg = yaml.safe_load(open(cfg_path))

# === Étape 2 : Configurer les credentials AWS ===
aws = cfg.get("aws", {})
os.environ["AWS_ACCESS_KEY_ID"]     = aws["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = aws["aws_secret_access_key"]
os.environ["AWS_DEFAULT_REGION"]    = aws.get("region_name", "us-east-1")
if aws.get("aws_session_token"):
    os.environ["AWS_SESSION_TOKEN"] = aws["aws_session_token"]

# === Étape 3 : Imports après credentials ===
import json
import torch
import mlflow
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from mlflow.models import infer_signature

# === Étape 4 : Préparer les données ===
data_path = cfg["preprocess"]["output"]

with open(data_path, "r", encoding="utf-8") as f:
    data = []
    for line in f:
        d = json.loads(line)
        q = d.get("question", None)
        a = d.get("answer", None)
        if q and a:
            data.append({"instruction": q, "output": a})


dataset = Dataset.from_list(data)

# === Étape 5 : Tokenizer et modèle ===
model_name = cfg["train"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Appliquer LoRA ===
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=cfg["train"].get("lora_r", 8),
    lora_alpha=cfg["train"].get("lora_alpha", 16),
    lora_dropout=cfg["train"].get("lora_dropout", 0.05),
    bias="none"
)
model = get_peft_model(model, peft_config)

# === Prétraitement ===
def preprocess(example):
    inputs = tokenizer(
        example["instruction"],
        truncation=True, padding="max_length", max_length=256
    )
    targets = tokenizer(
        example["output"],
        truncation=True, padding="max_length", max_length=128
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# === Étape 6 : Config MLflow ===
mlflow_uri = cfg["mlflow"]["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("FAQ_SupNum_FlanT5")

# === Étape 7 : Entraînement + Tracking MLflow ===
with mlflow.start_run(run_name="FAQ Fine-tuning"):
    # Log hyperparams
    mlflow.log_params({
        "model_name": model_name,
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "lr": float(cfg["train"]["learning_rate"])
    })

    output_dir = cfg["train"]["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["train"]["batch_size"],
        num_train_epochs=cfg["train"]["epochs"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",  # évite wandb
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        no_cuda=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[MlflowLoggingCallback()]  
    )

    trainer.train()

    # === Log artefacts ===
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    mlflow.log_artifacts(output_dir)

    # === Log final loss ===
    final_metrics = trainer.state.log_history[-1]
    if "loss" in final_metrics:
        mlflow.log_metric("final_loss", final_metrics["loss"])

    print(f"✅ Entraînement terminé. Modèle sauvegardé dans {output_dir}")

