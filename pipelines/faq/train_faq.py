import os
import yaml
import json
import shutil
import torch
import mlflow
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

# === Charger la config ===
cfg_path = "params.yaml"
cfg = yaml.safe_load(open(cfg_path))

aws = cfg.get("aws", {})
os.environ["AWS_ACCESS_KEY_ID"] = aws["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = aws["aws_secret_access_key"]
os.environ["AWS_DEFAULT_REGION"] = aws.get("region_name", "us-east-1")
if aws.get("aws_session_token"):
    os.environ["AWS_SESSION_TOKEN"] = aws["aws_session_token"]

# === Charger les données ===
data_path = cfg["preprocess"]["output"]
with open(data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if "question" in line and "answer" in line]
    dataset = Dataset.from_list([
        {"instruction": d["question"], "output": d["answer"]}
        for d in data
    ])

# === Modèle et Tokenizer ===
model_name = cfg["train"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Appliquer LoRA ===
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=cfg["train"]["lora_r"],
    lora_alpha=cfg["train"]["lora_alpha"],
    lora_dropout=cfg["train"]["lora_dropout"],
    bias="none"
)
model = get_peft_model(model, peft_config)

# === Tokenisation ===
def preprocess(example):
    inputs = tokenizer(example["instruction"], truncation=True, padding="max_length", max_length=256)
    targets = tokenizer(example["output"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# === Config Training ===
output_dir = cfg["train"]["output_dir"]
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=cfg["train"]["batch_size"],
    num_train_epochs=cfg["train"]["epochs"],
    learning_rate=float(cfg["train"]["learning_rate"]),
    save_strategy="epoch",
    logging_steps=1,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_first_step=True,
    no_cuda=not torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# === Enregistrement avec MLflow ===
mlflow.set_tracking_uri(cfg["mlflow"]["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("FAQ_SupNum_FlanT5")

class LoRAFAQModel(PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        from peft import PeftModel, PeftConfig

        config_path = context.artifacts["config"]
        adapter_path = context.artifacts["adapter"]

        with open(config_path, "r") as f:
            peft_config = json.load(f)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(peft_config["base_model"])
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer

    def predict(self, context, model_input):
        inputs = self.tokenizer(list(model_input), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

with mlflow.start_run() as run:
    trainer.train()

    mlflow.log_params({
        "model_name": model_name,
        "epochs": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "lr": float(cfg["train"]["learning_rate"]),
        "lora_r": cfg["train"]["lora_r"],
        "lora_alpha": cfg["train"]["lora_alpha"]
    })

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    mlflow.log_artifact("metrics/train_metrics.json")

    for record in trainer.state.log_history:
        if "loss" in record:
            mlflow.log_metric("loss", record["loss"], step=record.get("step", 0))

    # Sauvegarder adapter et config
    adapter_dir = "/tmp/faq_adapter"
    config_path = "/tmp/faq_config.json"
    model.save_pretrained(adapter_dir)

    config = {
        "base_model": model_name,
        "lora_r": cfg["train"]["lora_r"],
        "lora_alpha": cfg["train"]["lora_alpha"],
        "target_modules": ["q", "v"],
        "task_type": "SEQ_2_SEQ_LM"
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Pyfunc Model Saving
    model_path = "./temp_faq_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=LoRAFAQModel(),
        artifacts={"adapter": adapter_dir, "config": config_path},
        pip_requirements=["torch", "transformers", "peft", "accelerate"]
    )

    mlflow.log_artifacts(model_path, artifact_path="faq_model")

    # Enregistrement dans le Model Registry
    model_uri = f"runs:/{run.info.run_id}/faq_model"
    registered = mlflow.register_model(model_uri=model_uri, name="faq-model")
    print(f"✅ Modèle enregistré sous : {registered.name}, version : {registered.version}")
