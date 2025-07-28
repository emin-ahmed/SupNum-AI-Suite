import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# === Chemins ===
model_name = "google/flan-t5-base"
dataset_path = "data/processed/faq_chatml1.jsonl"
save_path = "models/faq_model_flan_base"

# === Charger le dataset JSONL format ChatML
with open(dataset_path, "r", encoding="utf-8") as f:
    data = []
    for line in f:
        entry = json.loads(line)
        messages = entry["messages"]
        question = next((m["content"] for m in messages if m["role"] == "user"), None)
        answer = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if question and answer:
            data.append({"instruction": question, "output": answer})

# === Formatage en Dataset HuggingFace
dataset = Dataset.from_list(data)

# === Tokenizer + Modèle
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# === Appliquer LoRA (Low-Rank Adaptation)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# === Prétraitement : tokenisation
def preprocess(example):
    inputs = tokenizer(
        example["instruction"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    targets = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# === Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=3e-4,
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
    no_cuda=False  # Utilise GPU si dispo
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# === Entraînement
trainer.train()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Entraînement terminé et modèle sauvegardé dans :", save_path)

