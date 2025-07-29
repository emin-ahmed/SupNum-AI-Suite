import yaml
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# === Charger la configuration ===
with open("params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# === Récupérer les chemins depuis la config ou définir par défaut ===
base_model_name = cfg["train"].get("model_name", "google/flan-t5-base")
adapter_dir = cfg.get("eval", {}).get("adapter_path", "./temp_faq_model/artifacts/adapter")
config_path = cfg.get("eval", {}).get("config_path", "./temp_faq_model/artifacts/config")

# === Vérification de l'existence des fichiers nécessaires ===
assert os.path.exists(adapter_dir), f"❌ Adapter path not found: {adapter_dir}"
assert os.path.exists(config_path), f"❌ Config file not found: {config_path}"

# === Charger la config PEFT ===
with open(config_path, "r", encoding="utf-8") as f:
    peft_config = json.load(f)

# === Charger le tokenizer et le modèle de base ===
tokenizer = AutoTokenizer.from_pretrained(peft_config["base_model"])
base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config["base_model"])

# === Charger l’adapter LoRA ===
model = PeftModel.from_pretrained(base_model, adapter_dir)

# === Générer une réponse ===
question = "Quels sont les modules du semestre 5 de la licence ?"
inputs = tokenizer(question, return_tensors="pt")

model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)

# === Sauvegarde et affichage ===
with open("eval_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Question: {question}\nRéponse: {response}")

print("✅ Réponse générée :\n", response)
