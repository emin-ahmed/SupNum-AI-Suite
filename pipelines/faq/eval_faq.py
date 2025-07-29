# pipelines/faq/eval_faq.py
import yaml
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Charger la config
with open("params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_path = cfg["train"]["output_dir"]
model_name = cfg["train"]["model_name"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

question = "Quels sont les modules du semestre 5 de la licence ?"
input_ids = tokenizer(question, return_tensors="pt").input_ids

with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)

with open("eval_output.txt", "w") as f:
    f.write(f"Question: {question}\nRéponse: {response}")

print("✅ Réponse générée :\n", response)

