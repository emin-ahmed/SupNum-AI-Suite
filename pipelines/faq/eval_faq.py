import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Chemins ===
model_path = "models/faq_model_flan_base"  # Le modèle fine-tuné
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Charger le modèle et tokenizer fine-tunés ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
model.eval()

# === Entrée utilisateur ===
question = "Quel est le volume horaire total de la formation ?"

# === Préparation entrée
inputs = tokenizer(
    question,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=256
).to(device)

# === Génération réponse
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )

# === Affichage réponse
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("✅ Réponse générée :\n", response)

