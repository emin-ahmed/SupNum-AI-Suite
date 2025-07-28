import json
import os

# Définir les chemins
RAW_PATH = "data/intialze.jsnol"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "faq_chatml1.jsonl")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chargement des données JSONL
with open(RAW_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Conversion en format ChatML
chatml_data = []
for line in lines:
    item = json.loads(line)
    question = item["question"].strip()
    answer = item["answer"].strip()

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    chatml_data.append({"messages": messages})

# Écriture du fichier de sortie
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in chatml_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Données converties en format ChatML et enregistrées dans {OUTPUT_PATH}")
