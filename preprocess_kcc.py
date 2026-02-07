import pandas as pd
import json

df = pd.read_csv("data/kcc_raw.csv")

df = df.dropna()
df["question"] = df["question"].str.strip()
df["answer"] = df["answer"].str.strip()

df.to_csv("data/clean_kcc.csv", index=False)

qa_pairs = []
for _, row in df.iterrows():
    qa_pairs.append({
        "question": row["question"],
        "answer": row["answer"]
    })

with open("data/kcc_qa_pairs.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=2)

print("✅ Preprocessing completed successfully") 