from src.config import *
import random

lines1 = []
with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        en = parts[0].strip()
        cn = " ".join(list(parts[1].strip()))
        lines1.append(f"{en}\t{cn}")

random.shuffle(lines1)

split_idx = int(len(lines1) * 0.8)
train_lines = lines1[:split_idx]
test_lines = lines1[split_idx:]

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))
with open(DEV_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(test_lines))

print(f"Restored files in {OUTPUT_DIR}")

