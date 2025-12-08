### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import os
import random

import pandas as pd

from itertools import cycle
from slugify import slugify

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------Create a directory to store prepared dataframes---------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
os.makedirs(
    name="sft_prep",
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Load prompt templates------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
templates = json.load(open("/teamspace/uploads/persona_prompts.json", "r", encoding="utf-8"))
user_prompt = json.load(open("/teamspace/uploads/EXIST_sexist.json", "r", encoding="utf-8"))["user_prompt"]

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------------Prepare datasets for SFT------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
EXIST_training_es = pd.read_csv(
    "EXIST_training_es.csv",
    encoding="utf-8"
)
EXIST_training_es["gender_annotator"] = EXIST_training_es["gender_annotator"].map({"M": "a man", "F": "a woman"})
EXIST_training_es["label_task1"] = EXIST_training_es["label_task1"].map({"NO": "No", "YES": "Yes"})
group = EXIST_training_es.groupby(["gender_annotator", "tweet"])["label_task1"]
unanimous_annotations = group.first()[group.nunique().eq(1)]
for persona, subset in unanimous_annotations.groupby(level="gender_annotator"):
    series = subset.reset_index(level="gender_annotator", drop=True)
    template_cycle = cycle(templates.values())
    lines = []
    for tweet, label in series.items():
        system_prompt = next(template_cycle).format(persona=persona)
        lines.append(json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(content=str(tweet))},
                {"role": "assistant", "content": str(label)}
            ]
        }, ensure_ascii=False))
    random.Random(42).shuffle(lines)
    out_path = f"sft_prep/train_EXIST_{slugify(persona, separator='_')}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")