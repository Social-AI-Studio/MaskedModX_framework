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
user_prompt = json.load(open("/teamspace/uploads/SBIC_offensiveYN.json", "r", encoding="utf-8"))["user_prompt"]

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------Define function to help prepare dataset for SFT--------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def construct_sft_dataset(
        df: pd.DataFrame,
        personas: dict[str, str],
        demographic_cols: list[str],
        split: str
) -> None:
    df["offensiveYN"] = df["offensiveYN"].replace({1.0: "Yes", 0.5: "Maybe", 0.0: "No"}).fillna("Incomprehensible")
    keys = demographic_cols + ["post"]
    group = df.groupby(keys)["offensiveYN"]
    unanimous_annotations = group.first()[group.nunique().eq(1)]
    levels = demographic_cols[0] if len(demographic_cols) == 1 else demographic_cols
    for demographic_vals, subset in unanimous_annotations.groupby(level=levels):
        if not isinstance(demographic_vals, tuple):
            demographic_vals = (demographic_vals,)
        persona_key = "_".join(val for val in demographic_vals)
        persona = personas.get(persona_key)
        if persona is None:
            continue 
        series = subset.reset_index(level=demographic_cols, drop=True)
        if series.empty:
            continue
        template_cycle = cycle(templates.values())

        lines = []
        for post, label in series.items():
            system_prompt = next(template_cycle).format(persona=persona)
            lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(content=str(post))},
                    {"role": "assistant", "content": str(label)}
                ]
            }, ensure_ascii=False))

        random.Random(42).shuffle(lines)

        out_path = f"sft_prep/{split}_{slugify(persona_key, separator='_')}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------Apply construct_sft_dataset function------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.trn.csv"),
    json.load(open("/teamspace/uploads/SBIC_race_ethnicity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorRace"],
    "train"
)
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.dev.csv"),
    json.load(open("/teamspace/uploads/SBIC_race_ethnicity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorRace"],
    "dev"
)
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.trn.csv"),
    json.load(open("/teamspace/uploads/SBIC_gender_identity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorGender"],
    "train"
)
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.dev.csv"),
    json.load(open("/teamspace/uploads/SBIC_gender_identity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorGender"],
    "dev"
)
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.trn.csv"),
    json.load(open("/teamspace/uploads/SBIC_political_leaning_personas_map.json", "r", encoding="utf-8")),
    ["annotatorPolitics"],
    "train"
)
construct_sft_dataset(
    pd.read_csv("SBIC.v2/SBIC.v2.dev.csv"),
    json.load(open("/teamspace/uploads/SBIC_political_leaning_personas_map.json", "r", encoding="utf-8")),
    ["annotatorPolitics"],
    "dev"
)