### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import random

import numpy as np
import pandas as pd

from itertools import cycle

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Load prompt templates------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
templates = list(json.load(open("/teamspace/uploads/persona_prompts.json", "r", encoding="utf-8")).values())
user_prompt = json.load(open("/teamspace/uploads/SBIC_offensiveYN.json", "r", encoding="utf-8"))["user_prompt"]

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Prepare dataset for SFT------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
SBIC_training = pd.read_csv(
    "SBIC.v2/SBIC.v2.trn.csv",
    encoding="utf-8"
)
SBIC_training = SBIC_training.replace(["na", ""], np.nan).dropna(subset=["annotatorGender", "annotatorRace", "annotatorPolitics", "annotatorAge"]).reset_index(drop=True)
SBIC_training = SBIC_training.drop_duplicates(subset=["WorkerId", "HITId", "annotatorGender", "annotatorRace", "annotatorPolitics", "annotatorAge"], keep="first").reset_index(drop=True)
SBIC_training["offensiveYN"] = SBIC_training["offensiveYN"].replace({1.0: "Yes", 0.5: "Maybe", 0.0: "No"}).fillna("Incomprehensible")
lines = []
SBIC_training["annotatorGender"] = SBIC_training["annotatorGender"].replace({"nonBinary": "non-binary individual",
                                                                             "transman": "transgender man"})
SBIC_training["annotatorRace"] = SBIC_training["annotatorRace"].replace({"asian": "Asian/Asian American",
                                                                         "black": "Black/African American",
                                                                         "hisp": "Hispanic/Latinx",
                                                                         "native": "Native Hawaiian/Pacific Islander or Native American/First Nations",
                                                                         "other": "mixed-race or of a race/ethnicity not otherwise listed",
                                                                         "white": "White/Caucasian"})
SBIC_training["annotatorPolitics"] = SBIC_training["annotatorPolitics"].replace({"cons": "a conservative political leaning",
                                                                                 "liberal": "a liberal/progressive political leaning",
                                                                                 "libert": "a libertarian political leaning",
                                                                                 "mod-cons": "a moderate conservative political leaning",
                                                                                 "mod-liberal": "a moderate liberal political leaning",
                                                                                 "other": "an independent political leaning or a non-mainstream political leaning"})
template_cycle = cycle(templates)
for _, row in SBIC_training.iterrows():
    persona = f"a {row['annotatorGender']} in the United States who identifies as {row['annotatorRace']}, is aged {str(int(row['annotatorAge']))}, and has {row['annotatorPolitics']}"
    lines.append(json.dumps({
        "messages": [
            {"role": "system", "content": next(template_cycle).format(persona=persona)},
            {"role": "user", "content": user_prompt.format(content=row["post"])},
            {"role": "assistant", "content": row["offensiveYN"]}
        ]
    }, ensure_ascii=False))

random.Random(42).shuffle(lines)
with open("SBIC_training.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
