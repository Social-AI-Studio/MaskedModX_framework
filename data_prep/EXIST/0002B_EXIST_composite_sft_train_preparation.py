### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import random

import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------Convert EXIST training dataset to csv format------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
with open("EXIST2024_training.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)
training_record_list = []
for _, details in training_data.items():
    for i in range(details["number_annotators"]):
        record = {
            "id_EXIST": details["id_EXIST"],
            "lang": details["lang"],
            "tweet": details["tweet"],
            "split": details["split"],
            "annotator": details["annotators"][i],
            "gender_annotator": details["gender_annotators"][i],
            "age_annotator": details["age_annotators"][i],
            "ethnicity_annotator": details["ethnicities_annotators"][i],
            "study_level_annotator": details["study_levels_annotators"][i],
            "country_annotator": details["countries_annotators"][i],
            "label_task1": details["labels_task1"][i],
            "label_task2": details["labels_task2"][i],
            "label_task3": details["labels_task3"][i]
        }
        training_record_list.append(record)
training = pd.DataFrame(training_record_list)
training.to_csv(
    "EXIST_training.csv",
    index=False,
    encoding="utf-8"
)
EXIST_training_es = training[training["split"] == "TRAIN_ES"].reset_index(drop=True)
EXIST_training_es.to_csv(
    "EXIST_training_es.csv",
    index=False,
    encoding="utf-8"
)

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Load prompt templates------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
templates = list(json.load(open("/teamspace/uploads/persona_prompts.json", "r", encoding="utf-8")).values())
user_prompt = json.load(open("/teamspace/uploads/EXIST_sexist.json", "r", encoding="utf-8"))["user_prompt"]

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Prepare dataset for SFT------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
lines = []
EXIST_training_es["gender_annotator"] = EXIST_training_es["gender_annotator"].map({"M": "man", "F": "woman"})
EXIST_training_es["ethnicity_annotator"] = EXIST_training_es["ethnicity_annotator"].replace({"other": "a person from an ethnic group not otherwise listed",
                                                                                             "Multiracial": "multiracial"})
EXIST_training_es["label_task1"] = EXIST_training_es["label_task1"].map({"NO": "No", "YES": "Yes"})
EXIST_training_es["age_annotator"] = EXIST_training_es["age_annotator"].replace({"46+": "46 or older"})
EXIST_training_es["study_level_annotator"] = EXIST_training_es["study_level_annotator"].str.replace('’', "'")
EXIST_training_es["study_level_annotator"] = EXIST_training_es["study_level_annotator"].replace({"Less than high school diploma": "less than a high school diploma",
                                                                                                 "Bachelor's degree": "a bachelor's degree",
                                                                                                 "High school degree or equivalent": "a high school degree or equivalent",
                                                                                                 "Master's degree": "a master's degree",
                                                                                                 "Doctorate": "a doctorate",
                                                                                                 "other": "a level of education not otherwise listed"})
for _, group in EXIST_training_es.groupby("tweet", sort=False):
    templates_list = templates * 2
    random.shuffle(templates_list)
    records = group.to_dict(orient="records")
    for row, template in zip(records, templates_list):
        persona = f"a {row['gender_annotator']} who identifies as {row['ethnicity_annotator']}, living in {row['country_annotator']}, aged {row['age_annotator']}, and has {row['study_level_annotator']}"
        lines.append(json.dumps({
            "messages":[
                {"role": "system", "content": template.format(persona=persona)},
                {"role": "user", "content": user_prompt.format(content=row["tweet"])},
                {"role": "assistant", "content": row["label_task1"]}
            ]
        }, ensure_ascii=False))

random.Random(42).shuffle(lines)
with open("EXIST_training_es.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
