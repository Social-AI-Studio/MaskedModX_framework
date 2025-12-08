### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json

import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------Convert EXIST dev dataset to csv format--------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
with open("EXIST2024_dev.json", "r", encoding="utf-8") as f:
    dev_data = json.load(f)
dev_record_list = []
for _, details in dev_data.items():
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
        dev_record_list.append(record)
dev = pd.DataFrame(dev_record_list)
dev.to_csv(
    "EXIST_dev.csv",
    index=False,
    encoding="utf-8"
)
dev[dev["split"] == "DEV_ES"].to_csv(
    "EXIST_dev_es.csv",
    index=False,
    encoding="utf-8"
)
dev[dev["split"] == "DEV_ES"]["tweet"].drop_duplicates().to_csv(
    "EXIST_dev_es_unique.csv",
    index=False,
    encoding="utf-8"
)