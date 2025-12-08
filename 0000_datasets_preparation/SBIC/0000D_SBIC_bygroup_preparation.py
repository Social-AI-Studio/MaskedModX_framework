### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import os

import pandas as pd

from slugify import slugify

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------Create a directory to store prepared dataframes---------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
os.makedirs(
    name="bygroup",
    exist_ok=True
)

### --------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------------Load dataset----------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
df = pd.read_csv(
    "SBIC.v2/SBIC.v2.tst.csv",
    encoding="utf-8"
)

### --------------------------------------------------------------------------------------------------------- ###
### ---------Define function to help prepare subsets where zero intra-group dissent w.r.t. question 1a------- ###
### --------------------------------------------------------------------------------------------------------- ###
def construct_bygroup_subsets(
        personas: dict[str, str],
        demographic_cols: list[str]
) -> None:
    df["offensiveYN"] = df["offensiveYN"].replace({1.0: "Yes", 0.5: "Maybe", 0.0: "No"}).fillna("Incomprehensible")
    group = df.groupby(demographic_cols + ["post"])["offensiveYN"]
    unanimous_annotations = group.first()[group.nunique().eq(1)]
    for key in personas:
        vals = key.split("_") if "_" in key else [key]
        output_df = unanimous_annotations.xs(vals[0] if len(demographic_cols) == 1 else tuple(vals)).rename("offensiveYN_unanimous").reset_index()
        output_df.to_csv(f"bygroup/{slugify(key, separator='_')}.csv", index=False, encoding="utf-8")

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------Apply construct_bygroup_subsets function----------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
construct_bygroup_subsets(
    json.load(open("/teamspace/uploads/SBIC_gender_identity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorGender"]
)
construct_bygroup_subsets(
    json.load(open("/teamspace/uploads/SBIC_race_ethnicity_personas_map.json", "r", encoding="utf-8")),
    ["annotatorRace"]
)
construct_bygroup_subsets(
    json.load(open("/teamspace/uploads/SBIC_political_leaning_personas_map.json", "r", encoding="utf-8")),
    ["annotatorPolitics"]
)
