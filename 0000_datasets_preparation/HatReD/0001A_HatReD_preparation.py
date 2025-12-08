### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------Download HatReD dataset---------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
pd.read_json(
    "https://raw.githubusercontent.com/Social-AI-Studio/HatReD/refs/heads/main/datasets/hatred/annotations/fhm_train_reasonings.jsonl",
    lines=True
).loc[lambda df: df["target"].map(len) == 1]\
    .assign(img=lambda df: "/teamspace/studios/0001-hatred-preparation/FHM/" + df.img)\
    .to_csv("fhm_train_reasonings.csv", index=False, encoding="utf-8")

pd.read_json(
    "https://raw.githubusercontent.com/Social-AI-Studio/HatReD/refs/heads/main/datasets/hatred/annotations/fhm_test_reasonings.jsonl",
    lines=True
).loc[lambda df: df["target"].map(len) == 1]\
    .assign(img=lambda df: "/teamspace/studios/0001-hatred-preparation/FHM/" + df.img)\
    .to_csv("fhm_test_reasonings.csv", index=False, encoding="utf-8")