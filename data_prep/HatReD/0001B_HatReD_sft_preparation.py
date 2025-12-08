### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import ast
import base64
import json

import pandas as pd

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------Load dataset and prompt---------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
df = pd.read_csv("fhm_train_reasonings.csv")
prompt = json.load(open("/teamspace/uploads/HatReD_reasonings.json"))["user_prompt"]

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------Prepare dataset for SFT------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
df.reasonings = df.reasonings.apply(lambda x: ast.literal_eval(x)[0])
df.img = df.img.str.removeprefix("/teamspace/studios/0001-hatred-preparation/")
with open("HatReD_train.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        with open(row["img"], "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        reasoning = "The meme " + row["reasonings"]
        example = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]},
                {"role": "assistant", "content": reasoning}
            ]
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
