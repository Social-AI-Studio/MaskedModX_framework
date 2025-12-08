### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
import json
import time

from lightning_sdk import Machine, Studio
from time import perf_counter

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------------------------------Run studios---------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio = Studio("1000_on_demand_inference")
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.start(Machine.CPU)
time.sleep(100)
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.run("python -m pip install --upgrade pip")
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.run("python -m pip uninstall -y tensorboard")
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_start = perf_counter()
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.run("python 1000_on_demand_inference.py \
    --inputfile /teamspace/studios/0002-exist-preparation/EXIST_dev_es_unique.csv  \
    --outputfile EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/EXIST_sexist.json \
    --simulatedpersonas /teamspace/uploads/EXIST_gender_identity_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --textcol tweet \
    --model accounts/jiawangexperiments-b89b99/models/exist-es-qwen3-30b-a3b-instruct-2507 \
    --id u6dv0o1x \
    --baseline")
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_stop = perf_counter()
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_studio.stop()
EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_time = EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_stop - EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_time": str(EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("EXIST_sexist_es_gender_identity_personas_qwen3_30b_a3b_instruct_2507_finetuned_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
