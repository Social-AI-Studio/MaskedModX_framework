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
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio = Studio("1000_on_demand_inference")
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.start(Machine.CPU)
time.sleep(100)
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.run("python -m pip install --upgrade pip")
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.run("python -m pip uninstall -y tensorboard")
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_start = perf_counter()
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.run("python 1000_on_demand_inference.py \
    --inputfile /teamspace/studios/0000-sbic-preparation/bygroup/mod_liberal.csv  \
    --outputfile SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/SBIC_offensiveYN.json \
    --simulatedpersonas /teamspace/uploads/mod_liberal_persona.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --textcol post \
    --model accounts/jiawangexperiments-b89b99/models/mod-liberal-qwen3-30b-a3b-instruct-2507 \
    --id btxazjqj")
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_stop = perf_counter()
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_studio.stop()
SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_time = SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_stop - SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_time": str(SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("SBIC_offensiveYN_mod_liberal_qwen3_30b_a3b_instruct_2507_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
