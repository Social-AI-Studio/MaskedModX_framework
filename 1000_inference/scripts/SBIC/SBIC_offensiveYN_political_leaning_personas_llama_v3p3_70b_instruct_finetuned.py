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
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio = Studio("1000_on_demand_inference")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.start(Machine.CPU)
time.sleep(100)
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.run("python -m pip install --upgrade pip")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.run("python -m pip uninstall -y tensorboard")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_start = perf_counter()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.run("python 1000_on_demand_inference.py \
    --inputfile /teamspace/studios/0000-sbic-preparation/offensiveYN_aggregated_test.csv  \
    --outputfile SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/SBIC_offensiveYN.json \
    --simulatedpersonas /teamspace/uploads/SBIC_political_leaning_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --textcol post \
    --model accounts/jiawangexperiments-b89b99/models/sbic-llama-v3p3-70b-instruct \
    --id ganq1mhs")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_stop = perf_counter()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_studio.stop()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_time = SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_stop - SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_time": str(SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_finetuned_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
