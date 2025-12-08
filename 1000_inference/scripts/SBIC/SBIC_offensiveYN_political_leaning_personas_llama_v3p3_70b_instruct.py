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
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio = Studio("1000_serverless_inference")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.start(Machine.DATA_PREP)
time.sleep(100)
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.run("python -m pip install --upgrade pip")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.run("python -m pip uninstall -y tensorboard")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_start = perf_counter()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.run("python 1000_serverless_inference.py \
    --inputfile /teamspace/studios/0000-sbic-preparation/offensiveYN_aggregated_test.csv  \
    --outputfile SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/SBIC_offensiveYN.json \
    --simulatedpersonas /teamspace/uploads/SBIC_political_leaning_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --textcol post \
    --model llama-v3p3-70b-instruct")
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_stop = perf_counter()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_studio.stop()
SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_time = SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_stop - SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_time": str(SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("SBIC_offensiveYN_political_leaning_personas_llama_v3p3_70b_instruct_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
