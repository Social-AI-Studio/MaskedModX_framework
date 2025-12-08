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
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio = Studio("1000_serverless_inference")
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.start(Machine.DATA_PREP)
time.sleep(100)
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.run("python -m pip install --upgrade pip")
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.run("python -m pip uninstall -y tensorboard")
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_start = perf_counter()
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.run("python 1000_serverless_inference.py \
    --inputfile /teamspace/studios/0000-sbic-preparation/offensiveYN_aggregated_test.csv  \
    --outputfile SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/SBIC_offensiveYN.json \
    --simulatedpersonas /teamspace/uploads/SBIC_gender_identity_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --textcol post \
    --model kimi-k2-instruct-0905")
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_stop = perf_counter()
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_studio.stop()
SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_time = SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_stop - SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_time": str(SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("SBIC_offensiveYN_gender_identity_personas_kimi_k2_instruct_0905_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
