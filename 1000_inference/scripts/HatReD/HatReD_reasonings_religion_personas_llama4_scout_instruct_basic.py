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
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio = Studio("1001_serverless_vlm_inference")
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.start(Machine.DATA_PREP)
time.sleep(100)
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.run("python -m pip install --upgrade pip")
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.run("python -m pip uninstall -y tensorboard")
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_start = perf_counter()
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.run("python 1001_serverless_vlm_inference.py \
    --inputfile /teamspace/studios/0001-hatred-preparation/fhm_test_reasonings.csv  \
    --outputfile HatReD_reasonings_religion_personas_llama4_scout_instruct_basic.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/HatReD_reasonings.json \
    --simulatedpersonas /teamspace/uploads/HatReD_religion_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --imgcol img \
    --model llama4-scout-instruct-basic")
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_stop = perf_counter()
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_studio.stop()
HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_time = HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_stop - HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_time": str(HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("HatReD_reasonings_religion_personas_llama4_scout_instruct_basic_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
