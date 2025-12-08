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
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio = Studio("1001_on_demand_vlm_inference")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.start(Machine.CPU)
time.sleep(100)
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.run("python -m pip install --upgrade pip")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.run("python -m pip uninstall -y tensorboard")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_start = perf_counter()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.run("python 1001_on_demand_vlm_inference.py \
    --inputfile /teamspace/studios/0001-hatred-preparation/fhm_test_reasonings.csv  \
    --outputfile HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/HatReD_reasonings.json \
    --simulatedpersonas /teamspace/uploads/HatReD_sexual_orientation_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --imgcol img \
    --model accounts/fireworks/models/qwen2p5-vl-72b-instruct \
    --id vcvv5blw")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_stop = perf_counter()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_studio.stop()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_time = HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_stop - HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_time": str(HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_72b_instruct_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
