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
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio = Studio("1001_on_demand_vlm_inference")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.start(Machine.CPU)
time.sleep(100)
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.run("python -m pip install --upgrade pip")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.run("python -m pip uninstall -y tensorboard")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.run("PIP_USE_PEP517=1 pip install -r /teamspace/uploads/requirements_serverless.txt")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_start = perf_counter()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.run("python 1001_on_demand_vlm_inference.py \
    --inputfile /teamspace/studios/0001-hatred-preparation/fhm_test_reasonings.csv  \
    --outputfile HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned.csv \
    --systemprompts /teamspace/uploads/persona_prompts.json \
    --userprompts /teamspace/uploads/HatReD_reasonings.json \
    --simulatedpersonas /teamspace/uploads/HatReD_sexual_orientation_personas.json \
    --inputcred /teamspace/uploads/FIREWORKS_CREDENTIALS.env \
    --imgcol img \
    --model accounts/jiawangexperiments-b89b99/models/hatred-qwen2p5-vl-32b-instruct \
    --id tr8885uq")
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_stop = perf_counter()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_studio.stop()
HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_time = HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_stop - HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_start

### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Record time taken------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
time_taken_dict = {
    "HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_time": str(HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_time)
}
time_taken_json = json.dumps(time_taken_dict)
with open("HatReD_reasonings_sexual_orientation_personas_qwen2p5_vl_32b_instruct_finetuned_time_taken.json", "w", encoding="utf-8") as outfile:
    outfile.write(time_taken_json)
