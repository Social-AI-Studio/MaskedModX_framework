### --------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------Import libraries-------------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from dotenv import load_dotenv
from fireworks import LLM
from itertools import chain, product
from parallel_pandas import ParallelPandas
from pathlib import Path
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import Mapping

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------------------------Initialize parallel-pandas-------------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
ParallelPandas.initialize(n_cpu=32)

### --------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------Instantiate the LLM client------------------------------------ ###
### --------------------------------------------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description="Generate content evaluations made by LLM personas/baselines")
parser.add_argument(
    "--inputfile",
    required=True,
    help="Absolute file path of .csv file containing hate speech data"
)
parser.add_argument(
    "--outputfile",
    required=True,
    help="File name of .csv file containing content evaluations made by LLM personas/baselines"
)
parser.add_argument(
    "--systemprompts",
    required=True,
    help="Absolute file path of JSON file containing system prompts and prompt templates used for inference"
)
parser.add_argument(
    "--userprompts",
    required=True,
    help="Absolute file path of JSON file containing user prompts and prompt templates used for inference"
)
parser.add_argument(
    "--simulatedpersonas",
    required=True,
    help="Absolute file path of JSON file containing demographic topic (e.g., race and ethnicity) as the key and a list of personas (e.g. a White/Caucasian individual in the United States) as the value"
)
parser.add_argument(
    "--inputcred",
    required=True,
    help="Absolute file path of .env file containing Fireworks API key"
)
parser.add_argument(
    "--textcol",
    required=True,
    help="Name of the column in the CSV containing the text to evaluate"
)
parser.add_argument(
    "--model",
    required=True,
    help="Model name as on Fireworks AI (e.g., llama-v3p3-70b-instruct)"
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="If set, run an additional baseline without persona/system prompt and append its outputs as a new column"
)
args = parser.parse_args()

### --------------------------------------------------------------------------------------------------------- ###
### ------------------------Set FIREWORKS_API_KEY environment variable for authentication-------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
load_dotenv(Path(args.inputcred))

### --------------------------------------------------------------------------------------------------------- ###
### --------------------------------------Instantiate a serverless LLM client-------------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
llm = LLM(
    model=args.model,
    deployment_type="serverless",
    api_key=os.getenv("FIREWORKS_API_KEY")
)

### --------------------------------------------------------------------------------------------------------- ###
### -----------------------Define a function to call the FIREWORKS AI API for inference---------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
@retry(
    wait=wait_random_exponential(min=1, max=100),
    stop=stop_after_attempt(max_attempt_number=50)
)
def inference(
        user_prompt: str,
        system_prompt: str | None = None
) -> tuple[str, str]:
    """
    This function calls the Fireworks AI API.
    This leverages code from https://docs.fireworks.ai/guides/querying-text-models

            Parameters:
                    user_prompt (str): User prompt passed to Fireworks AI API.
                    system_prompt (str | None): Optional system prompt passed to Fireworks AI API.

            Returns:
                    tuple containing
                            response_json (str): Response from calling the Fireworks AI API, in JSON string format.
                            response_text (str): Text content of the response.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = llm.chat.completions.create(
        messages=messages,
        max_tokens=512,
        temperature=0,
        top_k=1
    )
    response_json = response.model_dump_json()
    response_text = response.choices[0].message.content
    return response_json, response_text


### --------------------------------------------------------------------------------------------------------- ###
### ------------------------Define main function that applies the inference function------------------------- ###
### --------------------------------------------------------------------------------------------------------- ###
def main() -> None:
    data = pd.read_csv(
        filepath_or_buffer=args.inputfile,
        encoding="utf-8"
    )
    with open(args.systemprompts, mode="r", encoding="utf-8") as file:
        system_prompts: Mapping[str, str] = json.load(file)
    with open(args.userprompts, mode="r", encoding="utf-8") as file:
        user_prompts: Mapping[str, str] = json.load(file)
    with open(args.simulatedpersonas, mode="r", encoding="utf-8") as file:
        personas: Mapping[str, list[str]] = json.load(file)
    templates = ("persona_instruction1", "persona_instruction2", "persona_instruction3")
    for persona, prompt_template in product(chain.from_iterable(personas.values()), templates):
        p = re.sub(r"\W+", "_", str(persona), flags=re.ASCII).strip("_") + "_" + prompt_template
        system_prompt = system_prompts[prompt_template].format(persona=persona)
        data[[f"{p}_response_json", f"{p}_response_text"]] = data.p_apply(
            lambda row: inference(
                user_prompt=user_prompts["user_prompt"].format(content=row[args.textcol]),
                system_prompt=system_prompt,
            ),
            axis=1,
            result_type="expand"
        )
    if args.baseline:
        data[["baseline_response_json", "baseline_response_text"]] = data.p_apply(
            lambda row: inference(
                user_prompt=user_prompts["user_prompt"].format(content=row[args.textcol]),
            ),
            axis=1,
            result_type="expand"
        )
    data.to_csv(
        args.outputfile,
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
