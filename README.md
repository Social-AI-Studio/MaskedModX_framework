# MaskedModX
Code implementation for the paper "Evaluating The Sociodemographic Biases Of AI Models In Offensive Content Detection Using Simulated Personas"
## Datasets
* **English text:** SBIC 

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. [Social Bias Frames: Reasoning about Social and Power Implications of Language.](https://aclanthology.org/2020.acl-main.486) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5477–5490, Online. Association for Computational Linguistics.

* **Spanish text:** EXIST (Spanish partition)

Laura Plaza, Jorge Carrillo-de-Albornoz, Enrique Amigó, Julio Gonzalo, Roser Morante, Paolo Rosso, Damiano Spina, Berta Chulvi, Alba Maeso, and Víctor Ruiz. 2024. EXIST 2024: sEXism Identification in Social neTworks and Memes. In Advances in Information Retrieval: 46th European Conference on Information Retrieval, ECIR 2024, Glasgow, UK, March 24–28, 2024, Proceedings, Part V. Springer-Verlag, Berlin, Heidelberg, 498–504. https://doi.org/10.1007/978-3-031-56069-9_68

* **Multimodal:** HatReD

Ming Shan Hee, Wen-Haw Chong, and Roy Ka-Wei Lee. 2023. Decoding the underlying meaning of multimodal hateful memes. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI '23). Article 665, 5995–6003. https://doi.org/10.24963/ijcai.2023/665

> **Note:** All datasets are publicly available.

---
## Models
### Text Datasets (SBIC & EXIST-es)
- Qwen3-30B-A3B-Instruct-2507-FP8 (also used for SFT)
- Llama-3.3-70B-Instruct (also used for SFT)
- Kimi-K2-Instruct-0905
### Multimodal Dataset (HatReD)
- Qwen2.5-VL-32B-Instruct (also used for SFT)
- Qwen2.5-VL-72B-Instruct (also used for SFT)
- Llama-4-Scout-17B-16E-Instruct

---
## Usage
### Step 1: Prepare Dataset, Dependencies, and Resources
1. Create a new teamspace on [Lightning AI](https://lightning.ai/), following the [guide to create a teamspace](https://lightning.ai/docs/team-management/organizations/manage-teamspaces#create-a-teamspace).

2. Download all files from the `uploads` folder in this repository. Drag and drop them into the `uploads` folder on your Lightning AI teamspace's **Drive**.

> **Note:** The file ``uploads/SBIC_race_ethnicity_personas.json`` contains a persona ("an individual of mixed-race or of a race/ethnicity not otherwise listed") that was used in our runs but which contained a typographical error (missing the phrase "in the United States"). We subsequently ran the correct version ("an individual in the United States who is of mixed-race or of a race/ethnicity not otherwise listed") separately to make sure that all **SBIC** personas included the phrase "in the United States". To avoid incurring additional costs, we recommend deleting the incorrect version from the ``uploads/SBIC_race_ethnicity_personas.json`` file before execution.

3. Create a Fireworks AI API key, and save the key in a .env file named `FIREWORKS_CREDENTIALS.env`, in the following format
```bash
FIREWORKS_API_KEY=<insert your actual API key here>
```
4. Upload the .env file to the `uploads` folder on your Lightning AI teamspace's **Drive**.

#### SBIC Preparation
1. Create a new Studio in Lightning AI, named `0001_SBIC_preparation`.

2. Upload the following Python scripts (downloaded from the `0000_datasets_preparation/SBIC` folder in this GitHub repository) to the Studio:
    * `0000A_SBIC_preparation.py`
    * `0000B_SBIC_composite_sft_dev_preparation.py`
    * `0000B_SBIC_composite_sft_train_preparation.py`
    * `0000C_SBIC_specific_sft_preparation.py`
    * `0000C_SBIC_specific_sft_preparation_patch.py`
    * `0000D_SBIC_bygroup_preparation.py`
    * `0000D_SBIC_bygroup_preparation_patch.py`

3. The **SBIC** dataset is available via programmatic download in the script `0000A_SBIC_preparation.py`, hence no manual download required.

4. Sequentially, run the scripts in alphabetical order.

#### HatReD Preparation
1. Create a new Studio in Lightning AI, named `0001_HatReD_preparation`.

2. Upload the following Python scripts (downloaded from the `0000_datasets_preparation/HatReD` folder in this GitHub repository) to the Studio:
    * `0001A_HatReD_preparation.py`
    * `0001B_HatReD_sft_preparation.py`

3. The **HatReD** dataset (annotations) is available via programmatic download in the script `0001A_HatReD_preparation.py`, hence no manual download required for the annotations.

4. Download meme images from [Kaggle (Facebook Hateful Meme Dataset)](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset).
    * In the Explorer view sidebar on the left-hand side of the Lightning AI Studio interface, right-click and choose New Folder. Name it `FHM`,
    * Select all the meme image files and drag and drop them straight from your desktop into the Explorer view sidebar on the left-hand side of the Lightning AI Studio interface, under the folder `FHM`.

5. Sequentially, run the scripts in alphabetical order.

#### EXIST Preparation
1. Create a new Studio in Lightning AI, named `0002_EXIST_preparation`.

2. Complete the form at https://forms.office.com/pages/responsepage.aspx?id=SHBYtXCgrUO2VCCjHpstmc1J3Gu50zdMhFmXSTrhRZJUM0lEUzlTT0RZOE5TSldFWjFUSklRRDZIUS4u&route=shorturl, thereafter you will recieve a shared link via email to access the dataset.

3. Select `EXIST2024_training.json` and `EXIST2024_dev.json` and drag and drop them straight from your desktop into the Explorer view sidebar on the left-hand side of the Lightning AI Studio interface.

4. Upload the following Python scripts (downloaded from the `0000_datasets_preparation/EXIST` folder in this GitHub repository) to the Studio:
    * `0002A_EXIST_preparation.py`
    * `0002B_EXIST_composite_sft_train_preparation.py`
    * `0002C_EXIST_specific_sft_preparation.py`
    * `0002D_EXIST_bygroup_preparation.py`

5. Sequentially, run the scripts in alphabetical order.

### Step 2: Supervised Fine-tuning

1. Download all the JSONL files generated after running the `sft` preparation scripts.

2. Launch supervised fine-tuning jobs via the Fireworks AI UI, using the JSONL files and  the following hyperparameters: a LoRA rank of 16, a learning rate of 1&times;10<sup>-4</sup>, a maximum context length of 8,192, a token-based batch size of 32,768, a gradient accumulation step of 1, and 0 learning rate warmup steps, with training conducted for 3 epochs.

> **Note:** Scripts in this GitHub repository under the `1000_inference/scripts/SBIC`, `1000_inference/scripts/HatReD`, and `1000_inference/scripts/EXIST` folder, and containing a model value beginning with "accounts/jiawangexperiments-b89b99" on line 28, indicate scripts where inference is performed using fine-tuned models. You may wish to use the same name under ``Model Output Name`` when launching supervised fine-tuning jobs via the Fireworks AI UI. Names such as ``asian-llama-v3p3-70b-instruct`` and ``liberal-llama-v3p3-70b-instruct`` indicate specific (see paper) fine-tuning jobs for **SBIC**; names such as ``sbic-qwen3-30b-a3b-instruct-2507`` indicate composite (see paper) fine-tuning jobs for **SBIC**; names such as ``hatred-qwen2p5-vl-72b-instruct`` indicate composite fine-tuning jobs for **HatReD** (we only use composite fine-tuning for **HatReD**); names such as ``exist-man-qwen3-30b-a3b-instruct-2507`` indicate specific fine-tuning jobs for **EXIST-es**.; names such as ``exist-es-llama-v3p3-70b-instruct`` indicate composite fine-tuning jobs for **EXIST-es**.


### Step 3: Inference

1. For each row in the table below, create a new Studio in Lightning AI, with the studio name aligning with that stated below. Upload the corresponding Python scripts (downloaded from the `1000_inference` folder in this GitHub repository) as follows:

| Studio Name | Script to Upload |
| :--- | :--- |
| `1000_on_demand_inference` | `1000_on_demand_inference.py` |
| `1000_serverless_inference` | `1000_serverless_inference.py` |
| `1001_on_demand_vlm_inference` | `1001_on_demand_vlm_inference.py` |
| `1001_serverless_vlm_inference` | `1001_serverless_vlm_inference.py` |

2. Download all the Python scripts from the `1000_inference/scripts` folder. These scripts enable you to programatically launch Lightning AI Studios, specifically the four Studios dedicated for inferences as aforementioned. 

3. In a new Studio separate from the four aforementioned Studios, run
```bash
pip install lightning-sdk==2025.7.22
pip uninstall litai
```

Then, upload and run all the Python scripts downloaded from the `1000_inference/scripts` folder, in the Studio.

> **Note:** Scripts requiring dedicated Fireworks AI deployment will contain the --id parameter on line 29. This means you have to create a deployment on Fireworks AI, following the [deployment guide](https://docs.fireworks.ai/getting-started/ondemand-quickstart#deployments-quickstart). Then, change the value on line 29 accordingly to your deployment id. For fine-tuned models, also change the value on line 28 to the name of your fine-tuned model. Take note to change the Fireworks account ID to yours, even if your fine-tuned model has the same name. Note the deployment configurations (e.g., GPU hardware) used for deployment, from our paper.

### Step 4: Analysis

1. Create a folder in your Google Drive named `MaskedModX`, as well as the **sub-folders**, `SBIC`, `HatReD`, and `EXIST`.

2. Upload all the data files containing the inferences (from Step 3) into the respective Google Drive folders.

3. Download all the Colab notebooks (containing the R codes for analysis) from the `2000_analysis` folder in this GitHub repository.

4. Upload the Colab notebooks to your Google Drive and run the notebooks.
