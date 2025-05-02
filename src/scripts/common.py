from datasets import Dataset
from typing import Literal

#device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_path = "../models/nlpmaxxing-rtvslo-trfc-cluster"

input_tags = [
    "A1",
    "B1",
    "C1",
    "TitlePomembno",
    "ContentPomembno",
    "TitleNesrece",
    "ContentNesrece",
    "TitleZastoji",
    "ContentZastoji",
    "TitleVreme",
    "ContentVreme",
    "TitleOvire",
    "ContentOvire",
    "TitleDeloNaCesti",
    "ContentDeloNaCesti",
    "TitleOpozorila",
    "ContentOpozorila",
    "TitleMednarodneInformacije",
    "ContentMednarodneInformacije",
    "TitleSplosno",
    "ContentSplosno"
]

def format_training_prompt(input_data):
    return f"""
Generate a traffic report from the following input data.

## Inputs:

{input_data['input']}

## Traffic Report:

{input_data['target']}
"""

def format_inference_prompt(input_data):
    return f"""
Generate a traffic report from the following input data.

## Inputs:

{input_data['input']}

## Traffic Report:
"""

def format_single_input(input_item, i):
    input_string = f"\n### Input_{i+1}:\n"
    for tag in input_tags:
        if input_item[tag] != None:
            input_string = input_string + f"\n#### {tag}:\n{input_item[tag]}\n"

    return input_string.strip()

def format_sample(example):
    inputs = ""
    for i in range(0, len(example['inputs'])):
        inputs = inputs + format_single_input(example['inputs'][i], i)

    content = example['output']['content']
    content_parts = example['output']['content'].split("\n\n")
    if len(content_parts) > 1:
        header = content_parts[0]
        content = content.replace(header, "")

    return {
        "input": inputs,
        "target": content
    }

def prepare_dataset(name: Literal['tiny', 'test', 'test-mini', 'train'], rm_columns = ["inputs", "output"]):
    dataset = Dataset.load_from_disk(f"../data/hf/dataset-{name}")
    return dataset.map(format_sample, remove_columns=rm_columns)

def generate_special_tokens():
    tokens = ['## Inputs:', '## Traffic Report:']
    for tag in input_tags:
        tokens.append(f"#### {tag}:")

    for i in range(20):
        tokens.append(f"### Input_{i+1}:")

    return tokens
