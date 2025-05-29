from datasets import Dataset
from typing import Literal
from tqdm import tqdm

model_path = "../models/nlpmaxxing-rtvslo-trfc-split"

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

def format_single_input(input_item):
    input_string = f"## Input {input_item['input_index']} of {input_item['total_inputs']}:\n"
    for tag in input_tags:
        if input_item[tag] != None:
            input_string = input_string + f"### {tag}:\n{input_item[tag]}\n"

    return input_string.strip() + "\n\n"

def format_sample(example):
    input_s = format_single_input(example['input'])

    content = example['output']['content']
    content_parts = example['output']['content'].split("\n\n")
    if len(content_parts) > 1:
        header = content_parts[0]
        content = content.replace(header, "")

    return {
        "input": input_s,
        "input_index": example['input']["input_index"],
        "total_inputs": example['input']["total_inputs"],
        "target": content
    }



def format_gemma_chat_input(input_data):
    return f"""
Generate a traffic report from the following input data.

{input_data['input']}
"""

def format_gemma_chat_input_improve(input_data, previous_output):
    return f"""
[Continuing from the traffic report generate during the previous input]

{previous_output}

Generate a traffic report from the following input data.

{input_data['input']}
"""

def format_gemma_chat_output(input_data):
    return f"""
## Traffic Report:

{input_data['target']}
"""

def format_gemma_chat_output_improve(previous_output):
    return f"""
[Continuing from the traffic report generate during the previous input]

{previous_output}
"""

def convert_to_gemma_chat(sample):
    return [
        {"role": "user", "content": format_gemma_chat_input(sample)},
        {"role": "model", "content": format_gemma_chat_output(sample)}
    ]

def convert_to_gemma_chat_inference(sample):
    return [
        {"role": "user", "content": format_gemma_chat_input(sample)},
    ]

def convert_to_gemma_chat_inference_improve(sample, previous_output):
    return [
        {"role": "user", "content": format_gemma_chat_input_improve(sample, previous_output)},
        
    ]

def prepare_dataset(name: Literal['tiny', 'test', 'test-mini', 'train'], rm_columns = ["input", "output"]):
    dataset = Dataset.load_from_disk(f"../data/hf-split/dataset-{name}")
    return dataset.map(format_sample, remove_columns=rm_columns)


def count_tokens(tokenizer, ds):
    counts = []
    for i in tqdm(ds.iter(batch_size=1), total=ds.num_rows):
        p = tokenizer.apply_chat_template(convert_to_gemma_chat(i), tokenize=False, add_generation_prompt=False)
        counts.append(len(tokenizer.encode(p)))

    print("Maximum tokens count:", max(counts))