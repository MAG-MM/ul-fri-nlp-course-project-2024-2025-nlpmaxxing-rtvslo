import torch
from common import device, model_path, format_inference_prompt, prepare_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import polars as pl

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

results = {
    "id": [],
    "target": [],
    "predicted": []
}

test_ds = prepare_dataset("tiny", rm_columns=["inputs"])

for i in tqdm(test_ds.iter(batch_size=1), total=test_ds.num_rows):
    
    prompt = format_inference_prompt({
        "input": i['input'][0],
        "target": i['target'][0], # this is the original output, which for inference is removed from prompt
    })

    results["id"].append(i['output'][0]['id'])
    results["target"].append(i['target'][0])

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_size = inputs.input_ids.shape[1] - 5 # to include '## Traffic Report:'

    generated_tokens = model.generate(inputs["input_ids"], max_new_tokens=256)

    output = tokenizer.decode(
        generated_tokens[0][input_size:], 
        skip_special_tokens=True
    )

    results["predicted"].append(output)


df = pl.from_dict(results)

df.write_csv("../test-results.csv")

