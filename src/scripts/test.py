import torch
from common import model_path, format_inference_prompt, prepare_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import polars as pl

base_model_id = "GaMS-2B" # 9B or 2B
model_tag = "len1024bs2" # faster: len512bs4 len1024bs2 :slower
model_tag = "len1024bs4-v3-MERGED"

ft_model_path = f"{model_path}-{base_model_id}-{model_tag}"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, #'DTensor' object has no attribute 'compress_statistics' in 4bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # torch.float16
    #bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    ft_model_path,
    quantization_config=bnb_config,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

#model.to(device)

model.eval()

model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(ft_model_path)

results = {
    "id": [],
    "target": [],
    "predicted": []
}

test_ds = prepare_dataset("test-mini", rm_columns=["inputs"])

counter = 0

for i in tqdm(test_ds.iter(batch_size=1), total=test_ds.num_rows):
    
    prompt = format_inference_prompt({
        "input": i['input'][0],
        "target": i['target'][0], # this is the original output, which for inference is removed from prompt
    })

    results["id"].append(i['output'][0]['id'])
    results["target"].append(i['target'][0])

    #inputs = tokenizer(prompt, max_length=256, truncation=True, padding="max_length", return_tensors='pt').to(device) 
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_size = inputs.input_ids.shape[1] - 5 # to include '## Traffic Report:'
    input_size = inputs.input_ids.shape[1] - 2 # to include '## Traffic Report:', for v3

    with torch.no_grad():
        generated_tokens = model.generate(inputs["input_ids"], max_new_tokens=256) # max_new_tokens=256

    output = tokenizer.decode(
        generated_tokens[0][input_size:],
        skip_special_tokens=False
    )

    results["predicted"].append(output)

df = pl.from_dict(results)

df.write_csv(f"../test/test-results-{base_model_id}-{model_tag}.csv")

