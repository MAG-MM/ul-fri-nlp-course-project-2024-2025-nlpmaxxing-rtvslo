import torch
from common_s import model_path, prepare_dataset, convert_to_gemma_chat_inference, count_tokens, convert_to_gemma_chat_inference_improve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import polars as pl

base_model_id = "GaMS-2B-Instruct" # 9B or 2B
model_tag = "v6-4096-MERGED"

ft_model_path = f"{model_path}-{base_model_id}-{model_tag}"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    ft_model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(ft_model_path)

def save_all(res):
    df = pl.from_dict(res)
    df.write_csv(f"../test/test-results-{base_model_id}-{model_tag}.csv")
    print("Test results saved.")

def clean_output(text):
    parts = text.split("<start_of_turn>model")

    if len(parts) != 2:
        return "BAD OUTPUT"
    
    out = parts[1].strip()

    out = out.replace("## Traffic Report:", "")
    out = out.replace("<end_of_turn>", "")
    out = out.replace("<eos>", "")

    return out.strip()

results = {
    "id": [],
    "target": [],
    "predicted": []
}

test_ds = prepare_dataset("test-mini", rm_columns=[])

# count_tokens(tokenizer=tokenizer, ds=test_ds) # this can only be ran once to make sure there are no inputs over the max length limit

previous_output = ""

for i in tqdm(test_ds.iter(batch_size=1), total=test_ds.num_rows):
    sample = {
        "input": i['input'][0],
        "target": i['target'][0], # this is the original output, which for inference is removed from prompt
    }

    input_index = i['input_index'][0]
    total_inputs = i['total_inputs'][0]

    s = convert_to_gemma_chat_inference_improve(sample, previous_output)
    if (input_index == 1):
        s = convert_to_gemma_chat_inference(sample)


    prompt = tokenizer.apply_chat_template(s, tokenize=False, add_generation_prompt=False)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_new_tokens=512)

    output = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=False
    )

    output = clean_output(output)
    if output == "BAD OUTPUT":
        print("-------------------------------------------------Bad output---------------------------------------")

    previous_output = output

    if input_index == total_inputs:
        results["id"].append(i['output'][0]['id'])
        results["target"].append(i['target'][0])
        results["predicted"].append(output)
        previous_output = ""
        #print("-------------------------------------------------Completed---------------------------------------")
        #print(output)
        #print("-------------------------------------------------Original---------------------------------------")
        #print(i['target'][0])
        #break



save_all(results)



'''
v6 4096 model evaluation on the test-mini split took 4 hours on HPC H100 GPU


run this script from within src directory like: python scripts_split/test_s.py assuming you have the model in the root models directory

'''