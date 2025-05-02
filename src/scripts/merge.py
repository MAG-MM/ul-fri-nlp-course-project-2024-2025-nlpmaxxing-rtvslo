import torch
from common import model_path, generate_special_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "GaMS-2B" # 9B or 2B
base_model_path = f"cjvt/{base_model_id}"

model_tag = "len1024bs4-v3"

#https://www.datacamp.com/tutorial/fine-tuning-gemma-2


tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.add_tokens(generate_special_tokens(), special_tokens=True)
tokenizer.save_pretrained(f"{model_path}-{base_model_id}-{model_tag}-MERGED")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    #torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16, # this was used for training
    device_map="auto"
)

base_model.resize_token_embeddings(len(tokenizer))

peft_model = PeftModel.from_pretrained(
    base_model,
    model_id = f"{model_path}-{base_model_id}-{model_tag}",
    adapter_name="lora",
    is_trainable=False
)


merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{model_path}-{base_model_id}-{model_tag}-MERGED")






















