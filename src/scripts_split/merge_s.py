import torch
from common_s import model_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "GaMS-9B-Instruct" # 9B or 2B
base_model_path = f"cjvt/{base_model_id}"

model_tag = "v6-4096"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
#tokenizer.add_tokens(generate_special_tokens(), special_tokens=True)
tokenizer.save_pretrained(f"{model_path}-{base_model_id}-{model_tag}-MERGED")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    #torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16, # this was used for training
    device_map="auto"
)

# when warnings, this should be for training as well
base_model.generation_config.temperature=None
base_model.generation_config.top_p=None

#base_model.resize_token_embeddings(len(tokenizer))

peft_model = PeftModel.from_pretrained(
    base_model,
    model_id = f"{model_path}-{base_model_id}-{model_tag}",
    adapter_name="lora",
    is_trainable=False
)

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{model_path}-{base_model_id}-{model_tag}-MERGED")






















