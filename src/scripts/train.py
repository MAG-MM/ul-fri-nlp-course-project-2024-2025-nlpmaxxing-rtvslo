import torch
from common import model_path, format_training_prompt, prepare_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

base_model_id = "GaMS-9B" # 9B or 2B
base_model_path = f"cjvt/{base_model_id}"

model_tag = "len1024bs2"

#device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

#print("device is", device)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, #'DTensor' object has no attribute 'compress_statistics' in 4bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # torch.float16
    #bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    attn_implementation='eager',
    torch_dtype=torch.bfloat16,
    #device_map=device
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    #target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

def formatting_prompts_func(example):
    return format_training_prompt(example)

training_args = SFTConfig(
    output_dir=f"../outputs/gams-{base_model_id}-finetune-{model_tag}",
    per_device_train_batch_size=2, # 4 with 512, 2 with 1024
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=50,
    num_train_epochs=1, # 3, load best model
    fp16=True,
    save_strategy="epoch",
    max_length=512, # 1024 or 2048
)

train_ds = prepare_dataset("train")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    # add eval dataset later
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()

trainer.save_model(f"{model_path}-{base_model_id}-{model_tag}")



# "len512bs4" took 2h and 30 mins
# "len1024bs2" took 4h and 30 mins