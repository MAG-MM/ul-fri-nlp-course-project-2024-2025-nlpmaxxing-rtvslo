import torch
from common_s import model_path, prepare_dataset, convert_to_gemma_chat, count_tokens
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

base_model_id = "GaMS-2B-Instruct"
base_model_path = f"cjvt/{base_model_id}"
model_tag = "v6-4096"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print("device is", device)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    attn_implementation='eager',
    torch_dtype=torch.bfloat16,
    device_map='auto',
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
    return tokenizer.apply_chat_template(convert_to_gemma_chat(example), tokenize=False, add_generation_prompt=False)

training_args = SFTConfig(
    output_dir=f"../outputs/gams-{base_model_id}-finetune-{model_tag}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=50,
    num_train_epochs=1,
    fp16=False,
    bf16=True,
    save_strategy="epoch",
    max_length=4096,
)

train_ds = prepare_dataset("train")

count_tokens(tokenizer=tokenizer, ds=train_ds) # if this number is more than the max_length, chunk in smaller pieces

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

trainer.save_model(f"{model_path}-{base_model_id}-{model_tag}")


'''
training the v6 4096 model on HPC cluster H100 took 14:23 hours
'''