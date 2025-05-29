import torch
from common import model_path, format_training_prompt, prepare_dataset, generate_special_tokens, convert_to_gemma_chat
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

base_model_id = "GaMS-9B-Instruct" # 9B or 2B
base_model_path = f"cjvt/{base_model_id}"

model_tag = "v5-2048"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print("device is", device)

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
    device_map='auto'
)

# when warnings
model.generation_config.temperature=None
model.generation_config.top_p=None

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# tokenizer.add_tokens(generate_special_tokens(), special_tokens=True)
# model.resize_token_embeddings(len(tokenizer))

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
    return tokenizer.apply_chat_template(convert_to_gemma_chat(example), tokenize=False, add_generation_prompt=False) # + tokenizer.eos_token

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
    max_length=2048, 
)

train_ds = prepare_dataset("train")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    # add eval dataset later
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

trainer.save_model(f"{model_path}-{base_model_id}-{model_tag}")













