import time
import sys

import_start = time.time()
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/scratch/$USER/hf-cache"

import torch
torch.manual_seed(42)

import pandas as pd
import argparse
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

# Measure import end time
import_end = time.time()
print(f"Imports completed in {import_end - import_start:.2f} seconds.")

def build_prompt(timestamp, program, context, template):
    return f"""## NAVODILO:
Si prometni poročevalec. Na podlagi spodnjega konteksta napiši jasno in jedrnato prometno poročilo za program ob danem času za zadnje pol ure.
Uporabi enak format in stil kot v primerih.
Piši jedrnato — omeniti je treba le najpomembnejše informacije.

### ČAS:
{timestamp}

### PROGRAM:
{program}

### KONTEKST:
{context[:3000]}

### PRIMERI:
{template}

### GENERIRANO POROČILO:
"""

def main():
    parser = argparse.ArgumentParser(description="Generate traffic reports using Gemma.")
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument("output_csv", help="Path to output CSV file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Hugging Face model folder.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--resume-errors", action="store_true", help="Only reprocess rows where report_output starts with ERROR.")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    print(f"Setting up pipeline with model from {model_path}")

    try:
        pipeline_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        time.sleep(6)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size
        )

        pipeline_end = time.time()
        print(f"Pipeline created successfully in {pipeline_end - pipeline_start:.2f} seconds.")

    except Exception as e:
        print(f"Error creating pipeline: {str(e)}")
        sys.exit(1)

    template = """Prometne informacije\t 28. 04. 2022\t 16.30\t 2. program\n\n
Podatki o prometu.\n\n
Promet proti primorski avtocesti je sedaj upočasnjen samo še med razcepom Kozarje in priključkom Brezovica.\n
Tudi v nasprotni smeri, torej proti Ljubljani, promet ni več oviran zaradi nesreče na razcepu Kozarje.\n
Na dolenjski avtocesti proti Obrežju je zaradi pnevmatike na vozišču oviran promet pred priključkom Višnja Gora.\n
Na mejnem prehodu Gruškovje je povečan promet osebnih vozil pri izstopu iz države, vozniki tovornih vozil pa na vstop in izstop čakajo eno uro.\n
---
Prometne informacije\t 31. 12. 2023\t 11.00\t 1. in 3. program\n
Podatki o prometu.\n
Na avtocestnem odseku od Gabrka proti Fernetičem je zaradi del zaprt prehitevalni pas pred mejnim prehodom Fernetiči.\n
Zaradi praznikov bo od danes do torka od osmih do 22-ih prepovedan promet za tovorna vozila, težja od 7 ton in pol.\n
---
Prometne informacije\t 31.07.2024\t 15.00\t 1. in 3. program\n
Podatki o prometu\n
Na glavni cesti Hrastnik-Zidani Most odstranjuejo posledice prometne nesreče zato je tam promet urejen izmenično enosmerno.\n
Zastoji so na primorski avtocesti na posameznih odsekih od Razdrtega do Ljubljane. Čas potovanja se na tem odseku podaljša za približno 30 minut.\n
Na štajerski avtocesti sta zastoja pred  počivališčem Polskava proti Mariboru ter pred mejnim prehodom Šentilj.\n
Kilometer dolg zastoj je tudi pred predorom Karavanke proti Avstrji.\n
"""

    template = """Prometne informacije\t 28. 04. 2022\t 16.30\t 2. program\n\n
Podatki o prometu.\n\n
Promet proti primorski avtocesti je sedaj upočasnjen samo še med razcepom Kozarje in priključkom Brezovica.\n
Tudi v nasprotni smeri, torej proti Ljubljani, promet ni več oviran zaradi nesreče na razcepu Kozarje.\n
Na dolenjski avtocesti proti Obrežju je zaradi pnevmatike na vozišču oviran promet pred priključkom Višnja Gora.\n
Na mejnem prehodu Gruškovje je povečan promet osebnih vozil pri izstopu iz države, vozniki tovornih vozil pa na vstop in izstop čakajo eno uro.\n
---
Prometne informacije\t 31. 12. 2023\t 11.00\t 1. in 3. program\n
Podatki o prometu.\n
Na avtocestnem odseku od Gabrka proti Fernetičem je zaradi del zaprt prehitevalni pas pred mejnim prehodom Fernetiči.\n
Zaradi praznikov bo od danes do torka od osmih do 22-ih prepovedan promet za tovorna vozila, težja od 7 ton in pol.\n
"""

    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.limit is not None:
        df = df.head(args.limit)

    if 'report_output' not in df.columns:
        df['report_output'] = ""

    # Handle resume-errors option
    if args.resume_errors and Path(args.output_csv).exists():
        print(f"Resuming from {args.output_csv} and only reprocessing rows with ERROR reports...")
        df_existing = pd.read_csv(args.output_csv)

        # Safety check: make sure inputs match
        if not df_existing['input'].equals(df['input']):
            print("Error: Input CSV and output CSV do not match. Cannot safely resume.")
            sys.exit(1)

        # Update 'report_output' column from existing file output file
        df['report_output'] = df_existing['report_output']

        # Identify which rows need reprocessing
        rows_to_process = df['report_output'].str.startswith("ERROR:") | df['report_output'].isna() | (df['report_output'] == "")
        indices_to_process = df.index[rows_to_process].tolist()
        print(f"Found {len(indices_to_process)} rows to reprocess.")
    else:
        indices_to_process = df.index.tolist()

    print(f"Processing {len(indices_to_process)} rows.")

    # Prebuild prompts
    prompts = []
    idx_mapping = []  # To know which df index each prompt belongs to
    for idx in indices_to_process:
        row = df.loc[idx]
        prompts.append(build_prompt(row['timestamp'], row['programs'], row['input'], template))
        idx_mapping.append(idx)

    # Batched processing
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating reports"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        print(f"\nProcessing prompts {batch_start} to {batch_end-1}...")
        try:
            gen_start = time.time()

            results = generator(
                batch_prompts,
                do_sample=True,
                handle_long_generation='hole',
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=1024,
                return_full_text=False
            )

            for idx_in_batch, result_list in enumerate(results):
                df_idx = idx_mapping[batch_start + idx_in_batch]
                if len(result_list) > 0:
                    generated_text = result_list[0]['generated_text']
                    output = generated_text.strip()
                    df.at[df_idx, 'report_output'] = output
                else:
                    df.at[df_idx, 'report_output'] = "ERROR: Empty generation"

        except Exception as e:
            print(f"Error during generation for batch {batch_start}-{batch_end-1}: {str(e)}")
            for idx_in_batch in range(batch_end - batch_start):
                df_idx = idx_mapping[batch_start + idx_in_batch]
                df.at[df_idx, 'report_output'] = f"ERROR: {str(e)}"

        # Save partial results after each batch
        df[['input', 'output', 'report_output']].to_csv(args.output_csv, index=False)
        
        gen_end = time.time()
        print(f"Batch {batch_start}-{batch_end-1} completed and saved ({gen_end - gen_start:.2f} s).")

        gc.collect()
        torch.cuda.empty_cache()

    total_seconds = time.time() - import_end
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    print(f"Done. Processed {len(indices_to_process)} rows. Output saved to {args.output_csv} (ran {hours}h {minutes}m {seconds}s).")


if __name__ == "__main__":
    main()