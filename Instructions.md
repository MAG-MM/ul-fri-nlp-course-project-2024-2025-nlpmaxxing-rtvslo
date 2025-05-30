### Resources

We keep the **data**, **evaluation results** and **models** out of github.

The git-ignored contents will be available [here](https://unilj-my.sharepoint.com/:f:/g/personal/ms88481_student_uni-lj_si/EhlLZK7SJAxAlZvr1QZSUPQBqva_nnZvL4NGTRz8GnFUMQ?e=twjNdd) and everyone part of the UniLJ network can download them. They should be placed within it's corresponding data locations.

### Code

The `/src` directory contains source code for the project.

### Local Environment

Create a virtual environment: `python -m venv .venv`

Source it:

- For Mac or Linux, run: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

Install dependencies: `pip install -r requirements.txt`

### Dependencies (up to date 30.05.2025):

`pip install transformers datasets accelerate peft trl bitsandbytes protobuf blobfile sentencepiece polars`

### Running locally - preparations

To run this project successfully you need the data the following files generate. The order should be maintained.

1. Data Preprocessing:

   1. `data-consolidate-input.ipynb`:

      - consolidates all inputs from the excel file into csv, strips html tags, parses timestamp and prepares the data for further processing
      - creates the file `data/inputs-clean.csv`
      - result also available in the **Resources Shared Directory**

   2. `data-consolidate-output.ipynb`:

      - parses all output files and consolidates them into csv. Also parses date and time for further processing
      - creates the files `data/outputs-bad-dates.csv` and `data/outputs-clean.csv`
      - result also available in the **Resources Shared Directory**

   3. `data-inspect.ipynb`:

      - **optional**
      - Inspects the input and output csv files

   4. `data-merge-inputs-outputs.ipynb`:

      - Matches multiple inputs to one output with a time threshold of 3 minutes before the air time of the output. It also splits the dataset on train and test with 80:20 ratio
      - creates the files `data/dataset-full-debug.csv`, `data/dataset-full.csv`, `data/dataset-train.csv` and `data/dataset-test.csv`
      - result also available in the **Resources Shared Directory**

   5. `data-split-testset.ipynb`:
      - further splits the test into a mini test for faster evaluation
      - creates the file `data/dataset-test-mini.csv`
      - result also available in the **Resources Shared Directory**

2. Dataset preparation - depends on **Data Preprocessing** files

   1. `data-hf.ipynb`:

      - legacy, used by an older approach, required only if you try that approach exclusively for the scripts under `src/scripts`
      - adapts the data for fine tunning and converts it into a HuggingFace Dataset format
      - will create a directory structure under `data/hf`
      - **You need to run this** when you have the data from the earlier notebooks (downloaded or generated with the notebooks)

   2. `data-hf-single-in-out.ipynb`:
      - **required** for fine-tunning or inference of the latest model, used by the scripts under `src/scripts`
      - adapts the data for fine tunning and converts it into a HuggingFace Dataset format - for batched training where a sequence is one input and one output
      - will create a directory structure under `data/hf-split`
      - **You need to run this** when you have the data from the earlier notebooks (downloaded or generated with the notebooks)

### Running locally - fine-tuning

Once you have the `data/hf-split` generated, you can start the fine-tunning process.

The requirement is to have the local environment prepared and run the following scripts from the `src` directory:

- `python scripts_split/train_s.py` - this will teach the model and save the adapters in the `models` directory.
- `python scripts_split/merge_s.py` - this will merge the adapter weights with the original model.

To run this in `slurm`, there are shell example scripts under `src/scripts_split`. Also there is an example script in `src\scripts\prepare.sh` in order to prepare a singularity container.

### Running locally - inference

Once you have the `data/hf-split` generated, and the model trained or download and extracted inside `models`, you can run inference.

Running the script `python scripts_split/test_s.py` will run inference on the entire `test-mini` dataset and save the results inside the `test` directory.

You can adapt this script for a single example or use the `demo.ipynb` notebook.

Scripts:

- `scripts`: Python scripts for running locally and in the HPC cluster with the data with all inputs combined to form an output
- `scripts_split`: Python scripts for running locally and in the HPC cluster with the data chunked input by input to form an output to achieve fine-tunning without going over the maximum possible token count limit. This should be consider the final code for fine-tunning and running inference on a test set.

After a local environment is set up and the gitignored files are downloaded, run the `test_s.py` file inside the `scripts_split` directory. There is a sample command at the bottom of the file.

### Data preprocess

To run preprocessing, run consolidate input, output, then merge inputs and outputs.

### List of packages

Install pytorch.

pip install `jupyter polars fastexcel lxml tqdm striprtf scikit-learn`

`pip install transformers datasets accelerate peft trl bitsandbytes protobuf`

### Slurm scripts:

- `prepare.sh` will create a container and provision it with the required packages.
- `train.sh` will add some env vars and run the `train.py` script from within a container.
- `schedule-tran.sh` will simply create a queued job to run the `train.sh`
- Your home directory should have: containers, logs, data and scripts subdirectories.
