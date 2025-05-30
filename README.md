# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

This project aims to use an existing LLM, »fine-tune« it, leverage prompt engineering techniques to generate short traffic reports. You are given Excel data from promet.si portal and your goal is to generate regular and important traffic news that are read by the radio presenters at RTV Slovenija. You also need to take into account guidelines and instructions to form the news. Currently, they hire students to manually check and type reports that are read every 30 minutes.

Methodology

1. Literature Review: Conduct a thorough review of existing research and select appropriate LLMs for the task. Review and prepare an exploratory report on the data provided.

2. Initial solution: Try to solve the task initially only by using prompt engineering techniques.

3. Evaulation definition: Define (semi-)automatic evaluation criteria and implement it. Take the following into account: identification of important news, correct roads namings, correct filtering, text lengths and words, ...

4. LLM (Parameter-efficient) fine-tuning: Improve an existing LLM to perform the task automatically. Provide an interface to do an interactive test.

5. Evaluation and Performance Analysis: Assess the effectiveness of each technique by measuring improvements in model performance, using appropriate automatic (P, R, F1) and human evaluation metrics.

### Data

We will keep data out of version control.

The git-ignored contents will be available [here](https://unilj-my.sharepoint.com/:f:/g/personal/ms88481_student_uni-lj_si/Eg-AwdBXjatHhmnU9rrx2B0BQ0d61h3-_Jks1pwqtcrYBQ?e=6NGF0B) and everyone part of the UniLJ network can download them. They should be placed within it's corresponding data locations.

### Report

The `/report` directory contains LaTeX source code and generated pdf for the assignment report.

### Code

The `/src` directory contains source code for the project.

Notebooks:

- `data-consolidate-input`: consolidates all inputs from the excel file into csv, strips html tags, parses timestamp and prepares the data for further processing.
- `data-consolidate-output`: parses all output files and consolidates them into csv. Also parses date and time for further processing.
- `data-inspect`: Inspect the input and output csv files
- `data-merge-inputs-outputs`: Matches multiple inputs to one output with a time threshold of 3 minutes before the air time of the output. It also splits the dataset on train and test with 80:20 ratio.
- `data-split-testset`: further splits the test into a mini test for faster evaluation.
- `data-hf`: adapts the data for fine tunning and converts it into a HuggingFace Dataset format
- `data-hf-single-in-out`: adapts the data for fine tunning and converts it into a HuggingFace Dataset format - for batched training where a sequence is one input and one output.

- `evaluate_results`: the notebook for evaluating generated reports with evaluation metrics. Loads the csv file with generated and reference reports, calulates different metrics, draws some graphs for easier undertanding and then finds a few best reports based on chosen selection metrics.
- `view_results`: similar to `evaluate_results`, but it immediately loads the csv file with reports and coresponding metrics instead of calculating them. The file is available [here](https://unilj-my.sharepoint.com/:x:/g/personal/ms88481_student_uni-lj_si/EaC0oJgOflBHvQ41gMT-AC4Bg7Tz9u-XBi8CF6Ek1KlvjQ?e=QTeJM5). **Use this to check our results.**

Scripts:

- `scripts`: Python scripts for running locally and in the HPC cluster with the data with all inputs combined to form an output
- `scripts_split`: Python scripts for running locally and in the HPC cluster with the data chunked input by input to form an output to achieve fine-tunning without going over the maximum possible token count limit. This should be consider the final code for fine-tunning and running inference on a test set.

After a local environment is set up and the gitignored files are downloaded, run the `test_s.py` file inside the `scripts_split` directory. There is a sample command at the bottom of the file.

Evaluation:
- `/scripts/testing`: contains files used for evaluating generated reports.
- `/scripts/testing/evaluate_reports.py`: actual evaluation metrices. For GPT evaluation you need a valid API key and credits.
- `/scripts/testing/helper_functions.py`: functions used for displaying results.
- `/scripts/testing/evaluation_examples.ipynb`: contains some synthetic examples for checking evaluation.


### Local Environment

Create a virtual environment: `python -m venv .venv`

Source it:

- For Mac or Linux, run: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`


### Dependencies (up to date 30.05.2025):

`pip install transformers datasets accelerate peft trl bitsandbytes protobuf blobfile sentencepiece polars`

Also run: `pip install -r requirements.txt`

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


### Result evaluation:
- Use `src/view_results.ipynb` to check the results. If you want to run the file, download the actual [file](https://unilj-my.sharepoint.com/:x:/g/personal/ms88481_student_uni-lj_si/EaC0oJgOflBHvQ41gMT-AC4Bg7Tz9u-XBi8CF6Ek1KlvjQ?e=QTeJM5) and put it in the correct path `../outputs/test-results-GaMS-2B-Instruct-v6-4096-MERGED-eval-scores.csv`. And obviously install the dependencies.
