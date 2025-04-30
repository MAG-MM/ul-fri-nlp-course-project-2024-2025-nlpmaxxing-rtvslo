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

### Local Environment

Create a virtual environment: `python -m venv .venv`

Source it:

- For Mac or Linux, run: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

Install dependencies: `pip install -r requirements.txt`

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
