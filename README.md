# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

This project aims to use an existing LLM, »fine-tune« it, leverage prompt engineering techniques to generate short traffic reports. You are given Excel data from promet.si portal and your goal is to generate regular and important traffic news that are read by the radio presenters at RTV Slovenija. You also need to take into account guidelines and instructions to form the news. Currently, they hire students to manually check and type reports that are read every 30 minutes.

Methodology

1. Literature Review: Conduct a thorough review of existing research and select appropriate LLMs for the task. Review and prepare an exploratory report on the data provided.

2. Initial solution: Try to solve the task initially only by using prompt engineering techniques.

3. Evaulation definition: Define (semi-)automatic evaluation criteria and implement it. Take the following into account: identification of important news, correct roads namings, correct filtering, text lengths and words, ...

4. LLM (Parameter-efficient) fine-tuning: Improve an existing LLM to perform the task automatically. Provide an interface to do an interactive test.

5. Evaluation and Performance Analysis: Assess the effectiveness of each technique by measuring improvements in model performance, using appropriate automatic (P, R, F1) and human evaluation metrics.

### Team: nlpmaxxing

### Instructions for running

Check `Instructions.md`

### Report

The `/report` directory contains LaTeX source code and generated pdf for the assignment report.

### Result evaluation:

- Use `src/view_results.ipynb` to check the results. If you want to run the file, download the actual [file](https://unilj-my.sharepoint.com/:x:/g/personal/ms88481_student_uni-lj_si/EaC0oJgOflBHvQ41gMT-AC4Bg7Tz9u-XBi8CF6Ek1KlvjQ?e=QTeJM5) and put it in the correct path `../outputs/test-results-GaMS-2B-Instruct-v6-4096-MERGED-eval-scores.csv`. And obviously install the dependencies.
