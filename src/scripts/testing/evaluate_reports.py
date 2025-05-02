from bert_score import BERTScorer
import openai
from typing import Optional, Dict, List, Tuple
import re
import torch
import numpy as np


class TrafficReportEvaluator:
    def __init__(self, lang='sl', model='xlm-roberta-large'):
        self.lang = lang
        self.bertscore_model = model

    def bert_scores(self, generated: List[str], references: List[str]):
        """
        Compute BERTScore Precision, Recall, and F1 between two lists of texts.
        """
        scorer = BERTScorer(
            lang=self.lang,
            model_type=self.bertscore_model,
            rescale_with_baseline=False     # gives not ideal scores for Slovenian
        )
        precision, recall, f1 = scorer.score(generated, references)
        return {
            'bertscore_precision': precision,
            'bertscore_precision_mean': precision.mean().item(),
            'bertscore_recall': recall,
            'bertscore_recall_mean': recall.mean().item(),
            'bertscore_f1': f1,
            'bertscore_f1_mean': f1.mean().item()
        }

    def length_diff(self, generated: List[str], reference: List[str]) -> List[float]:
        """
        Compare lengths of generated reports.
        """
        lengths = []
        for i in range(len(generated)):
            lengths.append((len(generated[i].split()) - len(reference[i].split())) / max(len(reference[i].split()), 1))
        average_absolute_difference = np.mean(np.abs(lengths))
        return average_absolute_difference, lengths


if __name__ == "__main__":
    generated = [
        "Zaradi prometne nesreče je zaprta avtocesta A1 pri Celju.",
        "Fakulteta za računalništvo in informatiko",
        "Danes je lep dan, saj je lepo, sončno vreme.",
        "Danes je lep dan, saj je lepo, sončno vreme.",
        "Nek naključni testni niz.",
        "ja"
    ]
    references = [
        "Na avtocesti A1 pri Celju je zaradi nesreče zaprt promet.",
        "Fakulteta za računalništvo ter informatiko",
        "Na nebu sije sonce, kar je lepo.",
        "Danes pada sneg in dež.",
        "Komaj čakam na poletje, da se mi ni treba več s faksom ukvarjat.",
        "ne"
    ]
    evaluator = TrafficReportEvaluator()
    # evaluator = TrafficReportEvaluator(model='bert-base-multilingual-cased')
    bertscore_results = evaluator.bert_scores(generated, references)
    length_res_avg, length_results = evaluator.length_diff(generated, references)
    # print(bertscore_results)
    for key, value in bertscore_results.items():
        print(f"{key}: ", value)
    print("length diff abs avg: ", length_res_avg)
    print("length diff: ", length_results)
