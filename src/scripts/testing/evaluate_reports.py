from bert_score import BERTScorer
# import openai
from typing import Dict, List, Set
# import torch
import numpy as np
import classla


classla.download("sl")


def extract_named_entities(text: str, lang='sl') -> Set[str]:
    nlp = classla.Pipeline(lang=lang, processors='tokenize,ner', verbose=False)
    doc = nlp(text)
    named_entities = set()

    for sent in doc.sentences:
        for ent in sent.ents:
            ent_text = ent.text.lower().strip()
            # ent_type = ent.type  # e.g., LOC, PER, ORG

            named_entities.add(ent_text)

            # if ent_type not in entity_dict:
            #     entity_dict[ent_type] = set()
            # entity_dict[ent_type].add(ent_text)

    return named_entities


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

    def length_diff(self, generated: List[str], reference: List[str]):
        """
        Compare lengths of generated reports.
        """
        lengths = []
        for i in range(len(generated)):
            lengths.append((len(generated[i].split()) - len(reference[i].split())) / max(len(reference[i].split()), 1))
        average_absolute_difference = np.mean(np.abs(lengths))
        return average_absolute_difference, lengths

    def named_entity_evaluation(self, generated: List[str], reference: List[str]):
        precision = []
        recall = []
        f1 = []
        for i in range(len(generated)):
            ne_g = extract_named_entities(generated[i])
            ne_r = extract_named_entities(reference[i])
            true_positives = ne_g & ne_r

            precision.append(len(true_positives) / len(ne_g) if ne_g else 1.0 if not ne_r else 0.0)
            recall.append(len(true_positives) / len(ne_r) if ne_r else 1.0 if not ne_g else 0.0)
            f1.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1])
                      if (precision[-1] + recall[-1]) else 0.0)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


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
    
    # bertscore_results = evaluator.bert_scores(generated, references)
    # length_res_avg, length_results = evaluator.length_diff(generated, references)
    # # print(bertscore_results)
    # for key, value in bertscore_results.items():
    #     print(f"{key}: ", value)
    # print("length diff abs avg: ", length_res_avg)
    # print("length diff: ", length_results)
    
    # res = extract_named_entities(generated[0], nlp)
    evaluator.named_entity_evaluation(generated, references)
    
    print("")
