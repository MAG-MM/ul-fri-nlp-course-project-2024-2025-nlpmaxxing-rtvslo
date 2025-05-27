from bert_score import BERTScorer
# import openai
from typing import Dict, List, Set
# import torch
import numpy as np
import classla


# uncomment to download the Slovenian model
# commented so that it does not run on import
# classla.download('sl')


def extract_named_entities(text: str, nlp, lemmatize=True, extended=True) -> Set[str]:
    doc = nlp(text)

    if extended:
        named_entities = extract_named_entities_with_hyphen(text, doc)
    else:
        named_entities = set()
    
    for sentence in doc.sentences:
        for entity in sentence.ents:
            if entity.type == 'LOC':
                if lemmatize:
                    # Lemmatize each token in the entity
                    lemmas = [word.words[0].lemma.lower() for word in entity.tokens]
                    lemmatized_entities = " ".join(lemmas).strip()
                    named_entities.add(lemmatized_entities)
                else:
                    ent_text = entity.text.lower().strip()
                    named_entities.add(ent_text)

    return named_entities


def extract_named_entities_with_hyphen(text: str, doc) -> Set[str]:
    original_tokens = text.lower().split()
    named_entities = set()
    named_entities_hyphen = set()

    for sent in doc.sentences:
        for ent in sent.ents:
            named_entities.add(ent.text.lower())

    for token in original_tokens:
        if '-' in token:
            parts = token.split('-')
            if parts[0] in named_entities:
                named_entities_hyphen.add(token)

    return named_entities_hyphen


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

    def named_entity_evaluation(self, generated: List[str], reference: List[str],
                                lang='sl', lemmatize=True, extended=True):
        """
        precision: the share of intersecting named entities in generated named entities
        recall: the share of intersecting named entities in reference named entities
        f1: combination of precision and recall
        """
        if lemmatize:
            nlp = classla.Pipeline(lang=lang, processors='tokenize,ner,pos,lemma', verbose=False)
        else:
            nlp = classla.Pipeline(lang=lang, processors='tokenize,ner', verbose=False)

        precision = []
        recall = []
        f1 = []
        for i in range(len(generated)):
            ne_g = extract_named_entities(generated[i], nlp, lemmatize=lemmatize, extended=extended)
            ne_r = extract_named_entities(reference[i], nlp, lemmatize=lemmatize, extended=extended)
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
