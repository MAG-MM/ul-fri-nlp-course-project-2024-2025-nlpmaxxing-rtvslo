from bert_score import BERTScorer
from typing import List, Set
import numpy as np
import classla
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
from dotenv import load_dotenv
import os


# uncomment to download the Slovenian model
# commented so that it does not run on import
# classla.download('sl')


def extract_named_entities(text: str, nlp, lemmatize=True, extended=True) -> Set[str] | None:
    try:
        doc = nlp(text)
    except IndexError:
        print("error")
        return None

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
            if parts[0] in named_entities and parts[1] not in named_entities:
                named_entities_hyphen.add(token)

    return named_entities_hyphen


def split_report_paragraphs(report: str) -> List[str]:
    """
    Splits the input text into paragraphs (by newline) and returns only those
    paragraphs that contain 4 or more words.
    """
    paragraphs = report.split('\n')
    filtered = [p for p in paragraphs if len(p.split()) >= 4]
    return filtered


def list_labse_similarity(model, text_list: list[str]) -> List[float]:
    embeddings_list = []
    for text in text_list:
        e = model.encode(text, convert_to_tensor=True)
        embeddings_list.append(e)
    return embeddings_list


def labse_similarity(model, text_1: str, text_2: str) -> float:
    embedding_1 = model.encode(text_1, convert_to_tensor=True)
    embedding_2 = model.encode(text_2, convert_to_tensor=True)
    similarity = cos_sim(embedding_1, embedding_2)
    return similarity.item()


def gpt_compare_reports(reference_report: str, generated_report: str):
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    system_message = (
        "You are a traffic report evaluator. "
        "Given a reference report and a generated report in Slovenian language, "
        "respond with two integers only: first a semantic similarity score (0-3), "
        "then a readability score (0-3). "
        "No explanation, just the two numbers separated by a comma. "
        "Semantic similarity of 3 is the best, meaning full similarity, including location names, while 0 means no or minimal similarity. "
        "Readability score of 3 is the best, meaning well readable and understandable for humans without any spelling or grammatical mistakes, "
        "while score of 0 means some unreadable nonsense."
    )
    user_message = f"Reference report:\n{reference_report}\n\nGenerated report:\n{generated_report}"
    
    response = client.responses.create(
        model="gpt-4",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_message,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message,
                    }
                ]
            }
        ],
        temperature=0.0,
    )
    
    result = response.output_text
    similarity, readability = map(int, result.strip().split(","))
    return similarity, readability


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

    def labse_scores(self, generated: List[str], reference: List[str]):
        """
        Compute LaBSE embedding similarities for two lists of texts.
        """
        model = SentenceTransformer('sentence-transformers/LaBSE')
        scores = []
        for i in range(len(generated)):
            similarity = labse_similarity(model, generated[i], reference[i])
            scores.append(similarity)
        return scores
    
    def labse_paragraph_scores(self, generated: List[str], reference: List[str]):
        """
        Find the and match the most similar paragraphs in generated per LaBSE, then return their average.
        """
        model = SentenceTransformer('sentence-transformers/LaBSE')
        scores = []
        for i in range(len(generated)):
            generated_split = split_report_paragraphs(generated[i])
            reference_split = split_report_paragraphs(reference[i])
            if len(reference_split) <= 0:
                scores.append(None)
                continue
            
            generated_split_embeddings = list_labse_similarity(model, generated_split)
            reference_split_embeddings = list_labse_similarity(model, reference_split)
            
            # match paragraphs
            closest_labse_matches = []
            closest_labse_scores = []
            for i, embedding_ref in enumerate(reference_split_embeddings):
                best_sim = -1
                best_gen_index = 0
                for j, embedding_gen in enumerate(generated_split_embeddings):
                    sim = cos_sim(embedding_gen, embedding_ref).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_gen_index = j
                closest_labse_matches.append(best_gen_index)
                closest_labse_scores.append(best_sim)
            
            # calculate
            average_paragraph_score = np.average(closest_labse_scores).tolist()
            scores.append(average_paragraph_score)
            
        return scores
    
    def length_diff(self, generated: List[str], reference: List[str], abs=False):
        """
        Compare lengths of generated reports.
        """
        lengths = []
        for i in range(len(generated)):
            lengths.append((len(generated[i].split()) - len(reference[i].split())) / max(len(reference[i].split()), 1))
        average_absolute_difference = np.mean(np.abs(lengths))
        if not abs:
            return average_absolute_difference, lengths
        else:
            return average_absolute_difference, np.abs(lengths).tolist()

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

            if not ne_r or not ne_g:
                precision.append(None)
                recall.append(None)
                f1.append(None)
                continue

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
        
    def chagpt_evaluation(self, generated: List[str], reference: List[str]):
        scores_semantic_similarity = []
        scores_readibility = []
        for i in range(len(generated)):
            sem, read = gpt_compare_reports(generated_report=generated[i], 
                                            reference_report=reference[i])
            scores_semantic_similarity.append(sem)
            scores_readibility.append(read)
        return scores_semantic_similarity, scores_readibility


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
