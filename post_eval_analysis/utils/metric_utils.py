##### Functions and Helper functions to compute F1/EM for a single instance
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qc_overlap(question, context, n_gram=1):
    """
    Compute the question context overlap rate (recall)
    n_gram: The n-grams to compute the recall
    """
    prediction_token_list = normalize_answer(context).split()
    ground_truth_token_list = normalize_answer(question).split()
    
    pred_n_grams = zip(*[prediction_token_list[i:] for i in range(n_gram)])
    ground_truth_n_grams = zip(*[ground_truth_token_list[i:] for i in range(n_gram)])
    prediction_tokens = [" ".join(ngram) for ngram in pred_n_grams]
    ground_truth_tokens = [" ".join(ngram) for ngram in ground_truth_n_grams]
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for instance_idx in range(len(dataset)):
        qa = dataset[instance_idx]
        
        if qa["id"] not in predictions:
            continue
        else:
            total += 1
        ground_truths = qa['answers']['text']
        prediction = predictions[qa["id"]]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}