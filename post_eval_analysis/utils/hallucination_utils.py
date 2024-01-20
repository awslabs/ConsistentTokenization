##### Hallucination Analysis ######
import json
import csv
import numpy as np
import pdb

from utils.metric_utils import metric_max_over_ground_truths, f1_score, exact_match_score, evaluate, normalize_answer
from utils.helper_methods import helper_load_ground_truth, helper_get_full_model_list, helper_group_instances_by_length, helper_load_aggregate_predictions


base_dir = os.getenv("BASE_DIR")

def domain_vs_hallucination(model_list, base_dataset='squad', metric_name='F1'):
    '''
    Return (a, b, domain_list) where a is a list of list, where the first layer of list represents each domain, and the inner lists represent different
        generative model hallucination rate
        and b is a list of list, where the first layer of list represents each domain, and the inner lists represent different
        generative model performance
    '''
    datasets = [ "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "squad","TextbookQA"]
    full_model_list = helper_get_full_model_list(model_list)
    domain_v_hallu = np.zeros([len(full_model_list), len(datasets)])
    domain_v_perf = np.zeros([len(full_model_list), len(datasets)])

    for domain_idx, domain in enumerate(datasets):
        # Load ground truth data
        raw_datasets = helper_load_ground_truth(domain)
        for model_idx, model in enumerate(full_model_list):
            ## Load model prediction
            pred_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/eval_predictions.json')
            predictions = json.load(pred_file)
            hallu_rate = 0.0
            # Compute its hallucination rate
            for instance_idx in range(len(raw_datasets['test'])):
                instance = raw_datasets['test'][instance_idx]
                if normalize_answer(predictions[instance['id']]) not in normalize_answer(instance['context']):
                    hallu_rate += 1
            hallu_rate /= len(raw_datasets['test'])
            domain_v_hallu[model_idx][domain_idx] = hallu_rate
            ## Retrieve its performance on this domain
            perf_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/all_results.json')
            results = json.load(perf_file)
            domain_v_perf[model_idx][domain_idx] = results['eval_exact_match'] if metric_name == 'EM' else results['eval_f1']
            
            pred_file.close()
            perf_file.close()
    return domain_v_hallu, domain_v_perf, datasets, full_model_list

def eval_without_hallucination(model_list, output_hallucinations=False, base_dataset='squad'):
    '''
    Remove the hallucnated samples and reevaluate the models
    Save hallucinated samples to a file if output_hallucinations=True
    Returns
        list [number of model * domains]: The values are performance difference after removing.
        list [number of model * domains]: The values are performance ratio change (new - orig)/orig after removing.
        dataset
        full_model_list
    '''
    full_model_list = helper_get_full_model_list(model_list)
    datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    f1_change = np.zeros([len(full_model_list), len(datasets)])
    f1_change_ratio = np.zeros([len(full_model_list), len(datasets)])

    em_change = np.zeros([len(full_model_list), len(datasets)])
    em_change_ratio = np.zeros([len(full_model_list), len(datasets)])

    hallucinated_samples = []

    for domain_idx, domain in enumerate(datasets):
        hallucinated_ids = set()
        # Load ground truth data
        raw_datasets = helper_load_ground_truth(domain)
        for model_idx, model in enumerate(full_model_list):
            if 't5' in model or 'bart' in model:
                ## Load model prediction
                pred_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/eval_predictions.json')
                predictions = json.load(pred_file)
                # Record the id of hallucinated samples
                for instance_idx in range(len(raw_datasets['test'])):
                    instance = raw_datasets['test'][instance_idx]
                    if normalize_answer(predictions[instance['id']]) not in normalize_answer(instance['context']):
                        hallucinated_ids.add(instance['id'])
                        # Record the instance if need to output to a file
                        if output_hallucinations:
                            instance['prediction'] = predictions[instance['id']]
                            if 'hallucinated model' not in instance.keys():
                                instance['hallucinated model'] = [model]
                            else:
                                instance['hallucinated model'].append(model)
                            hallucinated_samples.append(instance)
        
        if output_hallucinations:
            # Output to csv
            with open('output/hallucinated_samples.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(hallucinated_samples[0].keys()))
                writer.writeheader()
                for data in hallucinated_samples:
                    writer.writerow(data)
                
        ### Evaluate models on the unhallucinated samples
        for model_idx, model in enumerate(full_model_list):
            ## Load model prediction
            pred_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/eval_predictions.json')
            predictions = json.load(pred_file)

            # Remove the hallucinated ids
            for id in hallucinated_ids:
                predictions.pop(id, None)
            # Evaluate the model
            new_results = evaluate(raw_datasets['test'], predictions)
            # Compute the performance difference and ratio for change
            # Retrieve its orig performance on this domain
            orig_perf_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/all_results.json')
            orig_results = json.load(orig_perf_file)
        
            f1_change[model_idx][domain_idx] = new_results['f1'] - orig_results['eval_f1']
            f1_change_ratio[model_idx][domain_idx] = (new_results['f1'] - orig_results['eval_f1'])/orig_results['eval_f1']
            em_change[model_idx][domain_idx] = new_results['exact_match'] - orig_results['eval_exact_match']
            em_change_ratio[model_idx][domain_idx] = (new_results['exact_match'] - orig_results['eval_exact_match'])/orig_results['eval_exact_match']

    return f1_change, f1_change_ratio, em_change, em_change_ratio, datasets, full_model_list


def eval_on_hallucination(model_list, output_hallucinations=False, base_dataset='squad'):
    '''
    Evaluate the models on datasets on hallucinated samples
    Save hallucinated samples to a file if output_hallucinations=True
    Returns
        list [number of model * domains]: The values are performance on hallucinated samples only.
        list [number of model * domains]: The values are performance ratio change (new - orig)/orig after removing.
        dataset
        full_model_list
    '''
    full_model_list = helper_get_full_model_list(model_list)
    datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    f1_hallu = np.zeros([len(full_model_list), len(datasets)])
    f1_unhallu = np.zeros([len(full_model_list), len(datasets)])

    em_hallu = np.zeros([len(full_model_list), len(datasets)])
    em_unhallu = np.zeros([len(full_model_list), len(datasets)])

    hallucinated_samples = []

    for domain_idx, domain in enumerate(datasets):
        hallucinated_ids = set()
        # Load ground truth data
        raw_datasets = helper_load_ground_truth(domain)
        for model_idx, model in enumerate(full_model_list):
            if 't5' in model or 'bart' in model:
                ## Load model prediction
                pred_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/eval_predictions.json')
                predictions = json.load(pred_file)
                # Record the id of hallucinated samples
                for instance_idx in range(len(raw_datasets['test'])):
                    instance = raw_datasets['test'][instance_idx]
                    if normalize_answer(predictions[instance['id']]) not in normalize_answer(instance['context']):
                        hallucinated_ids.add(instance['id'])
                        # Record the instance if need to output to a file
                        if output_hallucinations:
                            instance['prediction'] = predictions[instance['id']]
                            if 'hallucinated model' not in instance.keys():
                                instance['hallucinated model'] = [model]
                            else:
                                instance['hallucinated model'].append(model)
                            hallucinated_samples.append(instance)
        unhallu_dataset = raw_datasets.filter(lambda example: example['id'] not in hallucinated_ids)
        hallu_dataset = raw_datasets.filter(lambda example: example['id'] in hallucinated_ids)

        if output_hallucinations:
            # Output to csv
            with open('output/hallucinated_samples.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(hallucinated_samples[0].keys()))
                writer.writeheader()
                for data in hallucinated_samples:
                    writer.writerow(data)

        ### Evaluate models on the hallucinated and unhallucinated samples
        for model_idx, model in enumerate(full_model_list):
            ## Load model prediction
            pred_file = open(base_dir + '/results/' + base_dataset + '/' + model + '-' + domain + '/eval_predictions.json')
            predictions = json.load(pred_file)

            # Evaluate the model
            hallu_results = evaluate(hallu_dataset['test'], predictions)
            unhallu_results = evaluate(unhallu_dataset['test'], predictions)

            f1_hallu[model_idx][domain_idx] = hallu_results['f1']
            f1_unhallu[model_idx][domain_idx] = unhallu_results['f1']
            em_hallu[model_idx][domain_idx] = hallu_results['exact_match']
            em_unhallu[model_idx][domain_idx] = unhallu_results['exact_match']

    return f1_hallu, f1_unhallu, em_hallu, em_unhallu, full_model_list, datasets

def seq_len_vs_hallucination(dataset, model_list, num_buckets, sequence_to_look='answers', base_dataset='squad'):
    """
    Returns Tuple (a, b), where a = dict() where the keys are model names and the values are the hallucination rate w.r.t b;
        b = list() of sequence lengths
    """
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    model_list = helper_get_full_model_list(model_list)

    # Load model predictions
    preds = []
    for model_setup in model_list:
        if dataset != 'aggregate':
            file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
            preds.append(json.load(file))
        else:
            preds.append(helper_load_aggregate_predictions(model_setup, base_dataset))
    
    # Get the groupped instances
    gold_instances, ins_lens, ins_len_counts = helper_group_instances_by_length(raw_datasets, sequence_to_look, num_buckets)
    
    hallucination_rates = {model_name : [0] * len(gold_instances.keys()) for model_name in model_list}

    for model_idx in range(len(model_list)):
        # For each answer length, evaluate the model
        for key_idx, seq_length in enumerate(sorted(list(gold_instances.keys()))):
            hallu_rate = 0.0
            for instance in gold_instances[seq_length]:
                if normalize_answer(preds[model_idx][instance['id']]) not in normalize_answer(instance['context']):
                    hallu_rate += 1
            hallu_rate = hallu_rate / len(gold_instances[seq_length])
            hallucination_rates[model_list[model_idx]][key_idx] = hallu_rate

    return hallucination_rates, ins_lens, ins_len_counts, model_list


def qc_overlap_vs_hallucination(dataset, model_list, num_buckets, base_dataset='squad', n_gram=1):
    """
    Returns Tuple (a, b), where a = dict() where the keys are model names and the values are the hallucination rate w.r.t b;
        b = list() of sequence lengths
    """
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    model_list = helper_get_full_model_list(model_list)

    # Load model predictions
    preds = []
    for model_setup in model_list:
        if dataset != 'aggregate':
            file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
            preds.append(json.load(file))
        else:
            preds.append(helper_load_aggregate_predictions(model_setup, base_dataset))
    
    # Get the groupped instances
    gold_instances, ins_lens, ins_len_counts = helper_group_instances_by_length(raw_datasets, 'qc-overlap', num_buckets, n_gram=n_gram)
    
    hallucination_rates = {model_name : [0] * len(gold_instances.keys()) for model_name in model_list}

    for model_idx in range(len(model_list)):
        # For each answer length, evaluate the model
        for key_idx, seq_length in enumerate(sorted(list(gold_instances.keys()))):
            hallu_rate = 0.0
            for instance in gold_instances[seq_length]:
                if normalize_answer(preds[model_idx][instance['id']]) not in normalize_answer(instance['context']):
                    hallu_rate += 1
            hallu_rate = hallu_rate / len(gold_instances[seq_length])
            hallucination_rates[model_list[model_idx]][key_idx] = hallu_rate

    return hallucination_rates, ins_lens, ins_len_counts, model_list