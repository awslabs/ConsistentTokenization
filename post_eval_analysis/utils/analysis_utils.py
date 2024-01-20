from audioop import avg
from genericpath import isfile
import json
import os
import csv

import numpy as np
from scipy.stats import bootstrap

from statistics import median

from utils.metric_utils import metric_max_over_ground_truths, f1_score, exact_match_score, evaluate
from utils.helper_methods import helper_group_instances_by_length, helper_load_ground_truth, helper_get_full_model_list, helper_load_aggregate_predictions, helper_load_predictions

base_dir = os.getenv("BASE_DIR")

def produce_aggregate_csv(base_dataset='squad'):
    '''
    Produce the aggregate csv file to display performance
    '''
    # models = ['bert-base-uncased', 'roberta-base', 'facebook/bart-base', 'facebook/bart-base-tokenizer', 'google/t5-v1_1-base', 'google/t5-v1_1-base-tokenizer', 'google/t5-v1_1-small', 'google/t5-v1_1-small-tokenizer']
    models = ['bert-base-uncased', 'roberta-base', 'facebook/bart-base', 'facebook/bart-base-tokenizer', 'google/t5-v1_1-base', 'google/t5-v1_1-base-tokenizer']
    # model_names = ['BERT-base', 'RoBERTa-base', 'BART-base', 'BART-base-tokenizer', 'T5v1.1-base', 'T5v1.1-base-tokenizer', 'T5v1.1-small', 'T5v1.1-small-tokenizer']
    model_names = ['BERT-base', 'RoBERTa-base', 'BART-base', 'BART-base-tokenizer', 'T5v1.1-base', 'T5v1.1-base-tokenizer']
    lr_wrt_model = {'bert': '3e-5', 'roberta': '1e-5', 't5': '1e-4', 'bart': '2e-5'}
    random_seeds = ['0', '42', '12345']
    datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    headers = ['Model', 'Number of Epochs', 'Random Seed', 'Learning Rate'] + datasets

    f = open(base_dir + '/post_eval_analysis/perf_outputs/perf_' + base_dataset + '.csv', 'w')

    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(headers)

    for model_idx, model in enumerate(models):
        for seed in random_seeds:
            # Find the corresponding learning rate
            lr = '0.0'
            for model_name in lr_wrt_model:
                if model_name in model:
                    lr = lr_wrt_model[model_name]
            # Initialize current row
            row = [model_names[model_idx], '10', seed, lr]

            for dataset in datasets:
                # If the corresponding setup exist, begin to write
                if isfile(base_dir + '/results/' + base_dataset + '/' + model + '-' + lr + '-' + seed + '-' + dataset + '/all_results.json'):
                    # Load the model perfomance on this dataset and append to row
                    curr_perf = json.load(open(base_dir + '/results/' + base_dataset + '/' + model + '-' + lr + '-' + seed + '-' + dataset + '/all_results.json'))
                    row.append(str(round(curr_perf['eval_exact_match'], 2)) + '/' + str(round(curr_perf['eval_f1'], 2)))
            if len(row) > 4:
                # When the corresponding path exists, write a row to the csv file
                writer.writerow(row)
        
    # close the file
    f.close()

def produce_metric_csv(base_dataset='squad', metric_name='F1', avg_for_random_seeds=True):
    '''
    Produce the aggregate csv file to display performance in terms of a single metric
    '''
    # models = ['bert-base-uncased', 'roberta-base', 'facebook/bart-base', 'facebook/bart-base-tokenizer', 'google/t5-v1_1-base', 'google/t5-v1_1-base-tokenizer', 'google/t5-v1_1-small', 'google/t5-v1_1-small-tokenizer']
    models = ['bert-base-uncased', 'roberta-base', 'facebook/bart-base', 'facebook/bart-base-tokenizer', 'google/t5-v1_1-base', 'google/t5-v1_1-base-tokenizer']
    # model_names = ['BERT-base', 'RoBERTa-base', 'BART-base', 'BART-base-tokenizer', 'T5v1.1-base', 'T5v1.1-base-tokenizer', 'T5v1.1-small', 'T5v1.1-small-tokenizer']
    model_names = ['BERT-base', 'RoBERTa-base', 'BART-base', 'BART-base-tokenizer', 'T5v1.1-base', 'T5v1.1-base-tokenizer']
    lr_wrt_model = {'bert': '3e-5', 'roberta': '1e-5', 't5': '1e-4', 'bart': '2e-5'}
    random_seeds = ['0', '42', '12345']
    datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    if not avg_for_random_seeds:
        headers = ['Model', 'Number of Epochs', 'Random Seed', 'Learning Rate'] + datasets
    else:
        headers = ['Model'] + datasets

    f = open(base_dir + '/post_eval_analysis/perf_outputs/perf_' + base_dataset + '_' + metric_name + '.csv', 'w')

    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(headers)

    for model_idx, model in enumerate(models):
        seed_rows = []
        for seed in random_seeds:
            # Find the corresponding learning rate
            lr = '0.0'
            for model_name in lr_wrt_model:
                if model_name in model:
                    lr = lr_wrt_model[model_name]
                # Initialize current row
                row = [model_names[model_idx], '10', seed, lr]

                for dataset in datasets:
                    # If the corresponding setup exist, begin to write
                    if isfile(base_dir + '/results/' + base_dataset + '/' + model + '-' + lr + '-' + seed + '-' + dataset + '/all_results.json'):    
                        # Load the model perfomance on this dataset and append to row
                        curr_perf = json.load(open(base_dir + '/results/' + base_dataset + '/' + model + '-' + lr + '-' + seed + '-' + dataset + '/all_results.json'))
                        if metric_name == 'F1':
                            row.append(str(round(curr_perf['eval_f1'], 2)))
                        else:
                            row.append(str(round(curr_perf['eval_exact_match'], 2)))
                if not avg_for_random_seeds and len(row) > 4:
                    # write a row to the csv file
                    writer.writerow(row)
                elif len(row) > 4:
                    seed_rows.append(row[4:])
        if avg_for_random_seeds and len(seed_rows) > 0:
            # Average across seed_rows
            row = np.average(np.array([[float(x) for x in rows] for rows in seed_rows]), axis=0).tolist()
            row.insert(0, model_names[model_idx])
            writer.writerow(row)
        
    # close the file
    f.close()

def output_metric_for_single_model(base_dataset='squad', metric_name='F1', model_name='google/t5-v1_1-base', lr='1e-4', random_seed='42'):
    '''
    Return the ID and OOD performance for a given model
    '''
    datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    perf_dict = {}

    for dataset in datasets:    
        # Load the model perfomance on this dataset and append to row
        curr_perf = json.load(open(base_dir + '/results/' + base_dataset + '/' + model_name + '-' + lr + '-' + random_seed + '-' + dataset + '/all_results.json'))
        if metric_name == 'F1':
            perf_dict[dataset] = str(round(curr_perf['eval_f1'], 2))
        else:
            perf_dict[dataset] = str(round(curr_perf['eval_exact_match'], 2))
    return perf_dict


def perf_wrt_seq_len(dataset, model_list, num_buckets, metric_fn, base_dataset='squad', sequence_to_look='answers'):
    '''
    Return the performance with respect to answer length
    Returns: Tuple (a, b), where a = dict() where the keys are model names and the values are the performance w.r.t b;
        b = list() of sequence lengths
    '''
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    model_list = helper_get_full_model_list(model_list)

    # Load model predictions
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.append(json.load(file))
    
    gold_instances, ins_lens, ins_len_counts = helper_group_instances_by_length(raw_datasets, sequence_to_look, num_buckets=num_buckets)

    perfs = {model_name : [0] * len(gold_instances.keys()) for model_name in model_list}
    for model_idx in range(len(model_list)):
        # For each answer length, evaluate the model
        for key_idx, ans_length in enumerate(sorted(list(gold_instances.keys()))):
            overal_metric = 0.0
            for instance in gold_instances[ans_length]:
                overal_metric += metric_max_over_ground_truths(metric_fn, preds[model_idx][instance['id']], instance['answers']['text'])
            overal_metric = overal_metric * 100 / len(gold_instances[ans_length])
            perfs[model_list[model_idx]][key_idx] = overal_metric

    return perfs, ins_lens, ins_len_counts, model_list


def prediction_agreement_common_mistakes(dataset, model_list, metric_fn, base_dataset='squad'):
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    # Load model predictions
    model_list = helper_get_full_model_list(model_list)
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.append(json.load(file))
    
    # Get the samples that most models do wrong in
    common_mistaken_samples = []
    for instance_idx in range(len(raw_datasets['test'])):
        amount_of_correct = 0
        for model_pred in preds:
            if metric_max_over_ground_truths(exact_match_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text']) == 1.0:
                amount_of_correct += 1
        if amount_of_correct <= len(model_list)/2:
            common_mistaken_samples.append(raw_datasets['test'][instance_idx])


    conf_matrix = np.zeros([len(model_list), len(model_list)])
    for model1_idx, model1 in enumerate(model_list):
        for model2_idx, model2 in enumerate(model_list):
            # Compute the pairwise prediction agreement (F1 or EM)
            overal_metric = 0.0
            for instance in common_mistaken_samples:
                id = instance['id']
                overal_metric += metric_fn(preds[model1_idx][id], preds[model2_idx][id])
            overal_metric = overal_metric * 100 / len(preds[model1_idx].keys())
            conf_matrix[model1_idx][model2_idx] = overal_metric
    return conf_matrix, model_list


def common_mistakes_by_groups(dataset, model_list, group1, group2, metric_fn, threshold = 0.5, base_dataset='squad', error_allowance=1e-3):
    """
    Compute the common mistake by groups, save to two files group1_mistakes.csv, group2_mistakes.csv
    in which group1_mistakes contains instances that P(group1 model do wrong) >= threshold && P(group2 model do wrong) <= 1-threshold
    and group2_mistakes contains instances that P(group1 model do wrong) <= 1-threshold && P(group2 model do wrong) >= threshold
    """
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    # Load model predictions
    model_list = helper_get_full_model_list(model_list)
    group1 = helper_get_full_model_list(group1)
    group2 = helper_get_full_model_list(group2)
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.append(json.load(file))
    
    # Get the samples that most models do wrong in
    group1_mistaken_samples = []
    group2_mistaken_samples = []
    for instance_idx in range(len(raw_datasets['test'])):
        instance = raw_datasets['test'][instance_idx]
        group1_score = 0
        group2_score = 0
        for pred_idx, model_pred in enumerate(preds):
            if model_list[pred_idx] in group1:
                group1_score += metric_max_over_ground_truths(metric_fn, 
                        model_pred[instance['id']], 
                        instance['answers']['text'])
            else:
                group2_score += metric_max_over_ground_truths(metric_fn, 
                        model_pred[instance['id']], 
                        instance['answers']['text'])
            # Insert the model_prediction column
            instance[model_list[pred_idx] + '_preds'] = model_pred[instance['id']]

        if float(group1_score / len(group1)) <= (1-threshold + error_allowance) and float(group2_score / len(group2)) >= (threshold - error_allowance):
            group1_mistaken_samples.append(instance)
        if float(group1_score / len(group1)) >= (threshold - error_allowance) and float(group2_score / len(group2)) <= (1-threshold + error_allowance):
            group2_mistaken_samples.append(instance)

    if len(group1_mistaken_samples) != 0:
        with open('output/group1_mistakes_' + dataset + '_' + str(threshold) + '.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(group1_mistaken_samples[0].keys()))
            writer.writeheader()
            for data in group1_mistaken_samples:
                writer.writerow(data)

    if len(group2_mistaken_samples) != 0:
        with open('output/group2_mistakes_' + dataset + '_' + str(threshold) + '.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(group2_mistaken_samples[0].keys()))
            writer.writeheader()
            for data in group2_mistaken_samples:
                writer.writerow(data)

    return group1_mistaken_samples, group2_mistaken_samples


def prediction_agreement(dataset, model_list, metric_fn, base_dataset='squad'):
    # Load model predictions
    model_list = helper_get_full_model_list(model_list)
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.append(json.load(file))

    conf_matrix = np.zeros([len(model_list), len(model_list)])
    for model1_idx, model1 in enumerate(model_list):
        for model2_idx, model2 in enumerate(model_list):
            # Compute the pairwise prediction agreement (F1 or EM)
            overal_metric = 0.0
            for id in preds[model1_idx].keys():
                overal_metric += metric_fn(preds[model1_idx][id], preds[model2_idx][id])
            overal_metric = overal_metric * 100 / len(preds[model1_idx].keys())
            conf_matrix[model1_idx][model2_idx] = overal_metric
    return conf_matrix, model_list


def bootstrap_eval(dataset, model_setup, base_dataset='squad'):
    '''
    Return the bootstrap
    SciPy Bootstrap doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    '''
    raw_datasets = helper_load_ground_truth(dataset)

    # Load predictions
    file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
    predictions = json.load(file)

    # Compute their corresponding F1 for each predicted sample
    single_sample_f1s = np.zeros(len(raw_datasets['test']))
    ground_truths = []
    for idx in range(len(raw_datasets['test'])):
        ground_truths = raw_datasets['test'][idx]['answers']['text']
        single_sample_f1s[idx] = metric_max_over_ground_truths(f1_score, predictions[raw_datasets['test'][idx]['id']], ground_truths)

    # boostrap
    result = bootstrap((single_sample_f1s * 100,), np.average, confidence_level=0.95, method='percentile')
    print(result)
    print('Mid value is ', (result.confidence_interval.high - result.confidence_interval.low) / 2 + result.confidence_interval.low)
    return result


def performance_for_each_instance(dataset, model_list, group1, group2, metric_fn, base_dataset='squad'):
    '''
    Params:
    dataset (str): Specifies a dataset name
    model_list (list of str): contains a list of model names
    group1 (list of str): contains a subset of model names
    group1 (list of str): model_list - group1
    Returns:
    Return list 1 [length of datasets], values to be the average metric of group 1 model on that instance
    Return list 1 [length of datasets], values to be the average metric of group 1 model on that instance
    '''
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset)
    # Load model predictions
    model_list = helper_get_full_model_list(model_list)
    group1 = helper_get_full_model_list(group1)
    group2 = helper_get_full_model_list(group2)
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.append(json.load(file))
    
    # Get the performance for each sample
    group1_perf = []
    group2_perf = []
    for instance_idx in range(len(raw_datasets['test'])):
        instance = raw_datasets['test'][instance_idx]
        group1_score = 0
        group2_score = 0
        for pred_idx, model_pred in enumerate(preds):
            if model_list[pred_idx] in group1:
                group1_score += metric_max_over_ground_truths(metric_fn, 
                        model_pred[instance['id']], 
                        instance['answers']['text'])
            else:
                group2_score += metric_max_over_ground_truths(metric_fn, 
                        model_pred[instance['id']], 
                        instance['answers']['text'])

        group1_perf.append(float(group1_score / len(group1)))
        group2_perf.append(float(group2_score / len(group2)))
    
    return group1_perf, group2_perf


def qc_overlap_vs_perf(dataset, model_list, num_buckets, metric_fn, base_dataset='squad', n_gram=1):
    """
    Returns Tuple (a, b), where a = dict() where the keys are model names and the values are the perf metric w.r.t b;
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
    
    perfs = {model_name : [0] * len(gold_instances.keys()) for model_name in model_list}

    for model_idx in range(len(model_list)):
        # For each answer length, evaluate the model
        for key_idx, seq_length in enumerate(sorted(list(gold_instances.keys()))):
            perf_value = 0.0
            for instance in gold_instances[seq_length]:
                perf_value += metric_max_over_ground_truths(metric_fn, 
                        preds[model_idx][instance['id']], 
                        instance['answers']['text'])
            perf_value = perf_value / len(gold_instances[seq_length])
            perfs[model_list[model_idx]][key_idx] = perf_value

    return perfs, ins_lens, ins_len_counts, model_list


def perf_wrt_group_type(model_list, group1, group2, grouping_type='strategy', base_dataset='squad'):
    '''
    Return the performance with respect to answer length
    Returns: Tuple (a, b), where a = dict() where the keys are "group1"/"group2" and the values are the performance w.r.t b;
        b = list() of dataset creation strategies
    '''
    if grouping_type == 'strategy':
        # Mapping from strategy to datasets
        type_to_ds = {'crowdsource': 'squad NaturalQuestions NewsQA duorc', 'domain expert': 'bioasq TextbookQA', 
            'auto': 'SearchQA', 'mixed': 'TriviaQA'}
    else:
        # Mapping from Domain to datasets
        type_to_ds = {'Wiki': 'squad NaturalQuestions', 'News': 'NewsQA', 
            'Web': 'TriviaQA SearchQA', 'bio': 'bioasq', 'Textbook': 'TextbookQA', 'Movie': 'duorc'}
        
    # Load model predictions
    model_list = helper_get_full_model_list(model_list)
    group1 = helper_get_full_model_list(group1)
    group2 = helper_get_full_model_list(group2)

    type_v_f1 = np.zeros([2, len(type_to_ds.keys())])
    type_v_em = np.zeros([2, len(type_to_ds.keys())])
    for type_idx, type in enumerate(list(type_to_ds.keys())):
        # Load ground truth data
        ds_string = type_to_ds[type].replace(base_dataset, '')
        raw_datasets = helper_load_ground_truth(ds_string)
        # Evaluate on both group 1 and group 2
        group1_f1 = 0.0
        group1_em = 0.0
        for model_idx, model in enumerate(group1):
            preds = helper_load_predictions(model, ds_string, base_dataset)
            group1_f1 += evaluate(raw_datasets['test'], preds)['f1']
            group1_em += evaluate(raw_datasets['test'], preds)['exact_match']
        type_v_f1[0][type_idx] = group1_f1 / len(group1)
        type_v_em[0][type_idx] = group1_em / len(group1)

        group2_f1 = 0.0
        group2_em = 0.0
        for model_idx, model in enumerate(group2):
            preds = helper_load_predictions(model, ds_string, base_dataset)
            group2_f1 += evaluate(raw_datasets['test'], preds)['f1']
            group2_em += evaluate(raw_datasets['test'], preds)['exact_match']
        type_v_f1[1][type_idx] = group2_f1 / len(group2)
        type_v_em[1][type_idx] = group2_em / len(group2)

    return type_v_f1, type_v_em, group1, group2, list(type_to_ds.keys())


def load_training_curve_info(model_name, base_dataset):
    """
    Returns dictionary where the key is the random seed, values to be correspoinding
        metrics.
        losses dict([list])
        f1s dict([list])
        ems dict([list])
        epochs [list]
        losses_epochs [list]
    """
    model_list = helper_get_full_model_list([model_name])
    seeds = ['0', '42', '12345']

    losses = dict()
    f1s = dict()
    ems = dict()
    
    for model in model_list:
        epochs = []
        loss_epochs = []
        # Find corresponding random seed
        for seed in seeds:
            if '-' + seed in model:
                losses[seed] = []
                f1s[seed] = []
                ems[seed] = []
                break
        # Load the given model's trainer state
        trainer_state = json.load(open(base_dir + '/trained_models/' + base_dataset + '/' + model + '/trainer_state.json'))

        for metrics in trainer_state['log_history']:
            if 'eval_f1' in metrics:
                epochs.append(metrics['step'])
                f1s[seed].append(metrics['eval_f1'])
                ems[seed].append(metrics['eval_exact_match'])
                
            if 'loss' in metrics:
                loss_epochs.append(metrics['step'])
                losses[seed].append(metrics['loss'])
    return losses, f1s, ems, loss_epochs, epochs