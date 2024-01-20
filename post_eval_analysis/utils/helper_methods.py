import re
import os
import json
import numpy as np

from statistics import median
from datasets import concatenate_datasets, load_dataset, DatasetDict
from ast import literal_eval
from .metric_utils import qc_overlap
import pdb

base_dir = os.getenv("BASE_DIR")
full_dataset_list = ['bioasq', 'duorc', 'NaturalQuestions', 'NewsQA', 'SearchQA', 'squad', 'TextbookQA', 'TriviaQA']

def helper_load_single_dataset_ground_truth(dataset):
    split = 'test' if os.path.exists(base_dir + "/data/" + dataset + "/test.csv") else 'validation'
    # Load ground truth data
    raw_datasets = load_dataset('csv', data_files={'test': base_dir + '/data/' + dataset + '/' + split + '.csv'})
    def castVals(example):
        """
        Cast the answers into a dictonary due to python csv loader issue
        we need to manually eval the dictonary values
        """
        example['answers'] = re.sub('array\(', '', example['answers'])
        example['answers'] = re.sub("\), 'text'", ", 'text'", example['answers']) 
        example['answers'] = re.sub(', dtype=[A-Za-z0-9]*\)', '', example['answers'])
        example['answers'] = re.sub('dtype=[A-Za-z0-9]*\)', '', example['answers'])
        # This newline character can happens for SQuAD
        example['answers'] = re.sub('\n      , ', '', example['answers'])
        example['answers'] = literal_eval(example['answers'])
        return example
    raw_datasets = raw_datasets.map(castVals)
    return raw_datasets

def helper_load_ground_truth(dataset):
    '''
    Load the ground truth datsets
    if dataset == a datata set name, return that dataset
    if dataset == a string of datasets separate by space, return the concat of these datasets
    if dataset == 'aggregate', return the concat of all datasets
    '''
    if dataset != 'aggregate' and dataset in full_dataset_list:
        return helper_load_single_dataset_ground_truth(dataset)
    # Load a list of datasets
    ds_list = []

    if dataset == 'aggregate':
        dataset_list = full_dataset_list
    else:
        dataset_list = dataset.split()

    for dataset_name in dataset_list:
        ds_list.append(helper_load_single_dataset_ground_truth(dataset_name)['test'])
    return DatasetDict({'test': concatenate_datasets(ds_list)})

def helper_load_aggregate_predictions(model_setup, base_dataset):
    preds = {}
    for dataset in full_dataset_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        preds.update(json.load(file))
    return preds

def helper_load_predictions(model_setup, dataset, base_dataset):
    if dataset in full_dataset_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset + '/eval_predictions.json')
        return json.load(file)
    if dataset == 'aggregate':
        return helper_load_aggregate_predictions(model_setup, base_dataset)
    
    # Load a list of predictions and concat them
    preds = {}
    for ds in dataset.split():
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + ds + '/eval_predictions.json')
        preds.update(json.load(file))
    return preds


def helper_get_full_model_list(model_list):
    lr_wrt_model = {'bert': '3e-5', 'roberta': '1e-5', 't5': '1e-4', 'bart': '2e-5'}
    random_seeds = ['0', '42', '12345']
    full_model_list = []
    for model in model_list:
        # Find the corresponding lrs
        lr = '0.0'
        for model_name in lr_wrt_model:
            if model_name in model:
                lr = lr_wrt_model[model_name]
        for seed in random_seeds:
            full_model_list.append(model + '-' + lr + '-' + seed)
    return full_model_list

def helper_group_instances_by_length(raw_datasets, sequence_to_look, num_buckets, n_gram=1):
    """
    Optional input: n_gram: This is only used for compute QC-overlap
    """
    # Get the instance with maximum length
    all_lens = []
    for instance_idx in range(len(raw_datasets['test'])):
        if sequence_to_look == 'answers':
            ans_lens = []
            # Compute median gold answer length
            for ans_text in raw_datasets['test'][instance_idx]['answers']['text']:
                ans_lens.append(len(ans_text.split()))
            median_len = median(ans_lens)
        elif sequence_to_look == 'context' or sequence_to_look == 'question':
            median_len = len(raw_datasets['test'][instance_idx][sequence_to_look].split())
        else:
            # Compute question context overlap
            median_len = qc_overlap(raw_datasets['test'][instance_idx]['question'], raw_datasets['test'][instance_idx]['context'], n_gram=n_gram)
        all_lens.append(median_len)
    
    all_lens = np.array(all_lens)

    # Find the first threshold
    ans_len_thresholds = [0, np.percentile(all_lens, 100.0/num_buckets)]
    for tile in range(2, num_buckets + 1):
        curr_limit = np.percentile(all_lens, 100.0/num_buckets * tile)
        if curr_limit != ans_len_thresholds[-1]:
            break
    
    for new_tile in range(0, num_buckets):
        ans_len_thresholds.append(np.percentile(all_lens, 100.0/num_buckets * tile + (100 - 100.0/num_buckets * tile) / (num_buckets - 1) * new_tile))
    
    ans_len_counts = [0] * len(ans_len_thresholds)
    gold_instances = {key : [] for key in ans_len_thresholds[1:]}
    # Constrct a dictionary with key being sequence lengths and values be the instances whose sequence length
    #   is within that length
    for instance_idx in range(len(raw_datasets['test'])):
        if sequence_to_look == 'answers':
            ans_lens = []
            # Compute median gold answer length
            for ans_text in raw_datasets['test'][instance_idx]['answers']['text']:
                ans_lens.append(len(ans_text.split()))
            median_len = median(ans_lens)
        elif sequence_to_look == 'context' or sequence_to_look == 'question':
            median_len = len(raw_datasets['test'][instance_idx][sequence_to_look].split())
        else:
            # Compute question context overlap
            median_len = qc_overlap(raw_datasets['test'][instance_idx]['question'], raw_datasets['test'][instance_idx]['context'], n_gram=n_gram)

        for idx in range(1, len(ans_len_thresholds)):
            # Find the length range the instance belongs to
            if (median_len > ans_len_thresholds[idx-1] or ans_len_thresholds[idx-1] == 0) and median_len <= ans_len_thresholds[idx]:
                gold_instances[ans_len_thresholds[idx]].append(raw_datasets['test'][instance_idx])
                ans_len_counts[idx] += 1
                break
    
    return gold_instances, sorted(list(gold_instances.keys())), ans_len_counts

def helper_read_from_txt_list(path):
    # empty list to read list from a file
    ret = []

    # open file and read the content in a list
    with open(path, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            if line[-1] == '\n':
                x = line[:-1]
            else:
                x = line

            # add current item to the list
            ret.append(x)
    return ret

def helper_generate_file_prefix_conf(short_model_name, model_name_or_path, lr, seed, use_old_tokenization, dataset):
    prefix = base_dir + '/post_eval_analysis/output/prob_and_perp/' + short_model_name
    if '-tokenizer' in model_name_or_path:
        prefix += '-tokenizer'

    prefix += '-' + str(lr) + '-' + str(seed)

    if use_old_tokenization:
        prefix += '_old'
    else:
        prefix += '_new'
    prefix += '_' + dataset
    return prefix

def helper_get_all_file_prefix_conf(model_name, dataset):
    if 't5' in model_name:
        lr = 1e-04
    else:
        lr = 2e-05
    
    prefices, fixed_prefices, unfixed_new_prefices, unfixed_old_prefices = [], [], [], []
    
    for seed in ['0', '42', '12345']:
        for use_old_tokenization in [True, False]:
            for path in ['-tokenizer', '']:
                curr_prefix = helper_generate_file_prefix_conf(model_name, path, str(lr), seed, use_old_tokenization, dataset)
                if path == '':
                    prefices.append(curr_prefix)
                    if use_old_tokenization:
                        unfixed_old_prefices.append(curr_prefix)
                    else:
                        unfixed_new_prefices.append(curr_prefix)
                elif not use_old_tokenization:
                    prefices.append(curr_prefix)
                    fixed_prefices.append(curr_prefix)

    return prefices, fixed_prefices, unfixed_new_prefices, unfixed_old_prefices