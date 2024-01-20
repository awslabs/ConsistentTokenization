import json
import re
import csv
import os.path
import pdb
import math

import numpy as np
from scipy.stats import bootstrap

from statistics import median

from utils.metric_utils import metric_max_over_ground_truths, f1_score, exact_match_score, evaluate, normalize_answer
from utils.helper_methods import helper_generate_file_prefix_conf, helper_get_all_file_prefix_conf, helper_group_instances_by_length, helper_load_ground_truth, helper_get_full_model_list, helper_load_aggregate_predictions, helper_load_predictions, helper_read_from_txt_list

base_dir = os.getenv("BASE_DIR")

def get_changed_samples(model_name, dataset_name, base_dataset):
    """
    Params
        model_name (str): The name that is the same as in huggingface
        dataset_name (str)
    Return a few specific examples to look at where its corresponding prediction changed after we applied the tokenization changed, save to csv file as well
    """
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset_name)
    model_list = helper_get_full_model_list([model_name, model_name + '-tokenizer'])
    unfixed_group =  helper_get_full_model_list([model_name])
    fixed_group = helper_get_full_model_list([model_name + '-tokenizer'])
    
    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset_name + '/eval_predictions.json')
        preds.append(json.load(file))

    changed_instances = []
    
    for instance_idx in range(len(raw_datasets['test'])):
        instance = raw_datasets['test'][instance_idx]
        unfixed_hallu_rate = 0
        fixed_hallu_rate = 0
        for pred_idx, model_pred in enumerate(preds):
            # Find the halluciated examples in unfixed version
            if model_list[pred_idx] in unfixed_group:
                unfixed_hallu_rate += float(normalize_answer(model_pred[instance['id']]) not in normalize_answer(instance['context']))
            else:
                fixed_hallu_rate += float(normalize_answer(model_pred[instance['id']]) not in normalize_answer(instance['context']))
            # Insert the model_prediction column
            instance[model_list[pred_idx] + '_preds'] = model_pred[instance['id']]
        
        # Record them to list if the corresponding predictions no longer hallucinate
        if float(unfixed_hallu_rate / len(unfixed_group)) >= 0.5 and float(fixed_hallu_rate / len(fixed_group)) == 0:
            changed_instances.append(instance)
    # Save such samples to csv
    if "t5" in model_name:
        short_model_name = 't5'
    else:
        short_model_name = 'bart'

    with open(base_dir + '/post_eval_analysis/output/hallu_changed_' + short_model_name + '.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(changed_instances[0].keys()))
        writer.writeheader()
        for data in changed_instances:
            writer.writerow(data)
    return changed_instances

def evaluate_on_changed_instances(model_name, dataset_name, base_dataset):
    """
    Evaluate the models on the changed instances, corresponding to fixing tokenization
    and not fix the tokenizations
    Returns a dictonary with fixed F1/EMs and unfixed F1/EMs
    """
    # Load ground truth data
    raw_datasets = helper_load_ground_truth(dataset_name)
    model_list = helper_get_full_model_list([model_name, model_name + '-tokenizer'])
    unfixed_group =  helper_get_full_model_list([model_name])
    fixed_group = helper_get_full_model_list([model_name + '-tokenizer'])
    overall_F1s = {'unfixed': 0.0, 'fixed': 0.0}
    overall_EMs = {'unfixed': 0.0, 'fixed': 0.0}
    # num_instance_improved_perf = 0

    preds = []
    for model_setup in model_list:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset_name + '/eval_predictions.json')
        preds.append(json.load(file))

    tot_changed_samples = 0
    for instance_idx in range(len(raw_datasets['test'])):
        instance = raw_datasets['test'][instance_idx]
        unfixed_hallu_rate = 0
        fixed_hallu_rate = 0
        f1s = {'unfixed': 0.0, 'fixed': 0.0}
        ems = {'unfixed': 0.0, 'fixed': 0.0}
        for pred_idx, model_pred in enumerate(preds):
            # Find the halluciated examples in unfixed version
            if model_list[pred_idx] in unfixed_group:
                unfixed_hallu_rate += float(normalize_answer(model_pred[instance['id']]) not in normalize_answer(instance['context']))
                f1s['unfixed'] += metric_max_over_ground_truths(f1_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text'])
                ems['unfixed'] += metric_max_over_ground_truths(exact_match_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text'])
            else:
                fixed_hallu_rate += float(normalize_answer(model_pred[instance['id']]) not in normalize_answer(instance['context']))
                f1s['fixed'] += metric_max_over_ground_truths(f1_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text'])
                ems['fixed'] += metric_max_over_ground_truths(exact_match_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text'])
            
        # Record them to the computation of F1 and EM
        if float(unfixed_hallu_rate / len(unfixed_group)) >= 0.5 and float(fixed_hallu_rate / len(fixed_group)) == 0:
            tot_changed_samples += 1
            overall_F1s['unfixed'] += f1s['unfixed'] / len(unfixed_group)
            overall_F1s['fixed'] += f1s['fixed'] / len(fixed_group)

            overall_EMs['unfixed'] += ems['unfixed'] / len(unfixed_group)
            overall_EMs['fixed'] += ems['fixed'] / len(fixed_group)

    overall_F1s['unfixed'] /= tot_changed_samples
    overall_EMs['unfixed'] /= tot_changed_samples
    overall_F1s['fixed'] /= tot_changed_samples
    overall_EMs['fixed'] /= tot_changed_samples
    return overall_F1s, overall_EMs

    
def confidence_comparison(model_name, data_type, dataset_name, compare_type='ratio'):
    """
    Compare the model confidence/perplexity on the gold answer of before fixing
    and after fixing
    data_type can be 'perplexity' or 'prob'
    compare_type can be {'ratio', 'diff'}, indicating whether return data1/data2 or data1-data2
    Return result dict
        for each ratio, there are Fixed Tok/Unfixed Tok with new Tok
                                Fixed Tok/Unfixed Tok with old Tok
                                Unfixed Tok with new Tok/Unfixed Tok with old Tok
        ratios of avg perplexity dict{random seed: ratio}
        ratios of avg probability dict{random seed: ratio}
        all ratios of perplexity dict of list, were the lists are all the ratios for that random seed
    """
    # Load the model prefix
    prefices, fixed_prefices, unfixed_new_prefices, unfixed_old_prefices = helper_get_all_file_prefix_conf(model_name, dataset_name)

    result_avg_dict = {'fixed_v_unfixednew':{}, 'fixed_v_unfixedold':{}, 'unfixednew_v_unfixedold':{}}
    result_ratio_dict = {'fixed_v_unfixednew':{}, 'fixed_v_unfixedold':{}, 'unfixednew_v_unfixedold':{}}

    result_avg_dict = {'fixed_v_unfixedold':{}}
    result_ratio_dict = {'fixed_v_unfixedold':{}}

    # Update the fixed_v_unfixednew
    for ratio_name in result_avg_dict.keys():
        # Prefix1 is the numerator
        # Prefix2 is the denominator
        if ratio_name == 'fixed_v_unfixednew':
            prefix_list1 = fixed_prefices
            prefix_list2 = unfixed_new_prefices
        elif ratio_name == 'fixed_v_unfixedold':
            prefix_list1 = fixed_prefices
            prefix_list2 = unfixed_old_prefices
        else:
            prefix_list1 = unfixed_new_prefices
            prefix_list2 = unfixed_old_prefices
        for seed in ['0', '42', '12345']:
            # For each random seed, load and compute
            # Find the corresponding prefix
            for prefix1 in prefix_list1:
                if seed in prefix1:
                    break
            for prefix2 in prefix_list2:
                if seed in prefix2:
                    break
            avg_ratio, ratio_list = confidence_comparison_for_single_pair(prefix1 + '_' + data_type + '.txt', prefix2 + '_' + data_type + '.txt', compare_type)

            result_avg_dict[ratio_name][seed] = avg_ratio
            result_ratio_dict[ratio_name][seed] = ratio_list

    return result_avg_dict, result_ratio_dict

def confidence_comparison_for_single_pair(prefix1, prefix2, compare_type):
    """
    Return the averaged ratio of (items in) pair1/pair2
    Also return a list of raw ratios
    """
    # Read from the prefix
    pair1 = helper_read_from_txt_list(prefix1)
    pair2 = helper_read_from_txt_list(prefix2)
    ratios = []
    for instance_idx in range(len(pair1)):
        if 'prob' in prefix1:
            if compare_type == 'ratio':
                ratios.append(math.exp(float(pair1[instance_idx]))/math.exp(float(pair2[instance_idx])))
            else:
                ratios.append(float(pair1[instance_idx]) - float(pair2[instance_idx]))
        else:
            if compare_type == 'ratio':
                ratios.append(float(pair1[instance_idx])/float(pair2[instance_idx]))
            else:
                ratios.append(np.log(float(pair1[instance_idx])) - np.log(float(pair2[instance_idx])))

    return sum(ratios) / len(ratios), ratios

def confidence_stat(model_name, data_type, dataset_name):
    """
    Compare the model confidence/perplexity on the gold answer of before fixing
    and after fixing
    data_type can be 'perplexity' or 'prob'
    Return result dict
        Fixed Tok
        Unfixed Tok with new Tok
        Unfixed Tok with old Tok
        dict of list, were the lists are all the ratios for that random seed
    """
    # Load the model prefix
    prefices, fixed_prefices, unfixed_new_prefices, unfixed_old_prefices = helper_get_all_file_prefix_conf(model_name, dataset_name)

    result_dict = {'fixed':{}, 'unfixedold':{}, 'unfixednew':{}}

    # Update the fixed_v_unfixednew
    for ratio_name in result_dict.keys():
        # Prefix1 is the numerator
        # Prefix2 is the denominator
        if ratio_name == 'fixed':
            prefix_list = fixed_prefices
        elif ratio_name == 'unfixedold':
            prefix_list = unfixed_old_prefices
        else:
            prefix_list = unfixed_new_prefices
        for seed in ['0', '42', '12345']:
            # For each random seed, load and compute
            # Find the corresponding prefix
            for prefix in prefix_list:
                if seed in prefix:
                    break
            values = helper_read_from_txt_list(prefix + '_' + data_type + '.txt')
            
            if data_type == 'prob':
                # Take exp for all of it
                values = [math.exp(float(value)) for value in values]
            else:
                values = [float(value) for value in values]

            result_dict[ratio_name][seed] = values

    return result_dict

def significance_test(model_name, dataset_name, base_dataset):
    """
    Compare if the models F1 differs significantly
    """
    unfixed_group =  helper_get_full_model_list([model_name])
    fixed_group = helper_get_full_model_list([model_name + '-tokenizer'])

    unfixed_F1s = dict()
    fixed_F1s = dict()
    for name in unfixed_group:
        unfixed_F1s[name] = []
    for name in fixed_group:
        fixed_F1s[name] = []
    # Load gold
    raw_datasets = helper_load_ground_truth(dataset_name)

    # Load predictions
    preds = []
    for model_setup in unfixed_group + fixed_group:
        file = open(base_dir + '/results/' + base_dataset + '/' + model_setup + '-' + dataset_name + '/eval_predictions.json')
        preds.append(json.load(file))
    # Compute the corresponding F1 instance by instance
    for instance_idx in range(len(raw_datasets['test'])):
        instance = raw_datasets['test'][instance_idx]
        for pred_idx, model_pred in enumerate(preds):
            if pred_idx < len(unfixed_group):
                # Add the F1 prediction to unfixed F1
                unfixed_F1s[unfixed_group[pred_idx]].append(metric_max_over_ground_truths(f1_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text']))
            else:
                fixed_F1s[fixed_group[pred_idx - len(unfixed_group)]].append(metric_max_over_ground_truths(f1_score, 
                model_pred[raw_datasets['test'][instance_idx]['id']], 
                raw_datasets['test'][instance_idx]['answers']['text']))

    # Average the F1s over three random seeds
    fixed_F1_array = []
    unfixed_F1_array = []
    for name in fixed_group:
        fixed_F1_array.append(fixed_F1s[name])

    for name in unfixed_group:
        unfixed_F1_array.append(unfixed_F1s[name])
    fixed_average = np.mean(np.array(fixed_F1_array), axis=0)
    unfixed_average = np.mean(np.array(unfixed_F1_array), axis=0)

    # Compute the F1 difference and Bootstrap
    return bootstrap((fixed_average - unfixed_average,), np.average, confidence_level=0.95, method='percentile')

def calibration_experiment(model_name, base_dataset='squad', target_dataset='squad'):
    """
    Return the model calibration results
    
    Returns:
        f1_list: a list of F1 on each instance, averaged over three random seeds
        prob_list: a list of corresponding log likelihood, averaged over three random seeds
    """

    f1_lists = dict()
    em_lists = dict()
    prob_lists = dict()
    raw_datasets = helper_load_ground_truth(target_dataset)
    model_list = helper_get_full_model_list([model_name])
    for name in model_list:
        f1_lists[name] = []
        em_lists[name] = []
        prob_lists[name] = []
        
        # Load the predictions and compute F1
        pred = helper_load_predictions(name, target_dataset, base_dataset)
        for instance_idx in range(len(raw_datasets['test'])):
            instance = raw_datasets['test'][instance_idx]

            f1_lists[name].append(metric_max_over_ground_truths(f1_score, pred[instance['id']], instance['answers']['text']))
            em_lists[name].append(metric_max_over_ground_truths(exact_match_score, pred[instance['id']], instance['answers']['text']))
    
    # Load the computed probabilities
    for name in model_list:
        use_old_tok = 'new' if 'tokenizer' in name else 'old'
        # Generate path
        path = base_dir + '/post_eval_analysis/output/prob_and_perp/'
        if 'bart' in name:
            # Need to insert 0 for the learning rate, because it was
            # not converted correctly
            if 'tokenizer' in name:
                dir_name = name[9:32] + '0' + name[32:]
            else:
                dir_name = name[9:22] + '0' + name[22:]
    
        path = path + dir_name + '_' + use_old_tok + '_' + target_dataset + '_prob.txt'
        probs = helper_read_from_txt_list(path)
        for idx in range(len(probs)):
            probs[idx] = float(probs[idx])
        prob_lists[name] = probs
        prob_list = np.zeros([len(probs)])
        f1_list = np.zeros([len(probs)])
        em_list = np.zeros([len(probs)])

    # Compute average across random seeds
    for name in model_list:
        prob_list += np.array(prob_lists[name]) / len(model_list)
        f1_list += np.array(f1_lists[name]) / len(model_list)
        em_list += np.array(em_lists[name]) / len(model_list)
    
    prob_list = np.exp(prob_list)

    return f1_list.tolist(), em_list.tolist(), prob_list.tolist()