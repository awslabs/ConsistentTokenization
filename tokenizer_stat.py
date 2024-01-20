# The file to output statistics of tokenizer generating inconsistent tokenization on different datasets
import re
import os

from typing import Tuple, List
from datasets import load_dataset, load_metric
from ast import literal_eval
from os.path import exists
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
)

from post_eval_analysis.utils.metric_utils import f1_score, exact_match_score

import pdb

## Editable variables
models = ['facebook/bart-base', 'google/t5-v1_1-base']
datasets = ['bioasq', 'TextbookQA', 'duorc', 'squad', 'NaturalQuestions','TriviaQA', 'NewsQA', 'SearchQA']
split = 'validation'
use_old_tokenization = True
# Arguments
base_dir = os.getenv("BASE_DIR")

test_file = None
question_column = 'question'
context_column = 'context'
answer_column = 'answers'

padding = "max_length"
max_seq_length = 2048
max_answer_length = 30
ignore_pad_token_for_loss = True
preprocessing_num_workers = None

question_answering_column_name_mapping = {
    "squad_v2": ("question", "context", "answers"),
}

for dataset_name in datasets:
    for model_name_or_path in models:
        train_file = base_dir + "/data/" + dataset_name + "/train.csv"
        # train_file = ''
        if Path(base_dir + "/data/" + dataset_name + "/validation.csv").is_file():
            validation_file = base_dir + "/data/" + dataset_name + "/validation.csv"
        else:
            validation_file = base_dir + "/data/" + dataset_name + "/test.csv"
        
        if not Path(train_file).is_file():
            train_file = None

        data_files = {}        
        if train_file is not None:
            data_files["train"] = train_file
            extension = train_file.split(".")[-1]
        if validation_file is not None:
            data_files["validation"] = validation_file
            extension = validation_file.split(".")[-1]
        if test_file is not None:
            data_files["test"] = test_file
            extension = test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
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
        if 'train' in raw_datasets:
            raw_datasets['train'] = raw_datasets['train'].filter(lambda example: example['answers']['text'][0].lower() in example['context'].lower())
            
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)

        # column_names = raw_datasets["train"].column_names
        column_names = raw_datasets["validation"].column_names

        # Get the column names for input/target.
        dataset_columns = question_answering_column_name_mapping.get(dataset_name,None)
        if question_column is None:
            question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            question_column = question_column
            if question_column not in column_names:
                raise ValueError(
                    f"--question_column' value '{question_column}' needs to be one of: {', '.join(column_names)}"
                )
        if context_column is None:
            context_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            context_column = context_column
            if context_column not in column_names:
                raise ValueError(
                    f"--context_column' value '{context_column}' needs to be one of: {', '.join(column_names)}"
                )
        if answer_column is None:
            answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
        else:
            answer_column = answer_column
            if answer_column not in column_names:
                raise ValueError(
                    f"--answer_column' value '{answer_column}' needs to be one of: {', '.join(column_names)}"
                )
            

        def preprocess_squad_batch(
            examples,
            question_column: str,
            context_column: str,
            answer_column: str,
        ) -> Tuple[List[str], List[str]]:
            questions = examples[question_column]
            contexts = examples[context_column]
            answers = examples[answer_column]
            def generate_input(_question, _context):
                return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])
            inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
            targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
            
            target_idxs = []
            for ans_idx, answer in enumerate(answers):
                if len(answer["text"]) > 0:
                    # Directly compute start by using index if the given index is correct
                    curr_target_idx = answer['answer_start'][0]+len(questions[ans_idx]) + len("question:  context: ")
                    
                    if answer['text'][0].lower() in contexts[ans_idx].lower() and inputs[ans_idx][curr_target_idx: curr_target_idx + len(answer["text"][0])] != answer["text"][0]:
                        if re.search(f"\W{re.escape(answer['text'][0].lower())}\W", contexts[ans_idx].lower()):
                            # When the answer is wrapped by two non-word characters
                            curr_target_idx = re.search(f"\W{re.escape(answer['text'][0].lower())}\W", inputs[ans_idx].lower()).span()[0] + 1
                        elif re.search(f"\W{re.escape(answer['text'][0].lower())}$", contexts[ans_idx].lower()):
                            curr_target_idx = re.search(f"\W{re.escape(answer['text'][0].lower())}$", inputs[ans_idx].lower()).span()[0] + 1
                else:
                    curr_target_idx = 0
                target_idxs.append(curr_target_idx)

            return inputs, targets, target_idxs
        
        def replace_ans_w_context_ids(model_inputs, labels, target_idxs, targets, tokenizer):
            '''
            model_inputs and labels are tokenized inputs and targets
            '''
            for instance_idx in range(len(model_inputs['input_ids'])):
                # Find the start and end token in orig context
                start_in_orig = target_idxs[instance_idx]
                end_in_orig = target_idxs[instance_idx] + len(targets[instance_idx]) - 1

                start_idx = 0
                end_idx = 0
                # Find the start and end token in tokenized context
                for id_idx, mapping in enumerate(model_inputs['offset_mapping'][instance_idx]):
                    if start_in_orig >= mapping[0] and start_in_orig <= mapping[1]:
                        start_idx = id_idx
                    if end_in_orig >= mapping[0] and start_in_orig <= mapping[1]:
                        end_idx = id_idx
                # Replace the tokenized labels with context_ids[start, end]
                if 't5' in model_name_or_path:
                    if str(labels['input_ids'][instance_idx][:labels['input_ids'][instance_idx].index(1)])[1:-1] not in str(model_inputs['input_ids'][instance_idx]):
                        # The if clause here is to deal with the prefix 3 issue in T5 tokenizer. There is no such issue in BART so this line is only for T5.
                        end_idx = min(end_idx, start_idx + max_answer_length - 2)
                        new_ids = model_inputs['input_ids'][instance_idx][start_idx:end_idx+1] + [1] + [0 for i in range(len(labels['input_ids'][instance_idx])-end_idx+start_idx-2)]
                        new_ans_ids = new_ids[: new_ids.index(1) if 1 in new_ids else len(new_ids)]
                        if str(new_ans_ids)[1:-1] in str(model_inputs['input_ids'][instance_idx]) and tokenizer.pad_token_id not in new_ans_ids:
                            # If the fixed version exsit in context, replace
                            labels['input_ids'][instance_idx] = new_ids

                else:
                    end_idx = min(end_idx, start_idx + max_answer_length - 3)
                    labels['input_ids'][instance_idx] = [0] + model_inputs['input_ids'][instance_idx][start_idx:end_idx+1] + [2] + [1 for i in range(len(labels['input_ids'][instance_idx])-end_idx+start_idx-3)]
            return labels

        def preprocess_function(examples):
            inputs, targets, target_idxs = preprocess_squad_batch(examples, question_column, context_column, answer_column)
            model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True, return_offsets_mapping=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)
                orig_labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

            if not use_old_tokenization:
                labels = replace_ans_w_context_ids(model_inputs, labels, target_idxs, targets, tokenizer)

            # amount = 0
            # for instance_idx, label in enumerate(orig_labels['input_ids']):
            #     if str(tokenizer(" " + targets[instance_idx])['input_ids'][1:-1])[1:-1] not in str(model_inputs['input_ids'][instance_idx]) and str(label[1:-1])[1:-1] not in str(model_inputs['input_ids'][instance_idx]):
            #         if targets[instance_idx] in examples['context'][instance_idx]:
            #             pdb.set_trace()
            #     # if len(labels['input_ids'][instance_idx]) != 30:
            #     #     pdb.set_trace()
            #     if label != labels['input_ids'][instance_idx]:
            #         pdb.set_trace()
            #         amount += 1

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        target_dataset = raw_datasets[split]
        target_dataset = target_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )

        not_in_context = 0.0
        em = 0.0
        f1 = 0.0
        no_ans_count = 0
        for instance_idx, instance in enumerate(tqdm(target_dataset)):
            # Check if instance['labels'] is sublist of instance['input_ids']
            # and if the decoded answer equals to not decoded answers
            if 't5' in model_name_or_path or 'T0' in model_name_or_path:
                ans_ids = instance['labels'][ : instance['labels'].index(1) if 1 in instance['labels'] else len(instance['labels'])]
            else:
                ans_ids = instance['labels'][1: instance['labels'].index(2) if 2 in instance['labels'] else len(instance['labels'])]
            if str(ans_ids)[1:-1] not in str(instance['input_ids']):
                not_in_context += 1
            try:
                em += float(exact_match_score(tokenizer.decode(ans_ids), raw_datasets[split][instance_idx]['answers']['text'][0]))
                f1 += float(f1_score(tokenizer.decode(ans_ids), raw_datasets[split][instance_idx]['answers']['text'][0]))
            except:
                no_ans_count += 1
            
            try:
                if not float(exact_match_score(tokenizer.decode(ans_ids), raw_datasets[split][instance_idx]['answers']['text'][0])):
                    # pdb.set_trace()
                    print("Ans: ", tokenizer.decode(ans_ids))
                    print("True Answer: ", raw_datasets[split][instance_idx]['answers']['text'][0])
            except:
                print("Something wrong happened, check the status.")
                pdb.set_trace()

        print("Percentage of instance whose answer is not contained in the tokenized context for " + dataset_name + " in " + model_name_or_path +" : ", not_in_context / len(target_dataset) * 100.0)

        print("Exact match scores = ", em / len(target_dataset))
        print('F1: ', f1 / len(target_dataset))
        print("No answer count: ", no_ans_count)