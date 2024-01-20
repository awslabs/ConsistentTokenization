import re
import os

from datasets import load_dataset
from prettytable import PrettyTable
import pdb

def convert_MRQA(dataset_name):
    name_to_subset_key = {'bioasq': 'BioASQ', 'SearchQA': 'SearchQA', 'NewsQA':'NewsQA', 'squad': 'SQuAD', "TriviaQA": 'TriviaQA-web', "NaturalQuestions":'NaturalQuestionsShort',
    'TextbookQA': 'TextbookQA', 'duorc': 'DuoRC.ParaphraseRC'}

    mrqa = load_dataset('mrqa')
    dataset = mrqa.filter(lambda example: example['subset'] == name_to_subset_key[dataset_name])
    dataset = dataset.rename_column('qid', 'id')
    dataset = dataset.rename_column('detected_answers', 'raw_answers')
    dataset = dataset.remove_columns('subset')
    dataset = dataset.remove_columns('context_tokens')
    dataset = dataset.remove_columns('question_tokens')
    dataset = dataset.remove_columns('answers')

    # Split the train set to get test set
    for split in dataset.keys():
        answer_column = []
        if len(dataset[split]) != 0:
            for example in dataset[split]:
                curr_ans = {}
                curr_ans['text'] = []
                curr_ans['answer_start'] = []
                for ans_idx in range(len(example['raw_answers']['text'])):
                    curr_ans['text'] += [example['raw_answers']['text'][ans_idx]] * len(example['raw_answers']['char_spans'][ans_idx]['start'])
                    curr_ans['answer_start'] += example['raw_answers']['char_spans'][ans_idx]['start']
                answer_column.append(curr_ans)
            dataset[split] = dataset[split].add_column('answers', answer_column)
    dataset = dataset.remove_columns('raw_answers')
    output_data_stats(dataset_name, dataset)
    save_to_csv(dataset_name, dataset)

def save_to_csv(dataset_name, dataset):
    for split in dataset.keys():
        if len(dataset[split]) != 0:
            # Create the folder if not exist
            if not os.path.exists('data/' + dataset_name):
                os.makedirs('data/' + dataset_name)
            dataset[split].to_csv('data/' + dataset_name + '/' + split + '.csv')

def output_data_stats(dataset_name, dataset):
    print(f'The stats for dataset {dataset_name} shown below: ')
    
    # Construct table with dataset stats
    tab = PrettyTable()
    tab.add_column('Split', ['train', 'validation', 'test', 'Avg'])
    dataset_lens = [len(dataset['train']), len(dataset['validation']), len(dataset['test']), 0]
    tab.add_column('Number of Instances', dataset_lens)

    # Compute averaged number of words excluding special characters
    doc_len = []
    que_len = []
    ans_len = []
    for split in ['train', 'validation', 'test']:
        tot_doc_len = 0
        tot_que_len = 0
        tot_ans_len = 0
        total_ans = 0
        for instance in dataset[split]:
            tot_doc_len += len(re.findall(r'\w+', instance['context']))
            tot_que_len += len(re.findall(r'\w+', instance['question']))
            for ans in instance['answers']['text']:
                tot_ans_len += len(re.findall(r'\w+', ans))
                total_ans += 1
        
        doc_len.append(tot_doc_len / len(dataset[split]) if len(dataset[split]) != 0 else 0)
        que_len.append(tot_que_len / len(dataset[split]) if len(dataset[split]) != 0 else 0)
        ans_len.append(tot_ans_len / total_ans if total_ans != 0 else 0)
    
    # # Append average
    doc_len.append((doc_len[0]*dataset_lens[0] + doc_len[1]*dataset_lens[1] + doc_len[2] * dataset_lens[2]) / sum(dataset_lens))
    que_len.append((que_len[0]*dataset_lens[0] + que_len[1]*dataset_lens[1] + que_len[2] * dataset_lens[2]) / sum(dataset_lens))
    ans_len.append((ans_len[0]*dataset_lens[0] + ans_len[1]*dataset_lens[1] + ans_len[2] * dataset_lens[2]) / sum(dataset_lens))

    tab.add_column('Avg Doc Length', doc_len)
    tab.add_column('Avg Que Length', que_len)
    tab.add_column('Avg Ans Length', ans_len)
    print(tab)

def main():
    available_datasets = ["squad", "NewsQA", "NaturalQuestions", "SearchQA", "TriviaQA", "bioasq", "duorc", "TextbookQA"]
    for dataset_name in available_datasets:
        print("Converting dataset ", dataset_name)
        convert_MRQA(dataset_name)

if __name__ == "__main__":
    main()
