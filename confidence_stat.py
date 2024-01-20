# The file to output the statistics of prediction logits and perplexity
import logging
import math
import os
import re
import sys
import json
import torch

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ast import literal_eval
from tqdm import tqdm
import datasets
from datasets import load_dataset
from post_eval_analysis.utils.helper_methods import helper_generate_file_prefix_conf
import pdb

from qa_utils.trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    use_old_tokenization: bool = field(
        default=False, metadata={"help": "Choose whether to use old toknization"}
    )
    
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    val_max_answer_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length

torch.cuda.empty_cache()
question_answering_column_name_mapping = {
    "squad_v2": ("question", "context", "answers"),
}

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load and preprocess the dataset
    data_files = {}

    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Get the corresponding dataset name
    dataset_names = ['duorc', 'bioasq', 'NaturalQuestions', 'NewsQA', 'SearchQA', 'squad', 'TextbookQA', 'TriviaQA']
    for dataset_name in dataset_names:
        if dataset_name in data_args.validation_file:
            break

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

    # Load tokenizer and model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["validation"].column_names
    context_column = "context"
    question_column = "question"
    answer_column = "answers"

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else False



    ######## Data preprocessing ######
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
                # pdb.set_trace()
                if answer['text'][0].lower() in contexts[ans_idx].lower() and inputs[ans_idx][curr_target_idx: curr_target_idx + len(answer["text"][0])+1] != answer["text"][0]:
                    curr_target_idx = inputs[ans_idx].lower().find(answer['text'][0].lower())
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
            if 't5' in model_args.model_name_or_path:
                new_ids = model_inputs['input_ids'][instance_idx][start_idx:end_idx+1] + [1] + [0 for i in range(len(labels['input_ids'][instance_idx])-end_idx+start_idx-2)]
                new_ans_ids = new_ids[: new_ids.index(1) if 1 in new_ids else len(new_ids)]
                if str(new_ans_ids)[1:-1] in str(model_inputs['input_ids'][instance_idx]) and tokenizer.pad_token_id not in new_ans_ids:
                    # If the fixed version exsit in context, replace
                    labels['input_ids'][instance_idx] = new_ids

            else:
                labels['input_ids'][instance_idx] = [0] + model_inputs['input_ids'][instance_idx][start_idx:end_idx+1] + [2] + [1 for i in range(len(labels['input_ids'][instance_idx])-end_idx+start_idx-3)]
        return labels

    # Validation preprocessing
    def preprocess_validation_function(examples):
        inputs, targets, target_idxs = preprocess_squad_batch(examples, question_column, context_column, answer_column)

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            # return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_answer_length, padding=padding, truncation=True)

        if not data_args.use_old_tokenization:
            labels = replace_ans_w_context_ids(model_inputs, labels, target_idxs, targets, tokenizer)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        sample_mapping = list(range(len(model_inputs["input_ids"])))

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def replace_ans_w_context_ids_single_input(eval_tokenized_sample, example, tokenizer):
        input_text = " ".join(["question:", example['question'].lstrip(), "context:", example['context'].lstrip()])
        # For each answer, find th corresponding target idx
        target_idxs = []
        for ans_idx, answer in enumerate(example['answers']['text']):
            curr_target_idx = example['answers']['answer_start'][ans_idx] + len(example['question']) + len("question:  context: ")
            if answer in example['context'].lower() and input_text[curr_target_idx:curr_target_idx + len(answer)+1] != answer:
                curr_target_idx = input_text.lower().find(answer.lower())
            target_idxs.append(curr_target_idx)
        
        # For each target idx, find the corresponding label_ids
        label_ids = []
        for ans_idx, start_in_orig in enumerate(target_idxs):
            end_in_orig = start_in_orig + len(example['answers']['text'][ans_idx])
            start_idx = 0
            end_idx = 0
            for id_idx, mapping in enumerate(eval_tokenized_sample['offset_mapping']):
                if start_in_orig >= mapping[0] and start_in_orig <= mapping[1]:
                    start_idx = id_idx
                if end_in_orig >= mapping[0] and start_in_orig <= mapping[1]:
                    end_idx = id_idx
                # Replace the tokenized labels with context_ids[start, end]
                if 't5' in model_args.model_name_or_path:
                    new_ids = eval_tokenized_sample['input_ids'][start_idx:end_idx+1] + [1] + [0 for i in range(len(eval_tokenized_sample['labels'])-end_idx+start_idx-2)]
                    new_ans_ids = new_ids[: new_ids.index(1) if 1 in new_ids else len(new_ids)]
                    if str(new_ans_ids)[1:-1] in str(eval_tokenized_sample['input_ids']) and tokenizer.pad_token_id not in new_ans_ids:
                        # If the fixed version exsit in context, replace
                        label_ids.append(new_ids)

                else:
                    label_ids.append([0] + eval_tokenized_sample['input_ids'][start_idx:end_idx+1] + [2] + [1 for i in range(len(eval_tokenized_sample['labels'])-end_idx+start_idx-3)])
        return label_ids

    # Convert thevalidation dataset
    eval_examples = raw_datasets["validation"]
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_examples.map(
            preprocess_validation_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    training_args.per_device_eval_batch_size = 1
    # Initialize a trainer just to use the get dataloader method
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        post_process_function=None,
    )

    # Note: This only works for eval on single cuda!
    log_softmax = torch.nn.LogSoftmax(dim=2)
    eval_loader = trainer.get_test_dataloader(eval_dataset)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_seq_prob = []
    all_seq_perplexity = []
    seq_lens = []
    losses = []
    
    instance_idx = 0
    for batch in tqdm(eval_loader, total=len(eval_loader)):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        if len(eval_examples[instance_idx]['answers']['text']) >= 2 and eval_examples[instance_idx]['answers']['text'][0] != eval_examples[instance_idx]['answers']['text'][-1]:
            # Find tokens and their corresponding correct tokenization
            label_ids = replace_ans_w_context_ids_single_input(eval_dataset[instance_idx], eval_examples[instance_idx], tokenizer)
        else:
            label_ids = [batch['labels'][0]]

        with torch.no_grad():
            outputs = model(**batch)
            # Softmax to compute pobability
            softmaxed_logits = log_softmax(outputs.logits)
            # Compute the probability for gold sequence
            max_seq_prob = -1e10
            max_corresponding_len = 0
            losses.append(outputs.loss.item())
            for label_id in label_ids:
                seq_prob = 0.0
                seq_len = 0
                for token_idx, token in enumerate(label_id):
                    if token != -100 and token_idx < data_args.max_answer_length:
                        seq_len += 1
                        # Add the prob for current token
                        seq_prob += softmaxed_logits[0, token_idx, token]
                
                # Log sequence negative log probability
                if seq_prob.item() >= max_seq_prob:
                    max_seq_prob = seq_prob.item()
                    max_corresponding_len = seq_len
                
            # if max_seq_prob == -1e10:    
            #     pdb.set_trace()
            all_seq_prob.append(max_seq_prob)
            # Log perplexity of the sequence
            all_seq_perplexity.append(math.exp(-max_seq_prob/max_corresponding_len))
            seq_lens.append(max_corresponding_len)
        instance_idx += 1

    if "t5" in model_args.model_name_or_path:
        if "base" in model_args.model_name_or_path:
            short_model_name = 't5-base'
        else:
            short_model_name = 't5-small'
    else:
        short_model_name = 'bart-base'

    # Save these to files
    file_prefix = helper_generate_file_prefix_conf(short_model_name, model_args.model_name_or_path, str(training_args.learning_rate), str(training_args.seed), data_args.use_old_tokenization, dataset_name)

    perplexity_file_path = file_prefix + '_perplexity.txt'
    prob_file_path = file_prefix + '_prob.txt'
    seq_len_file_path = file_prefix + '_seq_len.txt'
    loss_file_path = file_prefix + '_loss.txt'

    with open(perplexity_file_path, 'w') as fp:
        fp.write('\n'.join(str(item) for item in all_seq_perplexity))
    with open(prob_file_path, 'w') as fp:
        fp.write('\n'.join(str(item) for item in all_seq_prob))
    with open(seq_len_file_path, 'w') as fp:
        fp.write('\n'.join(str(item) for item in seq_lens))
    with open(loss_file_path, 'w') as fp:
        fp.write('\n'.join(str(item) for item in losses))


if __name__ == "__main__":
    main()