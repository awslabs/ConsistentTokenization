#!/bin/bash

echo "This script automatically evaluate the model on all the available datasets."
echo "The default datasets are the one in paper, if other datasets are intended, please change the 'available_datasets' below."

read -p "Input the path to repository" base_dir
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data

read -p "Using consistent tokenization or not (y/n)?" fix_tok
read -p "Enter the model name to be evaluated: " model_name
read -p "Enter the base dataset that the models are trained on: " base_dataset
read -p "Enter the GPUs to evaluate on. Use comma to separater but do not include white space (e.g. 0,1): " gpus
read -p "Enter the random seed of model to be evaluated: " seed
read -p "Enter the learning rate: " lr

cd $base_dir
conda activate qa

declare -a available_datasets=("duorc" "squad" "NewsQA" "NaturalQuestions" "SearchQA" "TriviaQA" "bioasq" "TextbookQA")

# Create corresponding directories
mkdir -p $base_dir/results/$base_dataset/$model_name/
mkdir -p $base_dir/logs/test/$base_dataset/$model_name/

for dataset in "${available_datasets[@]}"
do
    # Find if dataset contains test.csv or validation.csv
    find_file_ary=(`find $data_dir/$dataset/ -name "test.csv"`)
    if [ ${#find_file_ary[@]} -gt 0 ]; then
        test_file_name="test.csv"
    else
        test_file_name="validation.csv"
    fi
    # If the model is bert/roberta, do extractive eval
    if [[ $model_name == *"bert"* ]]
    then
        CUDA_VISIBLE_DEVICES=$gpus python train_ext_qa.py   \
        --model_name_or_path $model_dir/$base_dataset/$model_name-$lr-$seed   \
        --validation_file "$data_dir/$dataset/$test_file_name" \
        --do_eval   \
        --max_seq_length 512  \
        --doc_stride 128   \
        --output_dir "$base_dir/results/$base_dataset/$model_name-$lr-$seed-$dataset"  2> "$base_dir/logs/test/$base_dataset/$model_name-$lr-$seed-$dataset.log"
    else
        # If fix tokenization, run the fixed_qa
        if [[ $fix_tok == *"y"* ]]
        then
            CUDA_VISIBLE_DEVICES=$gpus python run_fixed_gen_qa.py \
            --model_name_or_path "$model_dir/$base_dataset/$model_name-tokeinzer-$lr-$seed" \
            --validation_file "$data_dir/$dataset/$test_file_name" \
            --context_column context \
            --question_column question \
            --answer_column answers \
            --do_eval \
            --predict_with_generate \
            --max_seq_length 2048  \
            --doc_stride 128 \
            --output_dir "$base_dir/results/$base_dataset/$model_name-tokenizer-$lr-$seed-$dataset" 2> "$base_dir/logs/test/$base_dataset/$model_name-tokenizer-$lr-$seed-$dataset.log"
        else
            # If eval the unfixed model, eval with the unfixed script
            CUDA_VISIBLE_DEVICES=$gpus python run_unfixed_gen_qa.py \
            --model_name_or_path "$model_dir/$base_dataset/$model_name-$lr-$seed" \
            --validation_file "$data_dir/$dataset/$test_file_name" \
            --context_column context \
            --question_column question \
            --answer_column answers \
            --do_eval \
            --predict_with_generate \
            --max_seq_length 2048  \
            --doc_stride 128 \
            --output_dir "$base_dir/results/$base_dataset/$model_name-$lr-$seed-$dataset" 2> "$base_dir/logs/test/$base_dataset/$model_name-$lr-$seed-$dataset.log"
        fi
    fi
done