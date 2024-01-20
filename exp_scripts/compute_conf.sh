#!/bin/bash

echo "This script compute the dev set perplexity and probability for a given model."

echo "The resulting files will be stored in post_eval_analysis/output/prob_and_perp/. If different datasets are intended, chagne the 'available_datasets' in the script."

read -p "Input the path to repository" base_dir
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data

read -p "Enter the model name to be evaluated (Model name = {facebook/bart-base, google/t5-v1_1-base, t5-base}, the model has to be stored in the trained_models/ directory.): " model_name
read -p "Enter the base dataset that the models are trained on (base dataset = {squad, NewsQA, TriviaQA, NaturalQuestions}): " base_dataset
read -p "Enter the GPUs to evaluate on. Only ** single GPU ** is supported by this script (e.g. {0, 1, 2, 3,}): " gpus

cd $base_dir
conda activate qa

mkdir -p logs/temp/

if [[ $model_name == *"t5"* ]]
then
    lr='1e-4'
elif [[ $model_name == *"bart"* ]]
then
    lr='2e-5'
fi

declare -a random_seeds=('0' '42' '12345')
declare -a available_datasets=("NewsQA")

for seed in "${random_seeds[@]}"
do
    for dataset in "${available_datasets[@]}"
    do
        # Find if dataset contains test.csv or validation.csv
        find_file_ary=(`find $data_dir/$dataset/ -name "test.csv"`)
        if [ ${#find_file_ary[@]} -gt 0 ]; then
            test_file_name="test.csv"
        else
            test_file_name="validation.csv"
        fi

        # Output for original models
        CUDA_VISIBLE_DEVICES=gpus python confidence_stat.py  \
        --seed $seed \
        --model_name_or_path "$model_dir/$base_dataset/$model_name-$lr-$seed" \
        --output_dir "$base_dir/results/exp_temp" \
        --validation_file "$data_dir/$dataset/$test_file_name" \
        --learning_rate $lr \
        --use_old_tokenization True 2> logs/temp/temp.log

        # Output for models with consistent tokenization
        CUDA_VISIBLE_DEVICES=gpus python confidence_stat.py  \
        --seed $seed \
        --model_name_or_path "$model_dir/$base_dataset/$model_name-tokenizer-$lr-$seed" \
        --output_dir "$base_dir/results/exp_temp" \
        --validation_file "$data_dir/$dataset/$test_file_name" \
        --learning_rate $lr \
        --use_old_tokenization False 2> logs/temp/temp.log
    done
done