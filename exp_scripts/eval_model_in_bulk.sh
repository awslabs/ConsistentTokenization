#!/bin/bash

echo "This script evaluate the OOD performance for a given model, across different random seeds. The output will be stored in results/base_dataset"

echo "If you used a different set of random seeds/eval datasets, change the set of random seeds/eval dataset below."

read -p "Input the path to repository" base_dir
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data

read -p "Using consistent tokenization or not (y/n)?" fix_tok
read -p "Enter the model name to be evaluated (Model name = {facebook/bart-base, google/t5-v1_1-base, t5-base}, the model has to be stored in the trained_models/ directory.): " model_name
read -p "Enter the base dataset that the models are trained on (base dataset = {squad, NewsQA, TriviaQA, NaturalQuestions}): " base_dataset
read -p "Enter the GPUs to evaluate on. Use comma to separater but do not include white space (e.g. 0,1,2,3): " gpus

cd $base_dir
conda activate qa

# Create corresponding directories
mkdir -p $base_dir/results/$base_dataset/$model_name/
mkdir -p $base_dir/logs/test/$base_dataset/$model_name/

if [[ $model_name == *"t5"* ]]
then
    lr='1e-4'
elif [[ $model_name == *"bart"* ]]
then
    lr='2e-5'
fi

declare -a random_seeds=("0" "42" "12345")
declare -a available_datasets=("squad" "NewsQA" "NaturalQuestions" "SearchQA" "TriviaQA" "bioasq" "duorc" "TextbookQA")

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
    done
done

