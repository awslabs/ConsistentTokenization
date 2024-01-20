#!/bin/bash

echo "This script train models with consistent tokenization or original tokenization. trained models are saved in trained_models/base_dataset"

read -p "Input the path to repository" base_dir
export model_dir=$base_dir/trained_models
export data_dir=$base_dir/data

read -p "Using consistent tokenization or not (y/n)?" fix_tok
read -p "Enter the model name to train (e.g. facebook/bart-base, t5-base, google/t5-v1_1): " model_name
read -p "Enter the GPUs to train on. Use comma to separate but do not include white space (e.g. 0,1): " gpus
read -p "Enter the training dataset name: " dataset_name
read -p "Enter the learning rate: " lr
read -p "Enter the batch size: " batch_size
read -p "Enter the random seed: " seed

cd $base_dir
conda activate qa

# Create corresponding directory
mkdir -p $base_dir/logs/training/$dataset_name/
mkdir -p $model_dir/$dataset_name/$model_name/

if [[ $model_name == *"t5"* ]]
then
    epoch='15'
else
    epoch='10'
fi

if [[ $fix_tok == *"y"* ]]
then
    CUDA_VISIBLE_DEVICES=$gpus python run_fixed_gen_qa.py \
    --model_name_or_path $model_name \
    --train_file "$data_dir/$dataset_name/train.csv" \
    --validation_file "$data_dir/$dataset_name/validation.csv" \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 32 \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --max_seq_length 2048  \
    --doc_stride 128 \
    --save_strategy 'epoch' \
    --load_best_model_at_end True \
    --metric_for_best_model 'eval_f1' \
    --evaluation_strategy='epoch' \
    --output_dir "$model_dir/$dataset_name/$model_name-tokenizer-$lr-$seed" 2> "$base_dir/logs/training/$dataset_name/$model_name-tokenizer-$lr-$seed-err.log" 1> "$base_dir/logs/training/$dataset_name/$model_name-tokenizer-$lr-$seed.log" &
else
    CUDA_VISIBLE_DEVICES=$gpus python run_unfixed_gen_qa.py \
    --model_name_or_path $model_name \
    --train_file "$data_dir/$dataset_name/train.csv" \
    --validation_file "$data_dir/$dataset_name/validation.csv" \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --do_eval \
    --seed $seed \
    --predict_with_generate \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 32 \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --max_seq_length 2048  \
    --doc_stride 128 \
    --save_strategy 'epoch' \
    --load_best_model_at_end True \
    --metric_for_best_model 'eval_f1' \
    --evaluation_strategy='epoch' \
    --output_dir "$model_dir/$dataset_name/$model_name-$lr-$seed" 2> "$base_dir/logs/training/$dataset_name/$model_name-$lr-$seed-err.log" 1> "$base_dir/logs/training/$dataset_name/$model_name-$lr-$seed.log" &
fi