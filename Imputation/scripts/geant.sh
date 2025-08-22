#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set variables
seq_len=50
model="TM-LLM"
percent=100
mask_rate="0.1"
train_epochs="1"
llm_model="gpt2"
sample_num=3000 # number of training samples
Lambda=4
itr="1"
feature="300"

python -u run.py \
    --train_epochs $train_epochs \
    --itr $itr \
    --task_name "imputation" \
    --is_training "1" \
    --root_path "../datasets/net_traffic/GEANT" \
    --data_path "geant.csv" \
    --model_id "geant-feature_${feature}_${model}_samplenum${sample_num}" \
    --sample_num $sample_num \
    --llm_model $llm_model \
    --data "net_traffic_geant" \
    --seq_len $seq_len \
    --batch_size "35" \
    --learning_rate "0.001" \
    --mlp "1" \
    --d_model "768" \
    --n_heads "4" \
    --d_ff "768" \
    --enc_in $seq_len \
    --dec_in $feature \
    --c_out $feature \
    --freq "h" \
    --Lambda $Lambda \
    --percent $percent \
    --gpt_layer "6" \
    --model $model \
    --patience "5" \
    --mask_rate $mask_rate