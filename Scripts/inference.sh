#!/bin/bash

export PROJECT_PATH="your/project/root/folder/path/"
export MODEL_PATH="your/model/repository/root/folder/path/"
export CUDA_VISIBLE_DEVICES="0,1"  # gpu id

model_name="llama2-7b"  # SFT model name
max_output_token_num="16"
ckpt_step="10000"   # checkpoint step as you selected

dataset_list=(MultiArith GSM8K SVAMP AddSub SingleEq SingleOp)
category="X"
for i in ${dataset_list[*]}; do
    python Inference/ID_OOD_inference.py --model_name $model_name \
                                         --dataset "$i" \
                                         --category $category \
                                         --max_output_token_num $max_output_token_num \
                                         --ckpt_step $ckpt_step
done

dataset="MATH"
category_list=(algebra geometry counting_and_probability number_theory precalculus)
for i in ${category_list[*]}; do
    python Inference/ID_OOD_inference.py --model_name $model_name \
                                         --dataset $dataset \
                                         --category "$i" \
                                         --max_output_token_num $max_output_token_num \
                                         --ckpt_step $ckpt_step
done