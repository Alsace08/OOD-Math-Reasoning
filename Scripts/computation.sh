#!/bin/bash

export PROJECT_PATH="your/project/root/folder/path/"

model_name="llama2-7b"
max_output_token_num="16"
max_order="5"   # Differential Smoothing Order

dataset_list=(MultiArith GSM8K SVAMP AddSub SingleEq SingleOp)
category="X"
for i in ${dataset_list[*]}; do
    python Computation/ID_OOD_score.py --model_name $model_name \
                                       --dataset "$i" \
                                       --category $category \
                                       --max_output_token_num $max_output_token_num \
                                       --max_order $max_order
done

dataset="MATH"
category_list=(algebra geometry counting_and_probability number_theory precalculus)
for i in ${category_list[*]}; do
    python Computation/ID_OOD_score.py --model_name $model_name \
                                       --dataset $dataset \
                                       --category "$i" \
                                       --max_output_token_num $max_output_token_num \
                                       --max_order $max_order
done
