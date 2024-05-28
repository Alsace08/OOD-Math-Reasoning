#!/bin/bash

export PROJECT_PATH="your/project/root/folder/path/"
export MODEL_PATH="your/model/repository/root/folder/path/"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"   # gpu id

model_name="llama2-7b"  # SFT model name

python FineTune/ID_finetune.py --model_name $model_name