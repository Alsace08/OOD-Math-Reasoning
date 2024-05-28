import os
import sys

import argparse
import scipy.spatial
import json
import torch
import numpy as np
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from datasets import Dataset, load_dataset
from trl import SFTTrainer

project_root_path = os.environ['PROJECT_PATH']
model_root_path = os.environ['MODEL_PATH']
sys.path.append(project_root_path)
from Data.Raw_Dataset.load_data import DatasetInfo


def train(args):
    # load base model
    model_path = os.path.join(model_root_path, args.model_name)
    if args.model_name == "llama2-7b":
        per_device_train_batch_size = 2
    elif args.model_name == "gpt2-xl":
        per_device_train_batch_size = 32
    else:
        raise "Invalid Model!"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    # load training arguments
    model.enable_input_require_grads()
    output_dir = project_root_path + 'Checkpoints/' + args.model_name
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        optim="adamw_torch",
        learning_rate=1e-5,
        eval_steps=500,
        save_steps=500,
        logging_steps=1,
        evaluation_strategy="steps",
        group_by_length=False,
        num_train_epochs=100,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=10,
    )

    # load training set
    datasetinfo = DatasetInfo("MultiArith", "X")
    train_data = []
    val_data = []
    for i in range(600):
        input_data, output_data = datasetinfo.load_single_sample(i)
        tmp_data = "Q:" + input_data + "\nA: " + str(output_data) + "<\\s>"
        if i % 100 < 60:
            train_data.append({'text': tmp_data})
        else:
            val_data.append({'text': tmp_data})
    train_data = Dataset.from_dict({key: [dic[key] for dic in train_data] for key in train_data[0]})
    val_data = Dataset.from_dict({key: [dic[key] for dic in val_data] for key in val_data[0]})

    # load trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    output_dir = project_root_path + '/Training_Output/output_' + args.model_name
    trainer.train()
    trainer.model.save_pretrained(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="finetune")
    parser.add_argument("--model_name", type=str, default="llama2-7b", choices=["llama2-7b", "gpt2-xl"])
    args = parser.parse_args()
    train(args)
