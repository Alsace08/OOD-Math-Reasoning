import os
import sys

import argparse
import scipy.spatial
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

project_root_path = os.environ['PROJECT_PATH']
model_root_path = os.environ['MODEL_PATH']
sys.path.append(project_root_path)
from Data.Raw_Dataset.load_data import DatasetInfo


def load_ckpt(args):
    model_path = os.path.join(model_root_path, args.model_name)
    ckpt_name = os.path.join(project_root_path, "Checkpoints", args.model_name, "checkpoint-" + str(args.ckpt_step))
    print(f'from checkpoint: {ckpt_name}')

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map='auto',
    )
    model = model.from_pretrained(ckpt_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    return model, tokenizer


def get_output_file(args):
    model, tokenizer = load_ckpt(args)
    model.eval()
    generation_config = GenerationConfig()

    output_info = []
    datasetinfo = DatasetInfo(args.dataset, args.category)
    data_size = datasetinfo.data_size
    for i in tqdm(range(0, data_size)):
        input_data, _ = datasetinfo.load_single_sample(i)
        with torch.no_grad():
            inputs = tokenizer(input_data[:args.max_input_token_num], return_tensors="pt")
            input_ids = inputs.input_ids
            generation_output = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=args.max_output_token_num,
                output_hidden_states=True,
                output_scores=True,
            )

        hidden_states = generation_output.hidden_states
        output_scores = generation_output.scores
        output_seq = tokenizer.decode(generation_output.sequences[0][-args.max_output_token_num:])
        output_info.append({"id": i,
                            "hidden_state": hidden_states,
                            "output_scores": output_scores,
                            "output_seq": output_seq})

    filename = args.dataset + "_" + args.category + ".pkl"
    f = open(os.path.join(project_root_path, "Data/Inference_Data", args.model_name, filename), 'wb')
    pickle.dump(output_info, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID_OOD_Inference")
    parser.add_argument("--model_name", type=str, default="llama2-7b", choices=["llama2-7b", "gpt2-xl"])
    parser.add_argument("--dataset", type=str, default="MATH",
                        choices=["MultiArith", "GSM8K", "SVAMP", "AddSub", "SingleEq", "SingleOp", "MATH"])
    parser.add_argument("--category", type=str, default="algebra",
                        choices=["algebra", "counting_and_probability", "geometry", "number_theory", "precalculus", "X"])
    parser.add_argument("--max_input_token_num", type=int, default=4096)
    parser.add_argument("--max_output_token_num", type=int, default=16)
    parser.add_argument("--ckpt_step", type=int, default=10000)
    args = parser.parse_args()

    if args.dataset != "MATH":
        args.category = "X"
    if args.model_name == "llama2-7b":
        args.max_input_token_num = 4096
    elif args.model_name == "gpt2-xl":
        args.max_input_token_num = 1024

    get_output_file(args)
