import os
import re
import sys
import time
import json
import argparse
import random
from itertools import chain
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def parse_args():
    parser = argparse.ArgumentParser(prog="Data Selection Evaluation")

    # data args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)

    # model args
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)

    # inference args
    parser.add_argument("--prepare_func", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--direct_answer", action="store_true", default=False)

    # log args
    parser.add_argument(
        "--cache_file", default="cache.jsonl", help="name of the cache file"
    )
    parser.add_argument(
        "--result_file", default="result.json", help="name of the results file"
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path), exist_ok=True)

    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=4, sort_keys=False))

    return args

def load_dataset(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            datas = json.load(data_path)

    elif data_path.endswith(".jsonl"):
        with open(data_path, "r") as f:
            datas = [json.loads(line) for line in f.readlines()]
    
    else:
        raise NotImplementationError
    
    return datas

def prepare_dataset(data_path, output_path, cache_file, resume):
    datas = load_dataset(data_path)
    
    if os.path.exist(os.path.join(output_path, cache_file)) and resume:
        with open(os.path.join(output_path, cache_file), "r") as f:
            cache = [json.loads(line) for line in f.readlines()]
        datas = datas[len(cache):]

    return datas

def cot_infer(args, model, tokenizer, dataset):
    full_predictions, short_predictions = [], []
    for i in range(0, len(dataset), args.batch_size):
        batch_datas = dataset[i:i+args.batch_size]
        batch_targets = [data["message"][-1]["content"] for data in batch_datas]
        
        # cot inference
        # batch_inputs = [data["message"][-1]["content"] for data in batch_datas]
        batch_tokenized_inputs = tokenizer(batch_inputs, padding="longest", return_tensors="pt")
        batch_tokenized_inputs = {key: value.cuda() for key, value in batch_tokenized_inputs.items()}
        batch_tokenized_outputs = model(
            **batch_tokenized_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        batch_outputs = tokenizer.batch_decode(batch_tokenized_outputs)
        
        # get answer inference
        batch_get_answer_inputs = [f"""{output} Therefore, the answer is """ for output in batch_outputs]
        batch_tokenized_get_answer_inputs = tokenizer(batch_get_answer_inputs, padding="longest", return_tensors="pt")
        batch_tokenized_get_answer_inputs = {key: value.cuda() for key, value in batch_tokenized_get_answer_inputs.items()}
        batch_tokenized_answer_outputs = model(
            **batch_tokenized_get_answer_inputs,
            max_new_tokens=1,
            do_sample=False,
            sequence_bias={tuple(tokenizer.encode(key)[-1:]): 100.0 for key in labels},
        )
        batch_full_text = tokenizer.batch_decode(batch_tokenized_answer_outputs)
        
        
        batch_cot_outputs = [
            text[len(input):] for input, text in zip(batch_inputs, batch_full_text)
        ]
        batch_answer_outputs = [
            text[len(input):] for input, text in zip(batch_get_answer_inputs, batch_full_text)
        ]
        
    
    return full_predictions, short_predictions


def main():
    dataset = load_dataset(args.data_path)
    full_predictions, short_predictions = cot_infer(args, model, tokenizer, dataset)
    # acc = exact_match.compute(predictions=short_predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]

if __name__ == "__main__":
    args = parse_args()
    main()