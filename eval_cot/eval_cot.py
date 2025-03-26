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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
    # parser.add_argument("--sample_num", type=int, default=1)
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
        raise NotImplementedError
    
    return datas

def prepare_dataset(data_path, output_path, cache_file, resume):
    datas = load_dataset(data_path)
    
    if os.path.exists(os.path.join(output_path, cache_file)) and resume:
        with open(os.path.join(output_path, cache_file), "r") as f:
            cache = [json.loads(line) for line in f.readlines()]
        datas = datas[len(cache):]

    return datas

def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def update_cache(output_path, cache_file, temp_cache):
    with open(os.path.join(output_path, cache_file), "a") as f:
        for line in temp_cache:
            f.write(json.dumps(line) + "\n")

def direct_infer(args, model, tokenizer, dataset):
    label_set = list(set([data["messages"][-1]["content"] for data in dataset]))
    # DIRECT_INFER_PROMPT = f"""The answer is """
    for i in tqdm(range(0, len(dataset), args.batch_size), ncols=100):
        batch_datas = dataset[i:i+args.batch_size]
        batch_inputs = [f"""Question: {data["messages"][0]["content"]} Answer: """ for data in batch_datas]
        
        batch_tokenized_inputs = tokenizer(batch_inputs, padding="longest", return_tensors="pt")
        batch_tokenized_inputs = {key: value.cuda() for key, value in batch_tokenized_inputs.items()}
        batch_tokenized_answer_outputs = model.generate(
            **batch_tokenized_inputs,
            max_new_tokens=1,
            do_sample=False,
            sequence_bias={tuple(tokenizer.encode(label)[-1:]): 100.0 for label in label_set},
        )
        batch_full_text = tokenizer.batch_decode(batch_tokenized_answer_outputs, skip_special_tokens=True)
        batch_answer_outputs = [
            text[len(input):] for input, text in zip(batch_inputs, batch_full_text)
        ]
        
        for batch_id, data in enumerate(batch_datas):
            data["cot"] = ""
            data["prediction"] = batch_answer_outputs[batch_id]
        
        update_cache(args.output_path, args.cache_file, batch_datas)
        

def cot_infer(args, model, tokenizer, dataset):
    label_set = set([data["messages"][-1]["content"] for data in dataset])
    
    for i in tqdm(range(0, len(dataset), args.batch_size), ncols=100):
        batch_datas = dataset[i:i+args.batch_size]
        
        # cot inference
        batch_inputs = [concat_messages(data["messages"][0:1], tokenizer) for data in batch_datas]
        batch_tokenized_inputs = tokenizer(batch_inputs, padding="longest", return_tensors="pt")
        batch_tokenized_inputs = {key: value.cuda() for key, value in batch_tokenized_inputs.items()}
        batch_tokenized_outputs = model.generate(
            **batch_tokenized_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        batch_outputs = tokenizer.batch_decode(batch_tokenized_outputs, skip_special_tokens=True)
        
        # get answer inference
        batch_get_answer_inputs = [f"""{output} Therefore, the answer is """ for output in batch_outputs]
        batch_tokenized_get_answer_inputs = tokenizer(batch_get_answer_inputs, padding="longest", return_tensors="pt")
        batch_tokenized_get_answer_inputs = {key: value.cuda() for key, value in batch_tokenized_get_answer_inputs.items()}
        batch_tokenized_answer_outputs = model.generate(
            **batch_tokenized_get_answer_inputs,
            max_new_tokens=1,
            do_sample=False,
            sequence_bias={tuple(tokenizer.encode(label)[-1:]): 100.0 for label in label_set},
        )
        batch_full_text = tokenizer.batch_decode(batch_tokenized_answer_outputs, skip_special_tokens=True)
        
        batch_cot_outputs = [
            text[len(input):] for input, text in zip(batch_inputs, batch_full_text)
        ]
        batch_answer_outputs = [
            text[len(input):] for input, text in zip(batch_get_answer_inputs, batch_full_text)
        ]
        for batch_id, data in enumerate(batch_datas):
            data["cot"] = batch_cot_outputs[batch_id]
            data["prediction"] = batch_answer_outputs[batch_id]
        
        update_cache(args.output_path, args.cache_file, batch_datas)

def score(args):
    with open(os.path.join(args.output_path, args.cache_file), "r") as f:
        datas = [json.loads(line) for line in f.readlines()]
    
    acc = sum([data["prediction"] == data["messages"][-1]["content"] for data in datas]) / len(datas)
    results = {'score': {"acc": acc}, 'count': len(datas), 'args': vars(args)}
    with open(os.path.join(args.output_path, args.result_file), "w") as f:
        json.dump(results, f, indent=4, separators=(',', ': '))

def load_model_and_tokenizer(model_path, peft_path=None):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if peft_path is not None:
        model = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()
        model.to(torch.float16)
    
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.peft_path)
    dataset = prepare_dataset(args.data_path, args.output_path, args.cache_file, args.resume)
    
    if args.peft_path is not None and not args.direct_answer:
        cot_infer(args, model, tokenizer, dataset)
    else:
        direct_infer(args, model, tokenizer, dataset)
    score(args)
    
if __name__ == "__main__":
    args = parse_args()
    main()