import os
import re
import json

import torch

import random
import pathlib
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dataclasses import dataclass, field
from datasets import Dataset
from trl import (
    ScriptArguments,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

def concat_messages(example):
    output_texts = []
    
    if isinstance(example['messages'][0], dict):
        text = ""
        for message in example['messages']:
            if message["role"] == "system":
                text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                text += "<|assistant|>\n" + message["content"].strip()
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        output_texts.append(text)
    
    elif isinstance(example['messages'][0], list):
        for messages in example['messages']:
            text = ""
            for message in messages:
                if message["role"] == "system":
                    text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    text += "<|assistant|>\n" + message["content"].strip()
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            output_texts.append(text)
        
    return output_texts

def obtain_dataset(data_path):
    if os.path.isfile(data_path):
        if data_path.endswith(".json"):
            datas = json.load(open(data_path, "r"))

        elif data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                datas = [json.loads(line) for line in f.readlines()]

        else:
            raise NotImplementedError
            
    else:
        raise NotImplementedError

    data_list = [{"messages": data["messages"]} for data in datas]
    dataset = Dataset.from_list(data_list)
    # print(dataset[0])
    return dataset 

def split_dataset(dataset: Dataset, test_size=0.01):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.dataset_num_proc = 8
    model_config.lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]

    ################
    # Model init kwargs & Tokenizer
    ################
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    if getattr(config, "quantization_config", None) is not None:
        config.quantization_config["use_exllama"] = False
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    dataset = obtain_dataset(script_args.dataset_name)
    dataset = split_dataset(dataset, test_size=0.01)
    
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None,
        formatting_func=concat_messages,
        peft_config=get_peft_config(model_config),
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)