import random, json
import pdb
import os
import argparse
from convert_dataset import encode_instruction_example


def encode_text_and_write_json(output_path, examples):
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["prompt"],
                input=None,
                output=example["answer"],
                random_template=False,
                eos_token=None,
            )
            fout.write(
                json.dumps(
                    {
                        "dataset": example["dataset"],
                        "id": example["id"],
                        "messages": [
                            {"role": "user", "content": encoded_example["prompt"].strip()},
                            {"role": "assistant", "content": encoded_example["completion"].strip()},
                        ],
                    }
                )
                + "\n"
            )




def convert_pubmedqa_data(data_path, output_dir):
    task_name = "pubmedqa"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path) as fin:
        lines = json.load(fin)
    lines = lines[task_name]
    for sample_id, sample in lines.items():
        raw_data.append({
            "id": sample_id,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)

def convert_medqa_data(data_path, output_dir):
    task_name = "medqa"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path) as fin:
        lines = json.load(fin)
    lines = lines[task_name]
    for sample_id, sample in lines.items():
        raw_data.append({
            "id": sample_id,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)



def convert_medmcqa_data(data_path, output_dir):
    task_name = "medmcqa"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path) as fin:
        lines = json.load(fin)
    lines = lines[task_name]
    for sample_id, sample in lines.items():
        raw_data.append({
            "id": sample_id,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)



def convert_bioasq_data(data_path, output_dir):
    task_name = "bioasq"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path) as fin:
        lines = json.load(fin)
    lines = lines[task_name]
    for sample_id, sample in lines.items():
        raw_data.append({
            "id": sample_id,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)


def convert_mmlu_data(data_path, output_dir):
    task_name = "mmlu"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path) as fin:
        lines = json.load(fin)
    lines = lines[task_name]
    for sample_id, sample in lines.items():
        raw_data.append({
            "id": sample_id,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)


def convert_fpb_data(data_path, output_dir):
    task_name = "fpb"
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(data_path, encoding="ISO-8859-1") as fin:
        lines = fin.readlines()
    # for text in txt_data:
    #     news, answer = text.split("@")
    #     answer = answer.strip()
    #     raw_data.append({
    #         "news": news,
    #         "answer": answer
    #     })
    # # some numbers are in the `x,xxx` format, and we want to remove the comma
    # for example in test_data:
    #     example["question"] = "What is the sentiment of this news? A. negative, B. neutral, C. positive."

    for sample_id, sample in enumerate(lines):
        news, answer = sample.split("@")
        answer = answer.strip()
        raw_data.append({
            "id": str(sample_id),
            # "instruction": "Please only answer the following question with {A,B,C}."
            "question": "What is the sentiment of this news?",
            "options": {"A": "negative", "B": "neutral", "C": "positive"},
            "answer": answer,
        })
    
    examples = []
    for sample in raw_data:
        example = {"dataset": task_name, "id": sample["id"], "prompt": "", "answer": ""}
        example["prompt"] += sample["question"] + "\n"
        for option_idx, option_text in sample["options"].items():
            example["prompt"] += option_idx + "." + option_text + "\n"
        example["answer"] += sample["answer"]
        examples.append(example)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")

    encode_text_and_write_json(output_path, examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    globals()[f"convert_{args.task_name}_data"](
            args.data_path, os.path.join(args.output_dir, args.task_name)
        )
