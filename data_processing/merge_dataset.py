import json, argparse, os, random



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset_dir", type=str, default="data/downloads")
    arg_parser.add_argument("--output_dir", type=str, default="data/processed")
    arg_parser.add_argument("--output_name", type=str, default=None)
    arg_parser.add_argument("--to_merge_dataset", type=str)
    arg_parser.add_argument("--to_merge_num", type=str)
    arg_parser.add_argument("--select_num", type=str, default=None)
    
    args = arg_parser.parse_args()
    
    to_merge_dataset_list = args.to_merge_dataset.split(',')
    to_merge_num_list = args.to_merge_num.split(',')
    select_num_list = args.select_num.split(',')

    assert len(to_merge_dataset_list) == len(to_merge_dataset_list) == len(select_num_list)
    to_merge_task_list = [f"{ds}_{num}" for ds, num in zip(to_merge_dataset_list, to_merge_num_list)]
    
    if args.output_name is None:
        output_dataset_name = "+".join(to_merge_dataset_list)
        output_task_name = "+".join([f"{ds}_{num}" for ds, num in zip(to_merge_dataset_list, select_num_list)])
        output_path = os.path.join(args.output_dir, output_dataset_name, output_task_name + ".jsonl")
    else:
        output_path = os.path.join(args.output_dir, args.output_name, args.output_name + "_full" + ".jsonl")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    examples = []
    for dataset, task, num in zip(to_merge_dataset_list, to_merge_task_list, select_num_list):
        source_path = os.path.join(args.dataset_dir, dataset, task + ".jsonl")
        candidate_examples = []
        with open(source_path, "r") as p:
            for line in p:
                candidate_examples.append(json.loads(line))
        if num != "full":
            candidate_examples = random.sample(candidate_examples, k=int(num))
        examples = examples + candidate_examples
    with open(output_path, "w") as f:
        for line in examples:
            f.write(
                json.dumps(line) + "\n"
            )
