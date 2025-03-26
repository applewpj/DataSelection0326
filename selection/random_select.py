import argparse
import random
import json
import pdb
import os, json



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--converted_data_dir", type=str, default="data/downloads")
    arg_parser.add_argument("--selected_data_dir", type=str, default="./dataset/selected_dataset")
    arg_parser.add_argument("--dataset", type=str)
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument("--p", type=float, default=0.05)
    args = arg_parser.parse_args()
    
    # load converted data
    mode = "random"
    random.seed(args.seed)
    converted_data_name = args.dataset + "_data.jsonl"
    converted_data_path = os.path.join(args.converted_data_dir, args.dataset, converted_data_name)
    converted_data = []
    with open(converted_data_path, "r") as f:
        for line in f.readlines():
            converted_data.append(json.loads(line))
    
    # randomly select data
    num_data = len(converted_data)
    num_selected = int(num_data * args.p)
    index_selected = random.sample(range(num_data), num_selected)
    selected_data = [converted_data[idx] for idx in index_selected]
    
    # save selected data
    selected_data_name = converted_data_name[:-6] + f"_p{str(args.p)}.jsonl"
    selected_data_path = os.path.join(args.selected_data_dir, mode, args.dataset, selected_data_name)
    if not os.path.exists(os.path.dirname(selected_data_path)):
        os.makedirs(os.path.dirname(selected_data_path))
    with open(selected_data_path, "w") as f:
        # json.dump(selected_data, f)
        for idx in range(len(selected_data)):
            f.write(
                    json.dumps(
                        selected_data[idx]
                    )
                    + "\n"
                )
    print("Finished Random Selection!")