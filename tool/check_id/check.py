import json
import pdb
from datasets import load_dataset
from collections import Counter

def find_duplicates(lst):
    counter = Counter(lst)
    return [item for item, count in counter.items() if count > 1]

def main():
    json_path = "/mnt/petrelfs/wangpingjie/DataSelection/MY/dataset/converted_dataset/general/general_full.jsonl"
    dataset = []
    ID_list = []
    overlap_ID_list = []
    # with open(json_path) as f:
    #     for line in f:
    dataset = load_dataset("json", data_files=json_path)
    # for sample in dataset["train"]:
    #     ID = sample["id"]
    #     if ID not in ID_list:
    #         ID_list.append(ID)
    #     else:
    #         overlap_ID_list.append(ID)
    overlap_ID_list = find_duplicates(dataset["train"]["id"])
    pdb.set_trace()


main()