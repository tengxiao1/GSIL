"""
Multi Host Gpu Training Data Process（*.jsonl format files, Generation By 'batched_generate_vllm.py'）
"""
import datasets
import pandas as pd
from datasets import load_dataset
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random

from tqdm import tqdm


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="")
    return parser.parse_args()


def load_and_process_data_ultrachat(dataset_name, split):
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}],
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []


def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")


def save_to_parquet(dataset, path):
    try:
        pq.write_table(dataset.data.table, path)
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")


def read_jsonl(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                res.append(json.loads(line))
            except Exception as e:
                continue
    print("len of jsonl: {}".format(len(res)))
    return res


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        res = json.load(f)
    print("len of json: {}".format(len(res)))
    return res


def main(args):
    res = []
    file_list = os.listdir(args.data_dir)
    for file in tqdm(file_list):
        if "jsonl" in file:
            res.extend(read_jsonl(os.path.join(args.data_dir, file)))

    dataset_dict = datasets.Dataset.from_pandas(pd.DataFrame(data=res))
    store_dir = os.path.join(args.data_dir, "train_data")
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    pq.write_table(dataset_dict.data.table, os.path.join(store_dir, "train_prefs-00000-of-00001.parquet"))


if __name__ == "__main__":
    args = setup_arg_parser()
    main(args)
