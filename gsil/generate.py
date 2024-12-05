# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings

warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])


def save_jsonl(datas, file_path):
    with open(file_path, 'w') as f:
        for item in datas:
            json_string = json.dumps(item, ensure_ascii=False)
            f.write(json_string + "\n")
    print("write success in file: {}\nlen of file: {}".format(file_path, len(datas)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_porcess_index", type=int, default=0, help="Number of GPUs to be used per Worker")
    parser.add_argument("--data_porcess_num", type=int, default=1, help="Number of GPUs to be used per Worker")
    parser.add_argument('--model', type=str, default='alignment-handbook/zephyr-7b-sft-full')
    parser.add_argument('--output_dir', type=str, default='./data/zephyr_full_62k_greedy/iter0')
    parser.add_argument('--input_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument('--split', type=str, default='train_prefs')
    return parser.parse_args()


def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                max_length=1024,
                return_tensors="pt",
                padding='longest',
                truncation=True,
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok


def main():
    args = parse_arguments()
    model_path = args.model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datas = load_dataset(args.input_dir, split=args.split)
    data_start_index = int(args.data_porcess_index * len(datas) / args.data_porcess_num)
    data_end_index = int((args.data_porcess_index + 1) * len(datas) / args.data_porcess_num)
    if args.data_porcess_index == args.data_porcess_num - 1:
        data_end_index = len(datas)
    data = datas.select(range(data_start_index, data_end_index))

    print(f"Runing on host: {args.data_porcess_index}, start_index: {data_start_index}, end_index: {data_end_index}")
    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # following SPIN
    prompts_all = ["### Instruction: " + data[idx]['prompt'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx]['chosen'][0]['content'] for idx in range(len(data))]
    corrects_all = [data[idx]['chosen'][1]['content'] for idx in range(len(data))]

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=4)

        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized = model.generate(
                **prompts_tokenized,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id
            )
            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):]
                                 for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]
            # decode gen. tokens
            outputs = tokenizer.batch_decode(outputs_tokenized)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered = gather_object(results)
    results = [r.replace("</s>", "").lstrip() for r in results_gathered]

    res = []
    output_file = f"{args.output_dir}/{args.model.split('/')[-1]}" \
                  f"_{args.data_porcess_index}.jsonl"
    if accelerator.is_local_main_process:
        timediff = time.time() - start
        print(f"time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"real":
                    [
                        {"role": "user", "content": prompts_old[idx]},
                        {"role": "assistant", "content": corrects_all[idx]
                    }
                ],
                "generated":
                    [
                        {"role": "user", "content": prompts_old[idx]},
                        {"role": "assistant", "content": results[idx]}
                    ]
                }
            res.append(d)

    save_jsonl(res, output_file)
    print("Generation Completed!\n")


if __name__ == "__main__":
    main()
