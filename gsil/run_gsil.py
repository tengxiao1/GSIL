#!/usr/bin/env python
#
# Adapted from https://github.com/huggingface/alignment-handbook
import logging
import sys
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed


from alignment import (
    DataArguments,
    GSILConfig,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,

)

from alignment import GSILTrainer
import re


def apply_chat_template_dpo(
        example, tokenizer, task, auto_insert_empty_system_msg: bool = True,
):
    def is_openai_format(messages: Any) -> bool:
        """
        Check if the input messages are in OpenAI format.
        Args:
            messages (`Any`):
                Messages to check.
        Returns:
            `bool`: Whether the messages are in OpenAI format.
        """
        if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
            return all("role" in message and "content" in message for message in messages)
        return False

    def maybe_insert_system_message(messages, tokenizer):
        if messages[0]["role"] == "system":
            return

        # chat template can be one of two attributes, we check in order
        chat_template = tokenizer.chat_template
        if chat_template is None:
            chat_template = tokenizer.default_chat_template

        # confirm the jinja template refers to a system message before inserting
        if "system" in chat_template or "<|im_start|>" in chat_template:
            messages.insert(0, {"role": "system", "content": ""})

    if all(k in example.keys() for k in ("real", "generated")):
        if not is_openai_format(example["real"]) or not is_openai_format(example["generated"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the
        # final turn of a dialogue We therefore need to extract the N-1 turns to form the prompt
        if "prompt" in example and is_openai_format(example["prompt"]):
            prompt_messages = example["prompt"]
            chosen_messages = example["real"]
            rejected_messages = example["generated"]
        else:
            prompt_messages = example["real"][:-1]
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["real"][-1:]
            rejected_messages = example["generated"][-1:]

        # Prepend a system message if the first message is not a system message
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(prompt_messages, tokenizer)

        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        example["text_real"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `{task}` task! Require either the "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def apply_chat_template_spin(
        example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `{task}` task! "
            f"Require `[real, generated]` keys but found {list(example.keys())}"
        )
    return example


logger = logging.getLogger(__name__)


def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GSILConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.dataset_mixer is None and data_args.dataset_path:
        print(data_args.dataset_path)
        data_args.dataset_mixer = {}
        for idx in range(len(data_args.dataset_path)):
            data_args.dataset_mixer[data_args.dataset_path[idx]] = float(data_args.dataset_weight[idx])

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)
    training_args.per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    training_args.gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    # Increase distributed timeout to 3h to enable push to Hub to complete
    # accelerator = Accelerator()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template_spin,
        fn_kwargs={"tokenizer": tokenizer, "task": "gsil"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
        )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate spin trainer
    #########################
    gsil_trainer = GSILTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        alpha=training_args.alpha,
        loss_type=training_args.loss_type,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = gsil_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    gsil_trainer.log_metrics("train", metrics)
    gsil_trainer.save_metrics("train", metrics)
    gsil_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    gsil_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if training_args.local_rank == 0:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        gsil_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        gsil_trainer.model.config.use_cache = True
        gsil_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    # accelerator.wait_for_everyone()
    if training_args.local_rank > 0:
        torch.distributed.barrier()
    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
