#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

from torch.nn import CrossEntropyLoss
import argparse
import copy
import logging
import math
import os
import numpy as np
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
# Hardcoded to use the type I want
from transformers.models.gpt2.modeling_gpt2 import GPT2ProbedCLM, GPT2ForSequenceClassification
from transformers.utils.gen_xfactuals import gen_counterfactual


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--lm_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library) to train on language modeling.",
    )
    parser.add_argument(
        "--lm_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--probe_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library) to train on probing.",
    )
    parser.add_argument(
        "--probe_dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lm_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--eval_only", action="store_true", help="Only do evaluation.")
    args = parser.parse_args()

    # Sanity checks
    if args.lm_dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def _load_dataset(args, dataset_name, dataset_config_name, use_file=False):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if dataset_name is not None and not use_file:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(dataset_name, dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_group_text_fn(block_size):
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    return group_texts


def _train_lm(args, probed_model, tokenizer, accelerator, raw_datasets):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_lm_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_lm_datasets.map(
            get_group_text_fn(block_size),
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    train_with_prepped_dataset(train_dataset, eval_dataset, probed_model, args, accelerator, tokenizer)


def _train_probe(args, probed_model, tokenizer, accelerator, raw_datasets):
    # For imdb
    # text_column_name = 'review'
    # label_column_name = 'sentiment'
    # label_mapping = {'positive': 1, 'negative': 0}
    # For wizard
    text_column_name = 'text'
    label_column_name = 'gender'
    label_mapping = {0: 0, 1: 1, 2: 2}

    def tokenize_function(examples):
        tokenized_batch = tokenizer(examples[text_column_name])
        return tokenized_batch

    def label_to_int(batch):
        batch[label_column_name] = [label_mapping[label] for label in batch[label_column_name]]
        return batch

    def too_long(batch):
        return len(batch['input_ids']) < tokenizer.model_max_length

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    tokenized_datasets = tokenized_datasets.map(label_to_int, batched=True)
    tokenized_datasets = tokenized_datasets.filter(too_long, batched=False)
    tokenized_datasets = tokenized_datasets.rename_column(label_column_name, 'labels')
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    train_with_prepped_dataset(train_dataset, eval_dataset, probed_model, args, accelerator, tokenizer)


def train_with_prepped_dataset(train_dataset, eval_dataset, probed_model, args, accelerator, tokenizer):
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in probed_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in probed_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # Prepare everything with our `accelerator`.
    probed_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        probed_model, optimizer, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        probed_model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        if not args.eval_only:
            probed_model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = probed_model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

        probed_model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = probed_model(**batch)
            if args.eval_only:
                ids = batch['input_ids'].detach().cpu().numpy().tolist()[0]  # Convert tensor to flat list
                text_version = tokenizer.decode(ids)
                print("Input:\t\t", text_version)
                print("Pred logits:\t", outputs.logits)
                print("Label:\t\t", batch['labels'])
                print()
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")
        if args.eval_only:
            break  # Don't keep looping if you only evaluate once.


def _gen_xfacts(args, probed_model, tokenizer, accelerator, raw_datasets):
    # Given a sentence with, create counterfactuals for different genders and keep the original input
    # ids, so we end up creating a dataset of different embeddings all being supervised to the same next word
    # prediction.
    # Same prep as for training the probe.
    # For wizard
    text_column_name = 'text'
    label_column_name = 'gender'
    label_mapping = {0: 0, 1: 1, 2: 2}

    def tokenize_function(examples):
        tokenized_batch = tokenizer(examples[text_column_name])
        return tokenized_batch

    def label_to_int(batch):
        batch[label_column_name] = [label_mapping[label] for label in batch[label_column_name]]
        return batch

    def too_long(batch):
        return 3 < len(batch['input_ids']) < tokenizer.model_max_length

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    tokenized_datasets = tokenized_datasets.map(label_to_int, batched=True)
    tokenized_datasets = tokenized_datasets.filter(too_long, batched=False)
    tokenized_datasets = tokenized_datasets.rename_column(label_column_name, 'labels')
    train_dataset = tokenized_datasets["train"]
    print(next(iter(train_dataset)))
    eval_dataset = tokenized_datasets["validation"]
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )
    probed_model, train_dataloader, eval_dataloader = accelerator.prepare(
        probed_model, train_dataloader, eval_dataloader
    )
    completed_steps = 0
    probed_model.train()
    probe = probed_model.classifier.classifier
    cached_input_ids = []
    cached_z_primes = []
    for step, batch in enumerate(train_dataloader):
        batch['output_hidden_states'] = True
        with torch.no_grad():
            outputs = probed_model(**batch)
        hidden_states = outputs.hidden_states[-1].detach()  # Take last ones before the linear layer.
        # FIXME: not just last layer (see above)
        for targ_s in range(3):
            s_prime = torch.unsqueeze(torch.tensor(targ_s), 0).cuda()
            z_prime = gen_counterfactual(hidden_states, probe, s_prime)
            # We generate input ids and xfactual embeddings for the whole sequence, but one has to decide what parts to
            # add to the dataset. Regardless of the option, remember to shift!
            # Here's the whole sequence.
            squeezed_z = torch.squeeze(z_prime, dim=0)
            shifted_z = squeezed_z[:-1, :].contiguous()
            labels = batch['input_ids'].view((-1, 1))
            shifted_labels = labels[1:, ...].contiguous()
            cached_input_ids.append(shifted_labels)
            cached_z_primes.append(shifted_z)
            # Or one can focus only on specific tokens
            # Minus 2 for the penultimate token.
            # cached_input_ids.append(batch['input_ids'][0, -2].view((-1, 1)))
            # cached_z_primes.append(z_prime[0, -3].view(1, -1))  # Shift z-prime by one earlier to allow prediction.
            # Debugging tool prints out the token being added to the dataset
            # print("Decoding token", tokenizer.decode(cached_input_ids[-1][0]))
            completed_steps += 1
            if completed_steps >= args.max_train_steps:
                break
        if completed_steps >= args.max_train_steps:
            break
    stacked_z_primes = torch.vstack(cached_z_primes)
    stacked_input_ids = torch.vstack(cached_input_ids)
    xfact_dataset = torch.utils.data.TensorDataset(stacked_z_primes, stacked_input_ids)
    return xfact_dataset


def _xfact_training(args, probed_model, accelerator, xfact_dataset):
    # Just train the LM head on the xfact dataset, mapping from counterfactual embeddings to the desired next token id.
    loaded_data = DataLoader(xfact_dataset, shuffle=True)
    lm_head = probed_model.lm.lm_head
    optimizer = AdamW(lm_head.parameters(), lr=args.learning_rate)
    # Prepare everything with our `accelerator`.
    lm_head, optimizer, train_dataloader = accelerator.prepare(
        lm_head, optimizer, loaded_data
    )

    loss_fn = CrossEntropyLoss()
    for epoch in range(3):
        print("Epoch", epoch)
        lm_head.train()
        running_loss = 0
        for step, batch in enumerate(loaded_data):
            embedding, label = batch
            prediction = lm_head(embedding)
            loss = loss_fn(prediction, label.view(-1))
            running_loss += loss.detach().cpu().numpy().item()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        print("Loss:\t", running_loss / len(loaded_data))


def _measure_ppl(args, probed_model, tokenizer):
    probed_model.lm_mode = True
    probed_model.eval()
    probed_model.cuda()

    data = load_dataset(args.lm_dataset_name, data_files='suite.txt', split='train')
    def tokenize_function(examples):
        text_column_name = 'text'
        tokenized_batch = tokenizer(examples[text_column_name])
        return tokenized_batch
    encodings = data.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    # And parse out the specific words we want to have the indices of
    special_idxs = []
    with open(args.lm_dataset_name + '/surprisal_tokens.txt', 'r') as f:
        for word, sentence in zip(f, encodings['input_ids']):
            found_match = False
            for i, tok in enumerate(sentence):
                decoded = tokenizer.decode(tok).strip()
                if word.startswith(decoded):
                    special_idxs.append(i - 1)  # Shift by 1 to get the surprisal for prediction at the earlier step.
                    found_match = True
                    break
            assert found_match, "Could not find matching token for word " + word

    nlls = []
    for sentence, token_idx in zip(encodings['input_ids'], special_idxs):
        input_ids = torch.Tensor(sentence).cuda().long()
        with torch.no_grad():
            outputs = probed_model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
            # In addition to the overal surprisal, print out the surprisal for each token, which allows us to peek
            # into specific words.
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            surprisals = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print('surprisals', surprisals[token_idx])
        nlls.append(surprisals[token_idx].cpu().numpy())
    print(np.mean(nlls))


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    lm_raw_datasets = _load_dataset(args, args.lm_dataset_name, args.lm_dataset_config_name)
    # For imdb, use file
    # probe_raw_datasets = _load_dataset(args, args.probe_dataset_name, args.probe_dataset_config_name, use_file=True)
    # For wizard
    probe_raw_datasets = _load_dataset(args, args.probe_dataset_name, args.probe_dataset_config_name)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.model_name_or_path is not None:
        probed_model = torch.load(args.model_name_or_path + '/model.pt')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        if args.config_name:
            lm_config = AutoConfig.from_pretrained(args.config_name)
        elif args.lm_model_name_or_path:
            lm_config = AutoConfig.from_pretrained(args.lm_model_name_or_path)
        else:
            lm_config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.lm_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if args.lm_model_name_or_path:
            lm_model = AutoModelForCausalLM.from_pretrained(
                args.lm_model_name_or_path,
                from_tf=bool(".ckpt" in args.lm_model_name_or_path),
                config=lm_config,
            )
        else:
            logger.info("Training new model from scratch")
            lm_model = AutoModelForCausalLM.from_config(lm_config)

        lm_model.resize_token_embeddings(len(tokenizer))
        copied = copy.deepcopy(lm_config)
        copied.num_labels = 3  # FIXME. Works for Wizard, but not for example imdb
        classifier_model = GPT2ForSequenceClassification(copied)
        classifier_model.resize_token_embeddings(len(tokenizer))
        probed_model = GPT2ProbedCLM(lm_config, lm_model, classifier_model)
    if args.eval_only:
        for test_suite in ['data/mycal_gender_stereotypical', 'data/mycal_gender_counter']:
            args.lm_dataset_name = test_suite
            print("Test suite", test_suite)
            _measure_ppl(args, probed_model, tokenizer)
        return
    # Do the actual training
    for iteration in range(2):
        print("Iteration number", iteration)
        print("Training probe")
        probed_model.lm_mode = False
        _train_probe(args, probed_model, tokenizer, accelerator, probe_raw_datasets)
        accelerator.free_memory()
        print("Training LM")
        probed_model.lm_mode = True
        _train_lm(args, probed_model, tokenizer, accelerator, lm_raw_datasets)
        accelerator.free_memory()
        print("Generating xfacts")
        probed_model.lm_mode = False
        xfact_dataset = _gen_xfacts(args, probed_model, tokenizer, accelerator, probe_raw_datasets)
        print("Training with xfacts")
        _xfact_training(args, probed_model, accelerator, xfact_dataset)
        torch.cuda.empty_cache()
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(probed_model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        torch.save(unwrapped_model, args.output_dir + "/model.pt")


if __name__ == "__main__":
    main()
