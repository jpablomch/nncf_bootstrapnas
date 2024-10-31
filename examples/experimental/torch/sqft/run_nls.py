# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import datasets
import torch
import transformers
from datasets import Dataset
from datasets import load_dataset
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

from nncf import NNCFConfig
from nncf.common.utils.os import safe_open

logger = logging.getLogger(__name__)

try:
    from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
        create_compressed_model_from_algo_names,
    )
    from nncf.torch.model_creation import create_nncf_network

    is_nncf_available = True
except ImportError:
    is_nncf_available = False
    logger.info("NNCF is not installed. Please install it.")


def parse_nncf_config(nncf_config_path, num_hidden_layers=1, search_space=None, learning_rate=3e-4, num_epochs=3):
    """Parse and preprocess the NNCF configuration file and add learning rate and epochs.

    Args:
        nncf_config_path (str): Path to the NNCF configuration file.
        num_hidden_layers (int): Number of hidden layers to consider for the search space.
        search_space (list, optional): List of search space widths. Defaults to None.
        learning_rate (float): The initial learning rate to set. Defaults to 3e-4.
        num_epochs (int): The number of epochs to set. Defaults to 3.

    Returns:
        dict: The preprocessed and updated NNCF configuration.
    """
    with safe_open(Path(nncf_config_path)) as file:
        loaded_json = json.load(file)

    base_overwrite_groups = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"]
    base_overwrite_groups_widths = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"][
        "overwrite_groups_widths"
    ]
    overwrite_groups, overwrite_groups_widths = [], []
    for group, width in zip(base_overwrite_groups, base_overwrite_groups_widths):
        current_search_space = width if search_space is None else search_space
        if group[0].startswith("{re}"):
            new_group = [
                [item.replace("{re}", "").replace("{*}", str(i)) for item in group] for i in range(num_hidden_layers)
            ]
            new_width = [current_search_space for _ in range(num_hidden_layers)]
        else:
            new_group = [group]
            new_width = [current_search_space]
        overwrite_groups.extend(new_group)
        overwrite_groups_widths.extend(new_width)

    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"] = overwrite_groups
    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups_widths"] = overwrite_groups_widths

    # Add learning rate and epochs to the configuration
    loaded_json["bootstrapNAS"]["training"]["schedule"] = {
        "list_stage_descriptions": [
            {
                "train_dims": ["width"],
                "width_indicator": max([len(widths) for widths in overwrite_groups_widths]),
                "init_lr": learning_rate,
                "epochs": num_epochs,
                "epochs_lr": num_epochs,
            }
        ]
    }

    return loaded_json


def load_nncf_config(nncf_config_path, learning_rate=3e-4, num_epochs=3, num_hidden_layers=32, search_space=None):
    """Load and preprocess the NNCF configuration file.

    Args:
        nncf_config_path (str): Path to the NNCF configuration file.
        learning_rate (float): The initial learning rate to set.
        num_epochs (int): The number of epochs to set.
        num_hidden_layers (int): Number of hidden layers to consider for the search space.
        search_space (str, optional): Comma-separated string of search space widths. Defaults to None.

    Returns:
        NNCFConfig: The preprocessed NNCF configuration object.
    """
    if search_space is not None:
        search_space = [int(width) for width in search_space.split(",")]
    loaded_json = parse_nncf_config(
        nncf_config_path,
        num_hidden_layers=num_hidden_layers,
        search_space=search_space,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    nncf_config = NNCFConfig.from_dict(loaded_json)
    return nncf_config


@dataclass
class NeuralLoraSearchTrainingArguments(TrainingArguments):
    """
    Arguments for Neural Lora Search training, including LoRA and NNCF configurations.
    """

    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(
        default="q_proj,v_proj", metadata={"help": "The layers where LoRA adapters will be placed."}
    )
    nncf_config: str = field(
        default=None, metadata={"help": "NNCF configuration .json file for compression-enabled training"}
    )
    search_space: str = field(default=None, metadata={"help": "Low-rank search space of NLS training."})
    padding_size: str = field(default="right", metadata={"help": "Padding size for tokenization."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data used for training and evaluation.
    """

    val_set_size: int = field(default=0, metadata={"help": "Validation set size."})
    cutoff_len: int = field(default=256, metadata={"help": "Cutoff length for tokenization."})


@dataclass
class ModelArguments:
    """
    Base model/tokenizer that will be fine-tuned.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to the pre-trained model or model identifier from huggingface.co/models"}
    )
    dtype: str = field(default="float16", metadata={"help": "Data type for the model."})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    non_quant_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NeuralLoraSearchTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary
    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
            f"16-bits training: {training_args.fp16}"
        )
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=model_args.dtype,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    logger.info("Adding LoRA modules...")
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=training_args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    nncf_config = None
    compression_ctrl = None
    if training_args.nncf_config is not None:
        if not is_nncf_available:
            raise ImportError("NNCF is not installed. Please install it.")
        nncf_config = load_nncf_config(
            training_args.nncf_config,
            learning_rate=training_args.learning_rate,
            num_epochs=training_args.num_train_epochs,
            num_hidden_layers=model.config.num_hidden_layers,
            search_space=training_args.search_space,
        )

        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir

        if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
            os.makedirs(nncf_config["log_dir"])

        if nncf_config is not None:
            nncf_network = create_nncf_network(model, nncf_config)
            algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "neural_lora_search")
            compression_ctrl, model = create_compressed_model_from_algo_names(
                nncf_network, nncf_config, algo_names=[algo_name]
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = training_args.padding_size

    # Load data
    def tokenize(prompt, add_eos_token=True):
        """
        Tokenize the given prompt.
        """
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < data_args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """
        Generate and tokenize the prompt from the data point.
        """
        full_prompt = data_point["full_prompt"]
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    def add_prompt_func_gsm8k(document):
        """Add prompt to GSM8K dataset document.

        Args:
            document (dict): A dictionary containing the GSM8K dataset document.

        Returns:
            dict: The document with the added prompt.
        """

        def document_to_text(document):
            return f"Question: {document['question']}\nAnswer:"

        prompt = document_to_text(document)
        answer = document["answer"]
        document["full_prompt"] = prompt + " " + answer
        return document

    def load_gsm8k_dataset(split="train"):
        """Load the GSM8K dataset and add prompts.

        Args:
            split (str): The dataset split to load (default is "train").
            debug (bool): Whether to load only a subset of the data for debugging (default is False).

        Returns:
            Dataset: The GSM8K dataset with prompts.
        """
        gsm8k_dataset = load_dataset("gsm8k", "main", split=split)
        gsm8k_dataset = gsm8k_dataset.map(add_prompt_func_gsm8k)

        full_prompts = gsm8k_dataset["full_prompt"]
        dataset = Dataset.from_dict({"full_prompt": full_prompts})
        return dataset

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = load_gsm8k_dataset(split="train")
        validation_set_size = data_args.val_set_size
        if validation_set_size > 0:
            train_val = train_dataset.train_test_split(test_size=validation_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compression_ctrl=compression_ctrl,
    )
    model.config.use_cache = False

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
