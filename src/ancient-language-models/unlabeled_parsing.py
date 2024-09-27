#!/usr/bin/env python
# coding: utf-8

"""
Unlabeled dependency parsing script reading conllu data and a config.py file to run a wandb sweep.
"""

from functools import partial
import sys
import logging


from datasets import (
    Dataset,
    DatasetDict,
    Value,
    Features,
    Sequence,
)
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import wandb


from models import DependencyRobertaForTokenClassification
from utils import prepare_data


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input text and aligns the dependency head labels with the tokenized output.
    Adjusts head IDs to point to the correct subword indices, mapping root tokens to <bos>.
    Corrected indexing to handle 1-based head indices.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        return_special_tokens_mask=True,
    )

    batch_size = len(examples["tokens"])
    tokenized_head_ids = []

    for i in range(batch_size):
        head_ids = examples["labels"][i]

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_idx_to_token_idx = {}
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                word_idx_to_token_idx[word_idx] = token_idx
            previous_word_idx = word_idx
        bos_token_idx = 0
        adjusted_head_ids = []
        for word_idx, head_word_idx in enumerate(head_ids):
            token_idx = word_idx_to_token_idx.get(word_idx, None)
            if token_idx is None:
                continue
            if head_word_idx == 0:
                head_token_idx = bos_token_idx
            else:
                adjusted_head_word_idx = head_word_idx - 1
                head_token_idx = word_idx_to_token_idx[adjusted_head_word_idx]

            adjusted_head_ids.append((token_idx, head_token_idx))
        labels = [-100] * len(word_ids)
        for token_idx, head_token_idx in adjusted_head_ids:
            labels[token_idx] = head_token_idx

        tokenized_head_ids.append(labels)

    tokenized_inputs["labels"] = tokenized_head_ids
    return tokenized_inputs


def compute_metrics(eval_pred):
    """
    Computes metrics, ignoring padded positions.
    """

    logits, labels = eval_pred
    max_seq_length = logits.shape[1]
    if labels.shape[1] < max_seq_length:
        padding_size = max_seq_length - labels.shape[1]
        labels = torch.nn.functional.pad(
            torch.tensor(labels), (0, padding_size), value=-100
        )
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    mask = labels != -100
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]
    accuracy = accuracy_score(valid_labels, valid_predictions)
    return {"accuracy": accuracy}


def head_processor(sentence):
    """
    Processes a sentence to extract tokens and their corresponding heads.
    """
    sent_tokens = [token["form"] for token in sentence]
    sent_labels = [token["head"] for token in sentence]
    return sent_tokens, sent_labels


def preprocess_logits_for_metrics(logits, labels):
    """
    Pads logits to ensure consistent shape across batch and sequence dimensions.
    """
    max_seq_length = 512
    current_seq_length = logits.shape[1]
    if current_seq_length < max_seq_length:
        padding_size = max_seq_length - current_seq_length
        logits = torch.nn.functional.pad(
            logits, (0, padding_size, 0, padding_size), value=float("-inf")
        )
    return logits


def train(config=None):
    """
    Main method to run a wandb sweep.
    """

    with wandb.init(config=config) as run:
        run_config = run.config
        tokenizer = AutoTokenizer.from_pretrained(
            run_config.model_name_or_path,
            add_prefix_space=True,
            truncation=True,
            clean_up_tokenization_spaces=True,
        )
        model_config = AutoConfig.from_pretrained(run_config.model_name_or_path)
        model = DependencyRobertaForTokenClassification.from_pretrained(
            run_config.model_name_or_path,
            config=model_config,
        )
        logging.info(
            "Loaded model %s. Number of parameters: %s.",
            run_config.model_name_or_path,
            model.num_parameters(),
        )
        tokenized_datasets = datasets.map(
            partial(
                tokenize_and_align_labels,
                tokenizer=tokenizer,
            ),
            batched=True,
        )
        logging.info("Tokenized_datasets: %s", tokenized_datasets)

        training_arguments = {
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "per_device_train_batch_size": run_config.per_device_train_batch_size,
            "per_device_eval_batch_size": run_config.per_device_eval_batch_size,
            "num_train_epochs": run_config.num_train_epochs,
            "weight_decay": run_config.weight_decay,
            "push_to_hub": False,
            "metric_for_best_model": "accuracy",
            "load_best_model_at_end": True,
            "output_dir": output_dir,
            "run_name": "demorun",
            "report_to": "wandb",
            "learning_rate": run_config.learning_rate,
            "save_total_limit": 3,
        }
        args = TrainingArguments(**training_arguments)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        trainer.train()
        evaluation = trainer.evaluate()
        logging.info("Evaluation: %s", evaluation)
        trainer.save_model(output_dir)


config_file = sys.argv[1]
with open(config_file, encoding="utf-8") as f:
    cfg = f.read()
    print(cfg)
    exec(cfg)

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)
wandb.login()

train_sentences, train_labels = prepare_data(train_file, head_processor)
valid_sentences, valid_labels = prepare_data(valid_file, head_processor)

logging.info(
    "train: %s sentences, %s labels, example: %s -> %s",
    len(train_sentences),
    len(train_labels),
    train_sentences[0],
    train_labels[0],
)
logging.info(
    "valid: %s sentences, %s labels, example: %s -> %s",
    len(valid_sentences),
    len(valid_labels),
    valid_sentences[0],
    valid_labels[0],
)
features = Features(
    {
        "tokens": Sequence(Value("string")),
        "labels": Sequence(Value("uint32")),
    }
)

train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels}, features=features
)
datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
sweep_id = wandb.sweep(sweep_config, project="unlabeled-parsing")
wandb.agent(sweep_id, function=train)
