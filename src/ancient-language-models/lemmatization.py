#!/usr/bin/env python
# coding: utf-8

"""
Lemmatization script reading conllu data and a config.py file to run a wandb sweep.
"""

from functools import partial
import sys
import logging

import evaluate
from datasets import (
    Dataset,
    DatasetDict,
    Value,
    Features,
)
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import wandb


from utils import prepare_data


def lemma_processor(sentence, context_size=2):
    """
    Processes a sentence to extract tokens in their context (the context_size preceding
    and following tokens) and their lemmata.
    """
    sent_tokens = [token["form"] for token in sentence]
    sent_labels = [token["lemma"] for token in sentence]
    sent_contexts = []
    for idx, token in enumerate(sent_tokens):
        preceding_words = sent_tokens[max(0, idx - context_size) : idx]
        following_words = sent_tokens[
            idx + 1 : min(len(sent_tokens), idx + context_size + 1)
        ]
        token_context = (
            preceding_words
            + [f"<special_token_0> {token} <special_token_1>"]
            + following_words
        )
        sent_contexts.append(" ".join(token_context))
    return sent_contexts, sent_labels


def flatten(outer):
    """
    Flattens a nested list.
    """
    return [elem for inner in outer for elem in inner]


def compute_metrics(eval_pred, tokenizer):
    """
    Computes normalized exact match score for lemmatization.
    """
    eval_predictions, eval_labels = eval_pred
    decoded_preds = tokenizer.batch_decode(eval_predictions, skip_special_tokens=True)
    eval_labels = np.where(eval_labels != -100, eval_labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(eval_labels, skip_special_tokens=True)
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"exact_match": result["exact_match"]}
    return {k: round(v, 4) for k, v in result.items()}


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input text and aligns the labels with the tokenized output.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, max_length=512
    )  # is_split_into_words=True,
    with tokenizer.as_target_tokenizer():
        tokenized_labels = tokenizer(
            examples["labels"], max_length=512, truncation=True
        )
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs


config_file = sys.argv[1]
with open(config_file, encoding="utf-8") as f:
    cfg = f.read()
    print(cfg)
    exec(cfg)

logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)
wandb.login()

train_sentences, train_labels = prepare_data(train_file, lemma_processor)
valid_sentences, valid_labels = prepare_data(valid_file, lemma_processor)
train_sentences, train_labels = flatten(train_sentences), flatten(train_labels)
valid_sentences, valid_labels = flatten(valid_sentences), flatten(valid_labels)

logging.info(
    "train: %s contexts and labels, example: %s -> %s",
    len(train_sentences),
    train_sentences[0],
    train_labels[0],
)
logging.info(
    "valid: %s contexts and labels, example: %s -> %s",
    len(valid_sentences),
    valid_sentences[0],
    valid_labels[0],
)

features = Features(
    {
        "tokens": Value("string"),
        "labels": Value(dtype="string", id=None),
    }
)
train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels}, features=features
)
datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
metric = evaluate.load("exact_match")
labels = datasets["train"].features["labels"]


def train(config=None):
    """
    Main method to run a wandb sweep.
    """
    with wandb.init(config=config) as run:
        run_config = run.config
        tokenizer = AutoTokenizer.from_pretrained(
            run_config.model_name_or_path,
            add_space_prefix=True,
            clean_up_tokenization_spaces=True,
        )
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

        model = AutoModelForSeq2SeqLM.from_pretrained(run_config.model_name_or_path)
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
            "weight_decay": 0.01,
            "push_to_hub": False,
            "metric_for_best_model": "exact_match",
            "load_best_model_at_end": True,
            "output_dir": output_dir,
            "run_name": run_config.run_name,
            "report_to": "wandb",
            "learning_rate": run_config.learning_rate,
            "save_total_limit": 3,
            "predict_with_generate": run_config.predict_with_generate,
            "generation_max_length": run_config.generation_max_length,
            "generation_num_beams": run_config.generation_num_beams,
        }
        args = Seq2SeqTrainingArguments(**training_arguments)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id
        )
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()
        evaluation = trainer.evaluate()
        logging.info("Evaluation: %s", evaluation)
        trainer.save_model(output_dir)


sweep_id = wandb.sweep(sweep_config, project="lemmatization")
wandb.agent(sweep_id, function=train)
