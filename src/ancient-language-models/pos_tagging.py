#!/usr/bin/env python
# coding: utf-8

"""
PoS tagging script reading conllu data and a config.py file to run a wandb sweep.
"""

from functools import partial
import sys
import logging
import warnings


import evaluate
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Value,
    Features,
    Sequence,
)
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import wandb


from utils import prepare_data


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input text and aligns the labels with the tokenized output.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    tokenized_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_labels.append(label_ids)
    tokenized_inputs["labels"] = tokenized_labels
    return tokenized_inputs


def pos_processor(sentence):
    """
    Processes a sentence to extract tokens and their corresponding PoS labels.
    """
    sent_tokens = [token["form"] for token in sentence]
    sent_labels = [token["upos"] for token in sentence]
    return sent_tokens, sent_labels


def compute_metrics(p):
    """
    Computes common metrics to evaluate model performance.
    """

    predictions, label_ids = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, label_ids)
    ]
    true_labels = [
        [labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, label_ids)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


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
        model_config.label2id = label2id
        model_config.id2label = id2label
        model = AutoModelForTokenClassification.from_pretrained(
            run_config.model_name_or_path, config=model_config
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
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_train_epochs": 2,
            "weight_decay": 0.01,
            "push_to_hub": False,
            "metric_for_best_model": "f1",
            "load_best_model_at_end": True,
            "output_dir": output_dir,
            "run_name": "demorun",
            "report_to": "wandb",
            "learning_rate": run_config.learning_rate,
            "save_total_limit": 3,
        }
        args = TrainingArguments(**training_arguments)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()
        evaluation = trainer.evaluate()
        logging.info("Evaluation: %s", evaluation)
        trainer.save_model(output_dir)


warnings.filterwarnings(
    "ignore", message=".*seems not to be NE tag.*", category=UserWarning
)

config_file = sys.argv[1]
with open(config_file, encoding="utf-8") as f:
    cfg = f.read()
    print(cfg)
    exec(cfg)
logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)
wandb.login()

train_sentences, train_labels = prepare_data(train_file, pos_processor)
valid_sentences, valid_labels = prepare_data(valid_file, pos_processor)

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


labels = sorted(
    list({label for sentence in train_labels + valid_labels for label in sentence})
)
logging.info("%s labels: %s", len(labels), labels)


features = Features(
    {
        "tokens": Sequence(Value("string")),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels}, features=features
)
datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})
label2id = {label: i for i, label in enumerate(labels)}
id2label = dict(enumerate(labels))
metric = evaluate.load("seqeval")

sweep_id = wandb.sweep(sweep_config, project="pos-tagging")
wandb.agent(sweep_id, function=train)
