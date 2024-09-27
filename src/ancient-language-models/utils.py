#!/usr/bin/env python
# coding: utf-8

"""
General helper functions.
"""

from conllu import parse


def prepare_data(file_path, sentence_processor):
    """
    Reads a conllu file and returns the content sentences as well as their labels.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    sentences, labels = [], []
    for sentence in parse(data):
        sent_tokens, sent_labels = sentence_processor(sentence)
        sentences.append(sent_tokens)
        labels.append(sent_labels)

    return sentences, labels
