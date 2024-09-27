# Language Models for Classical Philology


## Overview 
This repository contains the resources for our papers [Exploring Large Language Models for Classical Philology](https://aclanthology.org/2023.acl-long.846/) and [_Graecia capta ferum victorem cepit._ Detecting Latin Allusions to Ancient Greek Literature](https://aclanthology.org/2023.alp-1.4/).

By request, we provide simple pipelines for:
- [Part-of-Speech (PoS) Tagging](src/ancient-language-models/pos_tagging.py)
- [Lemmatization](src/ancient-language-models/lemmatization.py)
- (Unlabeled) [Dependency Parsing](src/ancient-language-models/unlabeled_parsing.py)

Please note that while the general setup is similar to what we used in our paper, this version focuses on readability and flexibility rather than being an exact replica.

## Installation
We use [pdm](https://pdm-project.org/en/latest/) for easy dependency management. To install the required packages
- Install `pdm`
- Run the following command:
    ```
    pdm install
    ```

This will take care of all the necessary dependencies.

## Downloading Treebanks
To download the [Universal Dependencies](https://universaldependencies.org/) treebanks, you can use the following commands:
```
wget -P data/ https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz
tar --extract --file data/ud-treebanks-v2.14.tgz -C data/
rm data/ud-treebanks-v2.14.tgz
```
This will download the treebanks, extract them into the [data/](data) directory, and remove the downloaded archive.

## Running a Script
In the [configs/](configs) directory, you will find default configurations for the three tasks. Feel free to adjust them to your needs or create new ones. To run a script, use the following command format:
```
pdm run python src/ancient-language-models/script_name.py configs/config-name.py
```
For example, to run the unlabeled parsing script:
```
pdm run python src/ancient-language-models/unlabeled_parsing.py configs/unlabeled_parsing-config.py
```
The scripts include small `wandb` sweeps, which you may want to extend or adjust according to your needs. For example, the Dependency Parsing script only computes uncorrected results, meaning there is no theoretical guarantee that the resulting tree will not contain cycles or multiple root nodes. To address this, you might consider adding an algorithm like Chu-Liu-Edmonds to ensure a valid tree structure.

## Language Models
|                 | Greek    | Latin   | Multilingual |
|-----------------|----------|---------|--------------|
| **Encoder-only**    | [GrεBERTa](https://huggingface.co/bowphs/GreBerta) | [LaBERTa](https://huggingface.co/bowphs/LaBerta) | [PhilBERTa](https://huggingface.co/bowphs/PhilBerta)    |
| **Encoder-decoder** | [GrεTa](https://huggingface.co/bowphs/GreTa)    | [LaTa](https://huggingface.co/bowphs/LaTa)    | [PhilTa](https://huggingface.co/bowphs/PhilTa)       |


## Sentence Transformers
In our paper [_Graecia capta ferum victorem cepit._ Detecting Latin Allusions to Ancient Greek Literature](https://aclanthology.org/2023.alp-1.4/), we introduce SPhilBERTa, a Sentence Transformer model to identify cross-lingual references between Latin and Ancient Greek texts. SPhilBERTa can be found [here](https://huggingface.co/bowphs/SPhilBerta).

## Contact
If you have any questions or problems, feel free to [reach out](mailto:riemenschneider@cl.uni-heidelberg.de).

## Citation
```bibtex
@inproceedings{riemenschneider-frank-2023-exploring,
    title = "Exploring Large Language Models for Classical Philology",
    author = "Riemenschneider, Frederick  and
      Frank, Anette",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.846",
    doi = "10.18653/v1/2023.acl-long.846",
    pages = "15181--15199",
}
```