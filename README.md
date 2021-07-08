CVDD: Context Vector Data Description
=====================================

[![Actions Status](https://github.com/altescy/cvdd/workflows/CI/badge.svg)](https://github.com/altescy/cvdd/actions?query=workflow%3ACI)
[![License](https://img.shields.io/github/license/altescy/cvdd)](https://github.com/altescy/cvdd/blob/master/LICENSE)

This repository provides AllenNLP implementation of [Context Vector Data Description (CVDD)](https://github.com/lukasruff/CVDD-PyTorch).

```
❯ git clone https://github.com/altescy/cvdd.git
❯ cd cvdd
❯ poetry install
❯ make datasets
❯ poetry run allennlp train-with-mlflow configs/cvdd_20newsgroups.jsonnet
```
