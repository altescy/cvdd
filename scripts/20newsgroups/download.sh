#!/bin/bash

set -e

mkdir -p data/raw
cd data/raw

DATASET_URL=http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
DATASET_PATH=20news-19997.tar.gz

curl $DATASET_URL --output $DATASET_PATH
tar xvf $DATASET_PATH
