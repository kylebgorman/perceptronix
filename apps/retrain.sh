#!/bin/bash
# Simple script to retrain all three models.
# This only works if you're me.

set -euo pipefail

python -m case_restorer -v \
    --train=../../perceptronix_data/wsj.horizontal \
    --write=case_restorer.pb &
python -m pos_tagger -v \
    --train=../../perceptronix_data/wsj.vertical_tagged \
    --write=pos_tagger.pb &
python -m sentence_tokenizer -v \
    --train=../../perceptronix_data/wsj.txt \
    --write=sentence_tokenizer.pb &
wait
