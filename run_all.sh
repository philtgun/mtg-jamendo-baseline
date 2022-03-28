#!/usr/bin/env bash
for SPLIT in {0..4} ; do
for SUBSET in autotagging autotagging_genre autotagging_instrument autotagging_moodtheme autotagging_top50tags ; do
    python -m src.train "$1" "$2" --subset "$SUBSET" --split "$SPLIT" --max_epochs 100 --num-workers 12 --gpus 1 --batch-size 32 --sampling random --output-path out/results.csv --models-dir out
done
done
