#!/usr/bin/env bash

export WANDB_MODE=online
snakemake train_and_evaluate -j 12 --rerun-incomplete
