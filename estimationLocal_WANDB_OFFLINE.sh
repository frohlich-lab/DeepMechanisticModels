#!/usr/bin/env bash

export WANDB_MODE=offline
snakemake train_and_evaluate -j 10 --rerun-incomplete
