#!/usr/bin/env bash

export DEBUG=true
export WANDB_MODE=online
snakemake --unlock
snakemake train_and_evaluate -j 10 --rerun-incomplete
