#!/usr/bin/env bash

snakemake train_and_evaluate --rulegraph | dot -Tpng > rulegraph.png
