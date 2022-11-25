#!/usr/bin/env bash
docker run -t -a STDOUT -v $(pwd):/opt/project -e SYNAPSE_AUTH_TOKEN fabfroehlich/generic_parameter_estimation:main /bin/bash -c "cd opt/project;snakemake train_and_evaluate -j 2"