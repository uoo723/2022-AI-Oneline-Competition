#!/usr/bin/env bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

./scripts/run_colbert.sh 0
./scripts/run_monobert.sh 0
