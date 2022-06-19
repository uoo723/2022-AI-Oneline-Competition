#!/usr/bin/env bash
set -e

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

PORT=${PORT:-5050}

mlflow ui -h 0.0.0.0 -p $PORT --backend-store-uri ./logs
