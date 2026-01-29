#!/bin/bash

vllm serve LGAI-EXAONE/EXAONE-4.0-1.2B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
