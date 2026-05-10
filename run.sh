#!/bin/bash
pie-serve run /home/ec2327/pie/asyncLM/src/lib2.rs \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  -- --prompt "$1" --max-tokens 1024