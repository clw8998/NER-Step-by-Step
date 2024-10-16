#!/bin/bash

python inference.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --top_k 10 \
    --result_dir ./results \
    --inference_file ./data/test.pickle \
    --dtype int8 \
    --num_inference -1 \
    --use_tag "品牌" \
    --use_qwen_api \
    --save_results
    # --test_mode
