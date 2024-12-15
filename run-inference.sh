#!/bin/bash

# python inference_ner.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --top_k 10 \
#     --save_results \
#     --result_dir ./results_train_11000_brand_prompt_0_7B_fp16 \
#     --inference_file ./data/train.pickle \
#     --dtype fp16 \
#     --num_inference -1 \
#     --use_tag "品牌" \
#     --prompt_type 0 \
#     --rag_corpus True \
#     -i
#     # --corpus pchome \
#     # --rag_corpus True \
#     # --use_qwen_api True \
#     # --test_mode


# sleep 10



python inference_product_matching.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --top_k 10 \
    --save_results \
    --result_dir ./results_train_11000_brand_prompt_0_7B_fp16 \
    --inference_file ./data/train.pickle \
    --dtype fp16 \
    --num_inference -1 \
    --prompt_type 0 \
    --rag_corpus True \
    -i
    # --corpus pchome \
    # --rag_corpus True \
    # --use_qwen_api True \
    # --test_mode
