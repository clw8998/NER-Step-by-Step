#!/bin/bash

python inference_ner_api.py \
    --openrouter_api_key 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 \
    --model_name qwen/qwen3-235b-a22b:free \
    --inference_file ./data/test.pickle \
    --result_dir ./results_test_1500_brand_qwen3-235b-a22b:free \
    --use_tag 品牌 \
    --save_results