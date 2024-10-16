#!/bin/bash

python inference.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \   # Model to use
    --top_k 10 \                                # Retrieve top 10 related items
    --test_mode False \                          # Run in test mode (no real inference)
    --result_dir ./results \                    # Directory to save results
    --inference_file ./data/test.pickle \       # Input file for inference
    --dtype int8 \                              # Use 8-bit precision
    --save_results True \                       # Save results if True
    --num_inference -1 \                         # Process 5 items (-1 for all)
    --use_tag "品牌" \                          # Tag to use    
    --use_qwen_api True                         # Use Qwen API for inference

    # --use_tag "系列名稱" "產品類型" "產品序號" "顏色" "材質" "對象與族群" "適用物體、事件與場所" "特殊主題" "形狀" "圖案" "尺寸" "重量" "容量" "包裝組合" "功能與規格"
