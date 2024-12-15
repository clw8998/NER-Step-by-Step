import torch
import os
import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import get_topk_items
from opencc import OpenCC
import pickle
import random
from gradio_client import Client

import warnings
warnings.filterwarnings("ignore")

t2s = OpenCC('t2s')
s2t = OpenCC('s2t')

# Download the dataset if it does not exist
import get_dataset  # This import itself will download the dataset, DO NOT REMOVE

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with various configurations")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--test_mode', action='store_true', help="Whether to run in test mode or not")
    parser.add_argument('--result_dir', type=str, default="./results", help="Directory to save results")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'int4', 'fp16', 'bf16', 'fp32'], help="Data type for model precision")
    parser.add_argument('--save_results', action='store_true', help="Whether to save inference results")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--temerature', type=float, default=1e-5, help="Temperature for generation")
    parser.add_argument('--use_qwen_api', type=bool, default=False, help="Whether to use Qwen API for inference")
    parser.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode")
    parser.add_argument('--corpus', type=str, default='coupang', choices=['coupang', 'pchome', 'pchome_train_11000'], help="Corpus to use for inference")
    parser.add_argument('--rag_corpus', type=bool, default=False, help="Whether to use RAG for corpus retrieval")
    parser.add_argument('--rag_error', type=bool, default=False, help="Whether to use RAG for error cases retrieval")
    parser.add_argument('--prompt_type', type=int, default=0, choices=[0, 1, 2], help="Prompt type to use for inference")

    return parser.parse_args()

def get_related_items(current_item_names: str, corpus_name: str,  top_k: int = 5):
    related_items, _ = get_topk_items.tf_idf(current_item_names, corpus_name, top_k=top_k)
    return related_items

def run_instructions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                     prompts: list[str], system_message: str = None, temperature: float = 1e-5,
                     test_mode: bool = True, use_qwen_api: bool = False):
    messages = []
    print('\n\n========== Start Conversation ==========')
    if system_message:
        print('---------- System Message ----------')
        system_message = t2s.convert(system_message)
        messages.append({"role": "system", "content": system_message})
        print(system_message)
    
    if use_qwen_api:
        history = []
        client = Client("Qwen/Qwen2.5")

        for i in range(len(prompts)):
            prompts[i] = t2s.convert(prompts[i])
            print(prompts[i])

            if not test_mode:
                response = client.predict(
                    query=prompts[i],
                    history=history,
                    system=system_message,
                    radio='72B',
                    api_name="/model_chat_1"
                )
                history = response[1]
            else:
                response = ["This is a placeholder response.", "This is a placeholder response."]
            
            print(response[1])
        print('========== End Conversation ==========')
        return response[1]
    
    for i in range(len(prompts)):
        print(f'---------- Instruction {i} ----------')
        prompts[i] = t2s.convert(prompts[i])
        messages.append({"role": "user", "content": prompts[i]})
        print(prompts[i])

        print(f'---------- Response {i} ----------')
        if not test_mode:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=temperature)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            response = "This is a placeholder response."
        messages.append({"role": "assistant", "content": response})
        print(response)
    print('========== End Conversation ==========')
    return messages

def main():
    args = parse_args()
    print(args)

    # If use_qwen_api is set to True, skip model and tokenizer loading
    if not args.use_qwen_api:
        # Load model according to dtype
        if args.dtype == 'int8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif args.dtype == 'int4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif args.dtype == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif args.dtype == 'bf16':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        model, tokenizer = None, None  # If using API, no local model

    # Load TF-IDF model
    tfidf_model = get_topk_items.TFIDFModel(args.corpus)
    tfidf_model.initialize()

    # Load inference data
    with open(args.inference_file, 'rb') as file:
        pc_test_data = pickle.load(file)

    p_names = [item['context'] for item in pc_test_data]
    # drop duplicates
    p_names = sorted(set(p_names))

    # Control number of inference items
    if args.num_inference != -1:
        p_names = p_names[:args.num_inference]

    # 商品匹配用的 prompt 模板
    if args.prompt_type == 0:
        prompts_t = [
            '详细了解以下商品，尽可能辨认出你认识的所有关键词，并解释。\n{item_1}',
            '详细了解以下商品，尽可能辨认出你认识的所有关键词，并解释。\n{item_2}',
            '使用以上你对两个商品的分析结果，对这两个商品做商品匹配，判断他们是否相同。',
            '总结以上匹配结果，这两个商品是否相同? 仅须回答「是」或「否」。',
        ]

    system_message_t = "你是一位熟悉電子商務的助手"

    if args.rag_corpus:
        system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"

    if args.rag_error:
        system_message_t += "以下是你曾经预测错误的案例，请参考以下内容，避免再犯相同的错误：\n{error_corpus}"

    if args.rag_corpus & args.rag_error:
        system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"
        system_message_t += "另外以下是你曾经预测错误的案例，请参考以下内容，避免再犯相同的错误：\n{error_corpus}"

    # 若為互動模式，讓使用者自行輸入兩個商品名稱
    if args.interactive:
        while True:
            p_name_1 = input("Enter product name_1: ")
            p_name_2 = input("Enter product name_2: ")

            # 建立語料 (可依需求保留或省略)
            corpus_1 = '\n'.join(tfidf_model.query(p_name_1, top_k=args.top_k))
            corpus_2 = '\n'.join(tfidf_model.query(p_name_2, top_k=args.top_k))
            # 將兩個商品的相關資訊合併
            corpus = corpus_1 + '\n' + corpus_2
            system_message = system_message_t.format(corpus=corpus)

            prompts = [
                prompts_t[0].format(item_1=p_name_1),
                prompts_t[1].format(item_2=p_name_2),
                prompts_t[2],
                prompts_t[3]
            ]

            messages = run_instructions(
                model, tokenizer, prompts, system_message, args.temerature,
                test_mode=args.test_mode, use_qwen_api=args.use_qwen_api
            )

    else:
        # 非互動模式，可自行從 p_names 中選取 pair 進行匹配
        # 例如，隨機選取成對商品
        if len(p_names) >= 2:
            for i in range(0, len(p_names), 2):
                if i+1 >= len(p_names):
                    break
                p_name_1 = p_names[i]
                p_name_2 = p_names[i+1]

                corpus_1 = '\n'.join(tfidf_model.query(p_name_1, top_k=args.top_k))
                corpus_2 = '\n'.join(tfidf_model.query(p_name_2, top_k=args.top_k))
                corpus = corpus_1 + '\n' + corpus_2
                system_message = system_message_t.format(corpus=corpus)

                prompts = [
                    prompts_t[0].format(item_1=p_name_1),
                    prompts_t[1].format(item_2=p_name_2),
                    prompts_t[2],
                    prompts_t[3]
                ]

                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=args.test_mode, use_qwen_api=args.use_qwen_api
                )

                # Save messages only if save_results is True
                if args.save_results:
                    os.makedirs(args.result_dir, exist_ok=True)
                    with open(f'{args.result_dir}/{i}_match.pkl', 'wb') as file:
                        messages.insert(0, {'product_1': p_name_1, 'product_2': p_name_2})
                        pickle.dump(messages, file)

if __name__ == "__main__":
    main()
