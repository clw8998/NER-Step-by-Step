# Import libraries
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

# Define argparse for handling arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with various configurations")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--test_mode', action='store_true', help="Whether to run in test mode or not")
    parser.add_argument('--result_dir', type=str, default="./results", help="Directory to save results")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'int4'], help="Data type for model precision (int8 or int4)")
    parser.add_argument('--save_results', action='store_true', help="Whether to save inference results")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--use_tag', type=str, nargs='*', default=[], help="Tags to use during inference.")
    parser.add_argument('--temerature', type=float, default=1e-5, help="Temperature for generation")
    parser.add_argument('--use_qwen_api', type=bool, default=False, help="Whether to use Qwen API for inference")
    parser.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode")

    return parser.parse_args()

# Load items dataset
def load_items(path: str):
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, file_name)) for file_name in file_names]
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

# Declare retrieval function
def get_related_items(current_item_names: str, items_dataset: pd.DataFrame, top_k: int = 5):
    related_items, _ = get_topk_items.tf_idf(current_item_names, top_k=top_k)
    return related_items

# Declare multi-prompts inference function
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
                response = "This is a placeholder response."
            
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

# Main function for inference
def main():
    args = parse_args()

    # If use_qwen_api is set to True, skip model and tokenizer loading
    if not args.use_qwen_api:
        # Conditionally define BitsAndBytesConfig based on dtype
        if args.dtype == 'int8':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,          
                llm_int8_threshold=6.0,     
                llm_int8_enable_fp32_cpu_offload=True  
            )
        elif args.dtype == 'int4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  
                bnb_4bit_use_double_quant=True,  
                bnb_4bit_quant_type="nf4",  
                bnb_4bit_compute_dtype=torch.float16 
            )

        # Load the model with the BitsAndBytes configuration
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,  
            device_map="auto",               
            torch_dtype=torch.float16        
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        model, tokenizer = None, None  # Set these to None if using API

    # Load items dataset
    items_dataset = load_items('random_samples_1M')

    # Load inference data
    with open(args.inference_file, 'rb') as file:
        pc_test_data = pickle.load(file)

    p_names = [item['context'] for item in pc_test_data]
    random.shuffle(p_names)

    # Control number of inference items
    if args.num_inference != -1:
        p_names = p_names[:args.num_inference]

    annotation_rules = {
        "品牌": '商品品牌名称，指具体标识商品或服务来源的品牌或厂牌名称，如华硕、LG、Apple、Sony 等，涵盖科技产品、家电、服装、汽车等各行业的品牌名称。不包括电商平台中的店铺名称、卖场名称或个人卖家名称。',
        "系列": '商品中的所有产品系列，以及商家为特殊目的特别额外创造的商品名称，不包括特殊主题或产品类型，不含广告词。例如 iPhone 12、ROG 3060Ti。',
        "IP": '与特定知名IP（如漫画、电影、文学作品）或其创作者相关的产品或创作。例子包括知名作品（如《海贼王》）或直接与创作者本人（如J.K. Rowling）相关的内容。此定义不包括公司、发行商或经纪公司等机构。',
        "类型": '具体产品名称（产品类型）。例如：电脑、滑鼠、键盘、萤幕、玩具、饼干、卫生纸等实体产品。',
        # ... Add more tags here
    }

    system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"
    prompts_t = [
        '详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。\n{item}',
        '对该商品名称作命名实体识别（NER），找出目标实体。请注意，目标实体可能不存在于商品名称中。\n目标实体定义如下：\n{entity_definition}\n\n請簡短回答。',
        '根据以上信息，输出格式化的命名实体识别结果。\n请只输出命名实体识别结果，不要包含任何其他信息。\n以下範例：\n@@樂高## Art 31213 蒙娜麗莎\n20公升電子式微波爐 @@Whirlpool## AKM2064ES\n@@MUJI## 橡木組合收納櫃/抽屜/4段寬37*深28*高37 cm @@無印良品##\n@@AMANDA##@@艾曼达## 泳装 连身三角-黑魅-17101附帽\nLED壁掛式緊急照明燈 高亮度 台灣製造',
    ]

    if args.interactive:
        while True:
            p_name = input("Enter product name: ")
            corpus = '\n'.join(get_related_items(p_name, items_dataset, top_k=args.top_k))
            system_message = system_message_t.format(corpus=corpus)
            for tag, definition in annotation_rules.items():
                if tag not in args.use_tag:
                    continue
                prompts = [prompt.format(item=p_name, entity_definition=definition) for prompt in prompts_t]
                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=args.test_mode, use_qwen_api=args.use_qwen_api
                )
    else:
        for i, p_name in enumerate(p_names):
            corpus = '\n'.join(get_related_items(p_name, items_dataset, top_k=args.top_k))
            system_message = system_message_t.format(corpus=corpus)
            for tag, definition in annotation_rules.items():
                if tag not in args.use_tag:
                    continue
                # if not args.save_results or os.path.exists(f'{args.result_dir}/{i}_{tag}.pkl'):
                #     continue
                prompts = [prompt.format(item=p_name, entity_definition=definition) for prompt in prompts_t]

                # Call the inference function, passing None if using API
                messages = run_instructions(
                    model, tokenizer, prompts, system_message, args.temerature,
                    test_mode=args.test_mode, use_qwen_api=args.use_qwen_api
                )

                # Save messages only if save_results is True
                if args.save_results:
                    os.makedirs(args.result_dir, exist_ok=True)
                    with open(f'{args.result_dir}/{i}_{tag}.pkl', 'wb') as file:
                        pickle.dump(messages, file)

if __name__ == "__main__":
    main()











# old: 

# #%%
# # Import libraries
# import torch
# import os
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# # from opencc import OpenCC

# #%%
# # Download the dataset if it does not exist
# import get_dataset # This import itself will download the dataset, DO NOT REMOVE

# #%%
# # Define the model name
# model_name = "Qwen/Qwen2.5-7B-Instruct"

# # # Define the 8-bit configuration using BitsAndBytesConfig
# # bnb_config = BitsAndBytesConfig(
# #     load_in_8bit=True,          # Enable 8-bit quantization
# #     llm_int8_threshold=6.0,     # Threshold for mixed 8-bit precision (optional)
# #     llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for weights in fp32 (optional)
# # )

# # Define bitsandbytes configuration for loading in 4-bit or 8-bit precision
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # You can also set this to load_in_8bit=True for 8-bit quantization
#     bnb_4bit_use_double_quant=True,  # Optional: Enables double quantization for more efficiency
#     bnb_4bit_quant_type="nf4",  # Set the quantization type, options are 'fp4' or 'nf4'
#     bnb_4bit_compute_dtype=torch.float16  # Default is float32, set this to torch.float16 for reduced precision
# )

# # Load the model with the bitsandbytes config for 8-bit precision
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,  # Apply the 8-bit configuration
#     device_map="auto",               # Automatically map model to available devices
#     torch_dtype=torch.float16        # Optional: Use float16 precision where applicable
# )

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# #%%
# # Load items dataset
# def load_items(path: str):
#     # List all CSV files in the directory
#     file_names = [f for f in os.listdir(path) if f.endswith('.csv')]

#     # Load each CSV and concatenate them into a single DataFrame
#     df_list = []
#     for file_name in file_names:
#         df = pd.read_csv(os.path.join(path, file_name))
#         df_list.append(df)

#     df_all = pd.concat(df_list, ignore_index=True)
#     return df_all

# items_dataset = load_items('random_samples_1M')

# #%%
# # Declare retrieval function.
# def get_related_items(current_item_names: str, items_dataset: pd.DataFrame, top_k: int = 5):
#     # Randomly sample items (pseudo retrieval)
#     related_items = items_dataset.sample(top_k)
#     related_items = related_items['product_name'].tolist()

#     return related_items

# #%%
# # Declare multi-prompts inference function
# def run_instructions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
#                      prompts: list[str], system_message: str = None):
#     # Initialize the conversation
#     messages = []

#     print('========== Start Conversation ==========')
#     if system_message:
#         print('---------- System Message ----------')
#         messages.append({"role": "system", "content": system_message})
#         print(system_message)
    
#     for i in range(len(prompts)):
#         # Add user prompt to the conversation messages
#         print(f'---------- Instruction {i} ----------')
#         messages.append({"role": "user", "content": prompts[i]})
#         print(prompts[i])

#         # Tokenize the messages
#         text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#         # Generate the output
#         generated_ids = model.generate(**model_inputs, max_new_tokens=512)
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]

#         # Decode the generated output
#         print(f'---------- Response {i} ----------')
#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         messages.append({"role": "assistant", "content": response})
#         print(response)
#     print('========== End Conversation ==========')

#     return messages

# #%%
# # Define prompts template

# system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"

# prompts_t = [
#     '详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。\n{item}',
#     '对该商品名称作命名实体识别（NER），找出目标实体。请注意，目标实体可能不存在于商品名称中。\n目标实体定义如下：\n{entity_definition}',
#     '根据以上信息，重新输出商品名称，并将每个目标实体用 @@ 在开头、## 在结尾标记。\n请只输出商品名称，不要包含任何其他信息。',
# ]

# #%%
# # Input configs

# # Use one item from items_dataset as example
# # item = items_dataset['product_name'][0]
# item = '【御宅殿】25年3月預購 萬代 景品 火影忍者 Memorable Saga 宇智波佐助Ⅱ 咒印 1017'

# #%%
# # Build corpus (VERY IMPORTANT FOR THE ZERO-SHOT)

# # # Use related items as corpus. Think of it as we trained ecombert with MLM on the MOMO+PC dataset.
# # corpus = ', '.join(get_related_items(item, items_dataset, top_k=5))

# # Use external knowledge as corpus (e.g. wikipedia). Currently its fixed and manually set, it can be automated with a simple RAG.
# corpus = """《敗北女角太多了！》（日語：負けヒロインが多すぎる！，簡稱「マケイン」）是日本作家雨森焚火創作的輕小說系列，由《這個美術社大有問題！》作者伊右群擔綱插圖，2021年7月起經小學館旗下GAGAGA文庫發行。第15屆小學館輕小說大獎「GAGAGA獎」得獎作品[1]。改編漫畫由2022年4月29日起，在漫畫應用程式「Manga One」開始連載，由いたち繪畫[2]。2023年12月宣布電視動畫化[3]，於2024年7月至9月播放[4]。

# 故事舞台為作者雨森焚火的出身地愛知縣豐橋市[1]。書名中的「敗北女角」（負けヒロイン）指在後宮型作品的戀愛競爭中，爭奪不了男主角或不肯承認爭奪男友失敗的女性角色[5]，在簡體中文翻譯中則譯為「敗犬女主」。本作是雨森的出道作[1]，也是正職漫畫家伊右群，初次繪畫輕小說封面和插圖[5]。"""

# #%%
# # Entity definitions

# # Use one of the following definitions
# definition_brand = '商品品牌名称，指具体标识商品或服务来源的品牌或厂牌名称，如华硕、LG、Apple、Sony 等，涵盖科技产品、家电、服装、汽车等各行业的品牌名称。不包括电商平台中的店铺名称、卖场名称或个人卖家名称。'
# definition_series_THIS_TAG_SUCK = '商品中的所有产品系列，以及商家为特殊目的特别额外创造的商品名称，不包括特殊主题或产品类型，不含广告词。例如 iPhone 12、ROG 3060Ti。'
# definition_ip = '与特定知名IP（如漫画、电影、文学作品）或其创作者相关的产品或创作。例子包括知名作品（如《海贼王》）或直接与创作者本人（如J.K. Rowling）相关的内容。此定义不包括公司、发行商或经纪公司等机构。'
# definition_type = '具体产品名称（产品类型）。例如：电脑、滑鼠、键盘、萤幕、玩具、饼干、卫生纸等实体产品。'

# #%%
# # Inference

# # Build system message with retrieved items
# system_message = system_message_t.format(corpus=corpus)

# # Build prompts with item and tag definition
# prompts = [prompt.format(item=item, entity_definition=definition_ip) for prompt in prompts_t]

# # Run instructions
# messages = run_instructions(model, tokenizer, prompts, system_message)

# #%%