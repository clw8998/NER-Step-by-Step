#%%
# Import libraries
import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from opencc import OpenCC

#%%
# Download the dataset if it does not exist
import get_dataset # This import itself will download the dataset, DO NOT REMOVE

#%%
# Define the model name
model_name = "Qwen/Qwen2.5-7B-Instruct"

# # Define the 8-bit configuration using BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,          # Enable 8-bit quantization
#     llm_int8_threshold=6.0,     # Threshold for mixed 8-bit precision (optional)
#     llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading for weights in fp32 (optional)
# )

# Define bitsandbytes configuration for loading in 4-bit or 8-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # You can also set this to load_in_8bit=True for 8-bit quantization
    bnb_4bit_use_double_quant=True,  # Optional: Enables double quantization for more efficiency
    bnb_4bit_quant_type="nf4",  # Set the quantization type, options are 'fp4' or 'nf4'
    bnb_4bit_compute_dtype=torch.float16  # Default is float32, set this to torch.float16 for reduced precision
)

# Load the model with the bitsandbytes config for 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Apply the 8-bit configuration
    device_map="auto",               # Automatically map model to available devices
    torch_dtype=torch.float16        # Optional: Use float16 precision where applicable
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#%%
# Load items dataset
def load_items(path: str):
    # List all CSV files in the directory
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]

    # Load each CSV and concatenate them into a single DataFrame
    df_list = []
    for file_name in file_names:
        df = pd.read_csv(os.path.join(path, file_name))
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

items_dataset = load_items('random_samples_1M')

#%%
# Declare retrieval function.
def get_related_items(current_item_names: str, items_dataset: pd.DataFrame, top_k: int = 5):
    # Randomly sample items (pseudo retrieval)
    related_items = items_dataset.sample(top_k)
    related_items = related_items['product_name'].tolist()

    return related_items

#%%
# Declare multi-prompts inference function
def run_instructions(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                     prompts: list[str], system_message: str = None):
    # Initialize the conversation
    messages = []

    print('========== Start Conversation ==========')
    if system_message:
        print('---------- System Message ----------')
        messages.append({"role": "system", "content": system_message})
        print(system_message)
    
    for i in range(len(prompts)):
        # Add user prompt to the conversation messages
        print(f'---------- Instruction {i} ----------')
        messages.append({"role": "user", "content": prompts[i]})
        print(prompts[i])

        # Tokenize the messages
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate the output
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the generated output
        print(f'---------- Response {i} ----------')
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        print(response)
    print('========== End Conversation ==========')

    return messages

#%%
# Define prompts template

system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"

prompts_t = [
    '详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。\n{item}',
    '对该商品名称作命名实体识别（NER），找出目标实体。目标实体定义如下：\n{entity_definition}\n\n请注意，目标实体可能不存在于商品名称中。',
    '根据以上信息，重新输出商品名称，并将每个目标实体用 @@ 在开头、## 在结尾标记。\n请只输出商品名称，不要包含任何其他信息。',
]

#%%
# Input configs

# Use one item from items_dataset as example
# item = items_dataset['product_name'][0]
item = '【御宅殿】25年3月預購 萬代 景品 火影忍者 Memorable Saga 宇智波佐助Ⅱ 咒印 1017'

#%%
# Build corpus (VERY IMPORTANT FOR THE ZERO-SHOT)

# # Use related items as corpus. Think of it as we trained ecombert with MLM on the MOMO+PC dataset.
# corpus = ', '.join(get_related_items(item, items_dataset, top_k=5))

# Use external knowledge as corpus (e.g. wikipedia). Currently its fixed and manually set, it can be automated with a simple RAG.
corpus = """《敗北女角太多了！》（日語：負けヒロインが多すぎる！，簡稱「マケイン」）是日本作家雨森焚火創作的輕小說系列，由《這個美術社大有問題！》作者伊右群擔綱插圖，2021年7月起經小學館旗下GAGAGA文庫發行。第15屆小學館輕小說大獎「GAGAGA獎」得獎作品[1]。改編漫畫由2022年4月29日起，在漫畫應用程式「Manga One」開始連載，由いたち繪畫[2]。2023年12月宣布電視動畫化[3]，於2024年7月至9月播放[4]。

故事舞台為作者雨森焚火的出身地愛知縣豐橋市[1]。書名中的「敗北女角」（負けヒロイン）指在後宮型作品的戀愛競爭中，爭奪不了男主角或不肯承認爭奪男友失敗的女性角色[5]，在簡體中文翻譯中則譯為「敗犬女主」。本作是雨森的出道作[1]，也是正職漫畫家伊右群，初次繪畫輕小說封面和插圖[5]。"""

#%%
# Entity definitions

# Use one of the following definitions
definition_brand = '商品品牌名称，指具体标识商品或服务来源的品牌或厂牌名称，如华硕、LG、Apple、Sony 等，涵盖科技产品、家电、服装、汽车等各行业的品牌名称。不包括电商平台中的店铺名称、卖场名称或个人卖家名称。'
definition_series_THIS_TAG_SUCK = '商品中的所有产品系列，以及商家为特殊目的特别额外创造的商品名称，不包括特殊主题或产品类型，不含广告词。例如 iPhone 12、ROG 3060Ti。'
definition_ip = '与特定知名IP（如漫画、电影、文学作品）或其创作者相关的产品或创作。例子包括知名作品（如《海贼王》）或直接与创作者本人（如J.K. Rowling）相关的内容。此定义不包括公司、发行商或经纪公司等机构。'
definition_type = '具体产品名称（产品类型）。例如：电脑、滑鼠、键盘、萤幕、玩具、饼干、卫生纸等实体产品。'

#%%
# Inference

# Build system message with retrieved items
system_message = system_message_t.format(corpus=corpus)

# Build prompts with item and tag definition
prompts = [prompt.format(item=item, entity_definition=definition_ip) for prompt in prompts_t]

# Run instructions
messages = run_instructions(model, tokenizer, prompts, system_message)

#%%