# Importing necessary libraries
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from opencc import OpenCC

# Load the model and tokenizer from local files using Huggingface Transformers
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Loading model and tokenizer from local environment
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to load all CSV files from a directory and concatenate them into a DataFrame
def load_all_items(path: str):
    # List all CSV files in the directory
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]

    # Load each CSV and concatenate them into a single DataFrame
    df_list = []
    for file_name in file_names:
        df = pd.read_csv(os.path.join(path, file_name))
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

# Function to get related item names from a dataset (pseudo-sampling)
def get_related_item_names(current_item_names: str, items: pd.DataFrame, num_sample: int = 5):
    # Randomly sample related items
    related_items = items.sample(num_sample)
    related_item_names = related_items['product_name'].tolist()
    return related_item_names

# Load the item names from the specified path
item_names = load_all_items('random_samples_1M')

# Function to create the prompt based on the current item name and context
def get_prompt(current_item_name, item_names=item_names, num_sample=5):
    # Prompt template to pass to the model
    prompt_template = """1. 詳細瞭解以下電商網站的商品名稱，盡可能推論出此商品名稱的所有資訊。
2. 請對下列商品名稱作命名實體辨識(NER)，找出「產品系列」實體。 “產品系列”實體定義為：商品中所有的 “產品系列”，以及商人為了特殊目的所特別額外創造出的商品名稱，不包含特殊主題或是產品類型，不含廣告詞。如 Iphone 12、ROG 3060Ti。
3. 找出商品名稱中代表「產品系列」的字符，用 @@ 和 ## 圍起來。
4. 再次檢查輸出結果，確認符合定義。
5. 最後，換行一次，並寫上完整的答案。

{current_item_name}"""

    # Add context by sampling related item names
    context = '\n'.join(get_related_item_names(current_item_name, item_names, num_sample))
    # Format the template with the context and current item name
    context = prompt_template.format(item_names=context, current_item_name=current_item_name)
    # Convert traditional Chinese to simplified Chinese using OpenCC
    context_s = OpenCC('t2s').convert(context)
    return context_s

# Function to generate the response from the locally loaded model
def generate_response(current_item_name, model, tokenizer, num_sample=5):
    # Get the prompt
    prompt = get_prompt(current_item_name, num_sample=num_sample)
    print('----- Prompt -----')
    print(prompt)

    # Tokenize the prompt
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the output
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated output
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Iterate through sampled item names and generate responses
for item_name in item_names['product_name'].sample(10):
    response = generate_response(item_name, model, tokenizer, num_sample=20)
    
    # Print the output response
    print('----- Output -----')
    print(response)
