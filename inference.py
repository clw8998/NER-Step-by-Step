#%%
from openai import OpenAI
import pandas as pd
import os
from opencc import OpenCC

client = OpenAI(    
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

#%%
# Declare funcions

def load_all_items(path:str):
    # list all csv files under the dir, only csv
    file_names = [f for f in os.listdir(path) if f.endswith('.csv')]

    # load all csv files and concat them
    df_list = []
    for file_name in file_names:
        df = pd.read_csv(os.path.join(path, file_name))
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def get_related_item_names(current_item_names: str, items: pd.DataFrame, num_sample: int = 5):
    # get related items (pseudo-sampling)
    related_items = items.sample(num_sample)
    related_item_names = related_items['product_name'].tolist()
    return related_item_names

item_names = load_all_items('random_samples_1M')

#%%
def get_prompt(current_item_name, item_names=item_names, num_sample=5):
#     prompt_template = """商品資料庫（供參考）：
# {item_names}

# 1. 詳細瞭解以下電商網站的商品名稱，盡可能推論出此商品名稱的所有資訊。
# 2. 請對下列商品名稱作命名實體辨識(NER)，找出「產品系列」實體。 “產品系列”實體定義為：商品中所有的 “產品系列”，以及商人為了特殊目的所特別額外創造出的商品名稱，不包含特殊主題或是產品類型，不含廣告詞。如 Iphone 12、ROG 3060Ti。
# 3. 找出商品名稱中代表「產品系列」的字符，用 @@ 和 ## 圍起來。如：XXXX@@XX##XXXXXXXX
# 4. 再次檢查輸出結果，確認符合定義。
# 5. 最後，換行一次，並寫上完整的答案。

# {current_item_name}"""

    prompt_template = """1. 詳細瞭解以下電商網站的商品名稱，盡可能推論出此商品名稱的所有資訊。
2. 請對下列商品名稱作命名實體辨識(NER)，找出「產品系列」實體。 “產品系列”實體定義為：商品中所有的 “產品系列”，以及商人為了特殊目的所特別額外創造出的商品名稱，不包含特殊主題或是產品類型，不含廣告詞。如 Iphone 12、ROG 3060Ti。
3. 找出商品名稱中代表「產品系列」的字符，用 @@ 和 ## 圍起來。
4. 再次檢查輸出結果，確認符合定義。
5. 最後，換行一次，並寫上完整的答案。

{current_item_name}"""

    context = '\n'.join(get_related_item_names(current_item_name, item_names, num_sample))
    context = prompt_template.format(item_names=context, current_item_name=current_item_name)
    context_s = OpenCC('t2s').convert(context)
    return context_s

def send_request(current_item_name, client, num_sample=5):  
    prompt = get_prompt(current_item_name, num_sample=num_sample)
    print('----- Prompt -----')
    print(prompt)

    response = client.chat.completions.create(
        model="qwen2.5:32b-instruct-q4_0",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response


for item_name in item_names['product_name'].sample(10):

    response = send_request(item_name, 
                            client,
                            num_sample=20)
    
    # keep only the line with both @@ and ##
    output = response.choices[0].message.content
    # output = output.split('\n')
    # output = [line for line in output if '@@' in line and '##' in line]
    # output = '\n'.join(output)

    print('----- Output -----')
    print(output)

#%%