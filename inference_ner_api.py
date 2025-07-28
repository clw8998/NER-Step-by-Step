from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

from openai import OpenAI  # pip install openai>=1.30.2
from opencc import OpenCC  # pip install opencc
from tqdm import tqdm      # pip install tqdm

# ---------------------------- Constants ---------------------------- #
T2S = OpenCC("t2s")

ATTRIBUTE_RULES = {
    "品牌": '商品的品牌，但不包括电商平台中的店铺名称、卖场名称或个人卖家名称，也不包含地名、产地。',
    "系列名稱": "產品中所有的「產品系列」，以及商人為了特殊目的所特別額外創造出的商品名稱，使用者可能會利用該名稱搜尋商品，不包含特殊主題或是產品類型，不含廣告詞。如 Iphone 12、ROG 3060Ti",
    "產品類型": "實際產品名稱",
    "產品序號": "產品序號，該產品所擁有的唯一英數符號組合序號，不含系列名。",
    "顏色": "顏色資訊，包含化妝品以及明亮。如 花朵紅、藍色系、晶亮",
    "材質": "產品的製造材料，一般情況下不能食用，較接近原物料，不是產品成分，請注意，「紙」尿褲的材質是棉，不是紙。",
    "對象與族群": "人與動物的族群。如 新生兒、寵物鳥、高齡族群",
    "適用物體、事件與場所": "適用的物品、事件、場所。如 手部用、騎車用、廚房用",
    "特殊主題": "該物品富含特殊人為創造作，人物，並且該創作者有一定知名度。如 航海王、J.K. Rowling",
    "形狀": "商品形狀，囊括簡單的幾何形，以及明確從名稱中可以知道該商品屬於該形狀的詞，包含衣服版型。如 圓形、鈕扣形、可愛熊造型",
    "圖案": "商品上的圖案，囊括簡單的幾個圖形，以及明確從名稱中可以知道該商品屬於該圖案的詞",
    "尺寸": "商品大小，常以數字與單位或特殊規格形式出現。如 120x80x10cm (長寬高)、XL、ATX（主機板）",
    "重量": "商品重量，常以數字與單位或特殊規格形式出現。如 10g、極輕",
    "容量": "商品容量，常以數字與單位或特殊規格形式出現。如 128G (電腦)、大容量",
    "包裝組合": "產品包裝方式、包裝分層以及配品，如 10入、10g/包、鍵盤滑鼠墊組合、送電池",
    "功能與規格": "產品的功用，與其特殊規格，以及產品額外的特性。如 USB3.0、防曬、太陽能"
}

SYSTEM_PROMPT = "你是一位熟悉電子商務的助手"

PROMPTS_TEMPLATE: List[str] = [
    "请对商品名称「{item}」进行全面分析，并提供详细说明。",
    "从商品名称「{item}」中，提取属性「{attr}」的中文或英文实体，并输出结果。\n属性「{attr}」的定义是：{entity_definition}。\n请注意: 有些商品可能不包含任何实体内容，而对于书籍类商品，则一律不进行标注。",
    "仅输出属性「{attr}」的实体识别结果即可，不要输出任何多余的内容，若有多个实体，请用「、」分隔。请注意：若一个实体中同时出现意义相同的中文与英文（如「Sony索尼」），须将其拆分为两个独立实体，且实体名称中不得包含多余的标点符号。",
    "请确保输出仅包含实体及分隔符“、”，且所有实体必须严格为商品名称「{item}」中的连续子串；若已满足上述条件，请直接复用上一次输出，否则请先修正后再输出。",
]

# ------------------------- CLI Arguments --------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("NER inference via OpenRouter API (prints every turn)")
    p.add_argument("--openrouter_api_key", required=True, help="sk-or-...")
    p.add_argument("--model_name", default="qwen/qwen3-235b-a22b:online",
                   help="Model ID on OpenRouter (append :online for web search)")
    p.add_argument("--temperature", type=float, default=1e-5, help="Sampling temperature")
    p.add_argument("--inference_file", default="./data/test.pickle",
                   help="Pickle file containing test samples (list[dict] with 'context')")
    p.add_argument("--result_dir", default="./results", help="Where to save pickle logs")
    p.add_argument("--num_inference", type=int, default=-1, help="Limit #products (-1 = all)")
    p.add_argument("--use_tag", nargs="*", default=[], help="Subset of attribute tags")
    p.add_argument("--save_results", action="store_true", help="Save conversation pickle")
    return p.parse_args()

# --------------------------- Helpers ------------------------------- #

def build_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def run_chat(client: OpenAI, model: str, prompts: List[str], temperature: float) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n\n========== Start Conversation ==========")
    print("---------- System Message ----------")
    print(SYSTEM_PROMPT)

    for idx, prompt in enumerate(prompts):
        prompt_t = T2S.convert(prompt)
        print(f"---------- Instruction {idx} ----------")
        print(prompt_t)
        messages.append({"role": "user", "content": prompt_t})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        response = completion.choices[0].message.content.strip()

        print(f"---------- Response {idx} ----------")
        print(response)
        messages.append({"role": "assistant", "content": response})

    print("========== End Conversation ==========")
    return messages

# ------------------------------ Main ------------------------------- #

def main() -> None:
    args = parse_args()
    client = build_client(args.openrouter_api_key)

    with open(args.inference_file, "rb") as fh:
        data = pickle.load(fh)

    products = sorted({item["context"] for item in data})
    if args.num_inference != -1:
        products = products[: args.num_inference]

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(tqdm(products, desc="Products")):
        for tag, definition in ATTRIBUTE_RULES.items():
            if args.use_tag and tag not in args.use_tag:
                continue

            out_path = Path(args.result_dir) / f"{idx}_{tag}.pkl"
            if args.save_results and out_path.exists():
                print(f"Skipping {out_path} as it already exists.")
                continue  # already done

            prompts = [tpl.format(item=name, attr=tag, entity_definition=definition)
                       for tpl in PROMPTS_TEMPLATE]

            conversation = run_chat(client, args.model_name, prompts, args.temperature)

            if args.save_results:
                conversation.insert(0, {"context": name, "question": tag})
                with out_path.open("wb") as f_out:
                    pickle.dump(conversation, f_out)


if __name__ == "__main__":
    main()
