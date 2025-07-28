# Import libraries
import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from opencc import OpenCC
import pickle
import random
import gc   


import warnings
warnings.filterwarnings("ignore")

t2s = OpenCC('t2s')
s2t = OpenCC('s2t')

# Define argparse for handling arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with various configurations")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Name of the model")
    parser.add_argument('--top_k', type=int, default=10, help="Top K related items to retrieve")
    parser.add_argument('--test_mode', action='store_true', help="Whether to run in test mode or not")
    parser.add_argument('--result_dir', type=str, default="./results", help="Directory to save results")
    parser.add_argument('--inference_file', type=str, default='./data/test.pickle', help="Input file for inference")
    parser.add_argument('--dtype', type=str, default='int8', choices=['int8', 'int4', 'fp16', 'bf16', 'fp32'], help="Data type for model precision (int8 or int4)")
    parser.add_argument('--save_results', action='store_true', help="Whether to save inference results")
    parser.add_argument('--num_inference', type=int, default=-1, help="Number of items to infer, -1 means all")
    parser.add_argument('--use_tag', type=str, nargs='*', default=[], help="Tags to use during inference.")
    parser.add_argument('--temerature', type=float, default=1e-5, help="Temperature for generation")
    parser.add_argument('--use_qwen_api', type=bool, default=False, help="Whether to use Qwen API for inference")
    parser.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode")
    parser.add_argument('--corpus', type=str, default='coupang', choices=['coupang', 'pchome', 'pchome_train_11000'], help="Corpus to use for inference")
    parser.add_argument('--rag_corpus', type=bool, default=False, help="Whether to use RAG for corpus retrieval")
    parser.add_argument('--rag_error', type=bool, default=False, help="Whether to use RAG for error cases retrieval")
    parser.add_argument('--prompt_type', type=int, default=0, choices=[0, 1, 2, 3], help="Prompt type to use for inference")

    return parser.parse_args()

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
    
    
    for i in range(len(prompts)):
        print(f'---------- Instruction {i} ----------')
        prompts[i] = t2s.convert(prompts[i])
        messages.append({"role": "user", "content": prompts[i]})
        print(prompts[i])

        print(f'---------- Response {i} ----------')

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,  enable_thinking=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=8192, temperature=temperature)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


        think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        thinking, answer = [], []
        for ids in generated_ids:
            ids_list = ids.tolist()

            try:
                split = ids_list.index(think_end_id) + 1
            except ValueError:
                split = 0

            thinking.append(
                tokenizer.decode(ids_list[:split], skip_special_tokens=True).strip()
            )
            answer.append(
                tokenizer.decode(ids_list[split:], skip_special_tokens=True).strip()
            )

        thinking_content = thinking[0]
        # print(f"Thinking: {thinking_content}")
        response = answer[0]

        messages.append({"role": "assistant", "content": response})
        print(response)
    print('========== End Conversation ==========')
    return messages


def main():
    args = parse_args()
    print(args)
    if not args.use_qwen_api:
        if args.dtype == 'fp16':
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
        model, tokenizer = None, None  # Set these to None if using API


    # Load inference data
    with open(args.inference_file, 'rb') as file:
        pc_test_data = pickle.load(file)

    p_names = [item['context'] for item in pc_test_data]
    # drop duplicates
    p_names = sorted(set(p_names))

    # Control number of inference items
    if args.num_inference != -1:
        p_names = p_names[:args.num_inference]

    annotation_rules = {
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

    system_message_t = "你是一位熟悉電子商務的助手"

    if args.rag_corpus:
        system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"

    
    if args.rag_error:
        system_message_t += "以下是你曾经预测错误的案例，请参考以下内容，避免再犯相同的错误：\n{error_corpus}"

    if args.rag_corpus & args.rag_error:
        system_message_t = "你是一位熟悉電子商務的助手，以下是供你參考的語料庫：\n{corpus}"
        system_message_t += "另外以下是你曾经预测错误的案例，请参考以下内容，避免再犯相同的错误：\n{error_corpus}"
    

    if args.prompt_type == 0:

        prompts_t = [
            '详细了解以下商品，尽可能辨认出你认识的所有关键词，并解释。\n{item}',
            '对该商品作命名实体识别（NER），找出目标实体。\n目标实体定义如下：\n{entity_definition}',
            '根据以上你推论的结果，输出命名实体识别结果。\n请只输出命名实体识别结果，不要包含任何其他信息，若有多组实体，中间以「、」隔开。',
        ]
    elif args.prompt_type == 1:
        prompts_t = [
            '详细地理解「{item}」这个商品名称，尽可能辨认出你认识的所有中文或英文的关键词，并解释。',
            '從「{item}」中，找出所有中文或英文的目标实体。\n目标实体定义如下：\n{entity_definition}',
            '根据以上的推论结果，从「{item}」中，输出命名实体识别结果。\n请只输出命名实体识别结果，不要包含任何其他信息，若有多组实体，中间以「、」隔开。',
        ]
    elif args.prompt_type == 2:
        prompts_t = [
            '请对商品名称「{item}」进行全面分析，并提供详细说明。',
            '从商品名称「{item}」中，提取属性「{attr}」的中文或英文实体，并输出结果。\n请注意: 有些商品可能不包含任何实体内容，而对于书籍类商品，则一律不进行标注。',
            '仅输出属性「{attr}」的实体识别结果即可，不要输出任何多余的内容，若有多个实体，请用「、」分隔。请注意：若一个实体同时包含中文与英文，应将其拆分为两个独立实体，且实体名称中不得包含多余的标点符号。',
            '请确保输出仅包含实体及分隔符“、”，且所有实体必须严格为商品名称「{item}」中的连续子串；若已满足上述条件，请直接复用上一次输出，否则请先修正后再输出。'
        ]
    
    for i, p_name in enumerate(p_names):
        system_message = system_message_t
        
        for tag, definition in annotation_rules.items():
            if tag not in args.use_tag:
                continue
            if not args.save_results or os.path.exists(f'{args.result_dir}/{i}_{tag}.pkl'):
                print(f"Skipping {i}_{tag} as results already exist.")
                continue
            prompts = [prompt.format(item=p_name, attr=tag, entity_definition=definition) for prompt in prompts_t]

            messages = run_instructions(
                model, tokenizer, prompts, system_message, args.temerature,
                test_mode=args.test_mode, use_qwen_api=args.use_qwen_api
            )

            # Save messages only if save_results is True
            if args.save_results:
                os.makedirs(args.result_dir, exist_ok=True)
                with open(f'{args.result_dir}/{i}_{tag}.pkl', 'wb') as file:
                    messages.insert(0,{'context': p_name, 'question':tag})
                    pickle.dump(messages, file)

if __name__ == "__main__":
    MAX_RETRIES = 100

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n==== Attempt {attempt}/{MAX_RETRIES} ====")
            main()
            break 
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] Run {attempt} failed: {e}")
            print("→ Freeing memory and retrying …")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("!!! All 100 attempts failed, giving up.")
