# IT JUST WORKS

PLEASE NOTE THAT ALL OF THE FOLLOWING EXPERIMENTS ARE EVEN RUN ON A **4BIT** QUANTIZED **7B** MODEL.

TODO:
- [ ] Implement a product-based RAG: BERT + product dataset (e.g. Coupang, Ruten, PChome, etc.)
- [ ] Implement a knowledge-based RAG: BERT + in-domain corpus (e.g. Wikipedia, 萬物皆可萌, etc.)
- [ ] Optimize prompt templates. Minimalize the number of tokens.
- [ ] Try pydantic to enforce output schema. (e.g. json)

## Quick Start

```bash
python inference.py [--model_name MODEL_NAME] [--top_k TOP_K] [--test_mode] [--result_dir RESULT_DIR] [--inference_file INFERENCE_FILE] [--dtype {int8,int4}] [--save_results] [--num_inference NUM_INFERENCE] [--use_tag [USE_TAG [USE_TAG ...]]] [--temperature TEMPERATURE]
```

| Argument         | Description                                                   | Default                        |
|------------------|---------------------------------------------------------------|--------------------------------|
| `--model_name`   | Name of the model.                                            | `Qwen/Qwen2.5-7B-Instruct`     |
| `--top_k`        | Top K related items to retrieve.                              | `10`                           |
| `--test_mode`    | Whether to run in test mode or not.                           | `False`                        |
| `--result_dir`   | Directory to save results.                                    | `./results`                    |
| `--inference_file`| Input file for inference.                                    | `./data/test.pickle`           |
| `--dtype`        | Data type for model precision (int8 or int4).                 | `int8`                         |
| `--save_results` | Whether to save inference results.                            | `False`                        |
| `--num_inference`| Number of items to infer, -1 means all.                       | `-1`                           |
| `--use_tag`      | Tags to use during inference.                                 | `[]`                           |
| `--temperature`  | Temperature for generation.                                   | `1e-5`                         |

Run NER inference:
```bash
python inference.py --model_name "Qwen/Qwen2.5-7B-Instruct" --top_k 10 --result_dir ./results --inference_file ./data/test.pickle --dtype int4 --save_results --num_inference -1 --use_tag "品牌"
```

Run in test mode (use `--test_mode` flag):
```bash
python inference.py --model_name "Qwen/Qwen2.5-7B-Instruct" --top_k 10 --result_dir ./results --inference_file ./data/test.pickle --dtype int4 --save_results True --num_inference -1 --use_tag "品牌" --test_mode
```

## Samples

Each following samples include whole conversation, feel free to jump to the last part to see the final output. (output after `---------- Response 2 ----------`)

**Sample 1: *【御宅殿】25年5月預購 ANIPLEX 敗北女角太多了! 八奈見杏菜 PVC***

Basic example. This is the case where RAG (simulated) **successfully retrieves highly relevant information** from the given product name.

```
========== Start Conversation ==========
---------- System Message ----------
你是一位熟悉電子商務的助手，以下是供你參考的語料庫：
《敗北女角太多了！》（日語：負けヒロインが多すぎる！，簡稱「マケイン」）是日本作家雨森焚火創作的輕小說系列，由《這個美術社大有問題！》作者伊右群擔綱插圖，2021年7月起經小學館旗下GAGAGA文庫發行。第15屆小學館輕小說大獎「GAGAGA獎」得獎作品[1]。改編漫畫由2022年4月29日起，在漫畫應用程式「Manga One」開始連載，由いたち繪畫[2]。2023年12月宣布電視動畫化[3]，於2024年7月至9月播放[4]。

故事舞台為作者雨森焚火的出身地愛知縣豐橋市[1]。書名中的「敗北女角」（負けヒロイン）指在後宮型作品的戀愛競爭中，爭奪不了男主角或不肯承認爭奪男友失敗的女性角色[5]，在簡體中文翻譯中則譯為「敗犬女主」。本作是雨森的出道作[1]，也是正職漫畫家伊右群，初次繪畫輕小說封面和插圖[5]。
---------- Instruction 0 ----------
详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。
【御宅殿】25年5月預購 ANIPLEX 敗北女角太多了! 八奈見杏菜 PVC
---------- Response 0 ----------
根据提供的商品名称，可以辨认出以下关键词：

1. 御宅殿：这可能是商品的名称或品牌，但没有更多的上下文信息来确认其具体含义。
2. 25年5月预购：表示该商品预计在2025年5月开始销售或发货。
3. ANIPLEX：这是一个日本知名的动画、游戏和音乐品牌，隶属于索尼公司。
4. 失败女角太多了!：这是轻小说系列的名字，原名为《負けヒロインが多すぎる！》，简称为「マケイン」。
5. 八乃见杏菜：这是轻小说中的一个角色名字。
6. PVC：这是一种塑料材料，常用于制作模型、玩偶等商品。

综上所述，这个商品似乎是ANIPLEX即将发售的一款与轻小说《失败女角太多了!》相关的PVC模型玩具，主角为八乃见杏菜，预计在2025年5月开始预购。
---------- Instruction 1 ----------
对该商品名称作命名实体识别（NER），找出目标实体。目标实体定义如下：
与特定知名IP（如漫画、电影、文学作品）或其创作者相关的产品或创作。例子包括知名作品（如《海贼王》）或直接与创作者本人（如J.K. Rowling）相关的内容。此定义不包括公司、发行商或经纪公司等机构。

请注意，目标实体可能不存在于商品名称中。
---------- Response 1 ----------
针对商品名称“【御宅殿】25年5月預購 ANIPLEX 失败女角太多了! 八乃见杏菜 PVC”，进行命名实体识别（NER），并识别与特定知名IP或其创作者相关的产品或创作：

1. **失败女角太多了!**：这是轻小说系列的名字，属于知名IP。
2. **八乃见杏菜**：这是轻小说中的角色名字，属于该知名IP的一部分。

总结：
- 目标实体：失败女角太多了!
- 目标实体：八乃见杏菜

这两个实体都与轻小说《失败女角太多了!》及其角色相关，符合目标实体的定义。
---------- Instruction 2 ----------
根据以上信息，重新输出商品名称，并将每个目标实体用 @@ 在开头、## 在结尾标记。
请只输出商品名称，不要包含任何其他信息。
---------- Response 2 ----------
【御宅殿】25年5月預購 ANIPLEX @@失败女角太多了!## 八乃见杏菜 PVC
========== End Conversation ==========
```

**Sample 2: *【御宅殿】25年3月預購 萬代 景品 火影忍者 Memorable Saga 宇智波佐助Ⅱ 咒印 1017***

A bit more complex example. This is the case where RAG (simulated) **couldn't find relevant information** and even provided some missleading information.
But in this sample, the target entities are **much more popular**, so the pretrained model might already contain required knowledge.

```
========== Start Conversation ==========
---------- System Message ----------
你是一位熟悉電子商務的助手，以下是供你參考的語料庫：
《敗北女角太多了！》（日語：負けヒロインが多すぎる！，簡稱「マケイン」）是日本作家雨森焚火創作的輕小說系列，由《這個美術社大有問題！》作者伊右群擔綱插圖，2021年7月起經小學館旗下GAGAGA文庫發行。第15屆小學館輕小說大獎「GAGAGA獎」得獎作品[1]。改編漫畫由2022年4月29日起，在漫畫應用程式「Manga One」開始連載，由いたち繪畫[2]。2023年12月宣布電視動畫化[3]，於2024年7月至9月播放[4]。

故事舞台為作者雨森焚火的出身地愛知縣豐橋市[1]。書名中的「敗北女角」（負けヒロイン）指在後宮型作品的戀愛競爭中，爭奪不了男主角或不肯承認爭奪男友失敗的女性角色[5]，在簡體中文翻譯中則譯為「敗犬女主」。本作是雨森的出道作[1]，也是正職漫畫家伊右群，初次繪畫輕小說封面和插圖[5]。
---------- Instruction 0 ----------
详细了解以下商品名称，尽可能辨认出你认识的所有关键词，并解释。
【御宅殿】25年3月預購 萬代 景品 火影忍者 Memorable Saga 宇智波佐助Ⅱ 咒印 1017
---------- Response 0 ----------
根据您提供的信息，可以识别出以下几个关键词：

1. 御宅殿 - 这个词可能是指动漫爱好者或者对某部作品非常忠诚的粉丝。
2. 25年3月预购 - 表示该商品预计在25年的3月份进行预购。
3. 万代 - 日本著名玩具制造商和游戏开发商，经常推出各种动漫周边产品。
4. 景品 - 在日语中指的是抽奖活动中的奖品。
5. 火影忍者 - 这是一个著名的日本漫画和动画系列，讲述的是关于忍者的故事。
6. Memorable Saga - 这可能是产品的系列名称或者是某个特定事件/章节的名称。
7. 宇智波佐助Ⅱ - 宇智波佐助是《火影忍者》中的一个主要角色，这是他的第二版本。
8. 咒印 - 这可能是产品的主题或设计元素，可能指的是宇智波佐助身上的特殊标记。
9. 1017 - 这可能是产品的编号或者某种特别限定版的标识。

综上所述，这是一份来自万代的火影忍者相关商品，可能是限量版的宇智波佐助主题模型或玩具，预计在2025年3月开始预售，具体细节还需进一步确认。
---------- Instruction 1 ----------
对该商品名称作命名实体识别（NER），找出目标实体。目标实体定义如下：
与特定知名IP（如漫画、电影、文学作品）或其创作者相关的产品或创作。例子包括知名作品（如《海贼王》）或直接与创作者本人（如J.K. Rowling）相关的内容。此定义不包括公司、发行商或经纪公司等机构。

请注意，目标实体可能不存在于商品名称中。
---------- Response 1 ----------
根据您的定义和商品名称，进行命名实体识别（NER）后，可以识别出以下目标实体：

1. **火影忍者** - 这是与知名IP相关的部分，指的是《火影忍者》这一漫画和动画系列。
2. **宇智波佐助** - 这是与知名IP相关的人物角色，是《火影忍者》中的主要角色之一。

具体来说：

- **火影忍者**：这是一个知名的漫画和动画系列，符合与知名IP相关的定义。
- **宇智波佐助**：这是《火影忍者》中的一个重要角色，也符合与知名IP相关的定义。

其他部分如“御宅殿”、“万代”、“景品”、“Memorable Saga”、“咒印”和“1017”等词汇虽然提供了额外的信息，但它们并不直接与特定知名IP或其创作者相关联，因此不属于目标实体。
---------- Instruction 2 ----------
根据以上信息，重新输出商品名称，并将每个目标实体用 @@ 在开头、## 在结尾标记。
请只输出商品名称，不要包含任何其他信息。
---------- Response 2 ----------
@@火影忍者## @@宇智波佐助## Memorable Saga 咒印 1017
========== End Conversation ==========
```