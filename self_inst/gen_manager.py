import os
import os.path
import random
from functools import partial
from multiprocessing import Pool
from dataclasses import dataclass

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from typing import cast

import numpy as np
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from langchain_core.language_models import BaseChatModel

from client.concrete.self_inst_gen import SelfInstructGenerator

from util.file_tools import JsonHandler, JsonlHandler
from util.path_tools import OutputPathCreator

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-MS-7b-v0.1")

TASK_LIST_LENGTH = 10
FEW_SHOT_NUM = 3


@dataclass
class GenerateConfig:
    input_file: str
    output_dir: str
    num_instructions_to_generate: int  # トータルで生成する目標個数
    num_tasks_to_generate: int = 7  # 同時生成するタスクの個数
    num_few_shot: int = 3  # few-shot で与える個数
    num_cpu: int = 16


def post_process_response(
        num_few_shot: int,
        response: List[Dict]
):
    if response is None:
        return []

    instructions = []
    for idx, task_dict in enumerate(response):
        idx += num_few_shot + 1

        no = task_dict.get('no', 0)
        inst = task_dict.get('instruction')
        output = task_dict.get('output')

        # instruction が""は生成に失敗している可能性が高いので除外する
        if not inst:
            logger.error(f"may be generation error: {inst}")
            # print('may be error:', inst)
            continue

        # #　数が一致していない場合は、最初の数個がseedと同じ内容になっている可能性が高い
        if no != idx and no <= num_few_shot:
            logger.error(f"may be generation error: {no} - {idx}")
            # print('may be error:', no, idx)
            continue

        if len(inst) <= 10:
            logger.error(f"filter out too short instructions. instruction lengths: {len(inst)}")
            continue

        ''' ここからは文字関係のフィルタリング '''
        # instがすべて英語のASCII文字である場合
        if all(char.isascii() for char in inst):
            logger.error("Filter: instはすべて英語のASCII文字です")
            continue

        new_inst = {"instruction": inst, "output": output}

        instructions.append(new_inst)

    return instructions


class GenerateManager(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
    ):
        self._model = main_llm
        # RougeScorerの日本語化対応
        self._tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-MS-7b-v0.1")

    def __call__(
            self,
            cfg: GenerateConfig
    ) -> None:
        self.file_handling(cfg)

    def file_handling(
            self,
            cfg: GenerateConfig
    ):
        input_file = cfg.input_file
        output_dir = cfg.output_dir
        num_inst_to_gen = cfg.num_instructions_to_generate
        num_tasks_to_gen = cfg.num_tasks_to_generate
        num_few_shot = cfg.num_few_shot
        num_cpu = cfg.num_cpu

        # インプットファイル名に基づいで、OUTPUT用のファイル名を作成するツール
        op_gen = OutputPathCreator(output_dir=output_dir, out_suffix='.json', add_stem='-regen')
        output_file = op_gen(input_file)

        inst_gen = SelfInstructGenerator(self._model, num_tasks_to_gen, use_gen_num_check=True)
        jlh = JsonlHandler()
        jh = JsonHandler()

        msg = """
        'input_file'にはAlpaca形式のデータが格納された'*.jsonl'を指定してください。
        """
        assert input_file.endswith(".jsonl"), msg
        seed_tasks = jlh.read(input_file)

        seed_instruction_data = [
            {
                "instruction": t["instruction"],
                "output": t["instances"][0]["output"],
            }
            for t in seed_tasks
        ]
        logger.info(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

        request_idx = 0
        # load the LM-generated instructions
        machine_instruction_data = []

        if os.path.exists(output_file):
            machine_instruction_data = jh.read(output_file)
            logger.info(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

        scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=self._tokenizer, use_stemmer=False)

        # now let's generate new instructions!
        progress_bar = tqdm(total=num_inst_to_gen)
        if machine_instruction_data:
            progress_bar.update(len(machine_instruction_data))

        all_instructions = [d["instruction"] for d in seed_instruction_data] + [
            d["instruction"] for d in machine_instruction_data
        ]
        all_instruction_tokens = [
            scorer._tokenizer.tokenize(inst) for inst in all_instructions
        ]

        while len(machine_instruction_data) < num_inst_to_gen:
            request_idx += 1

            # only sampling from the seed tasks
            prompt_seeds = random.sample(
                seed_instruction_data, num_few_shot
            )

            few_shot = inst_gen.encode_few_shot_prompt(prompt_seeds)
            next_no = len(prompt_seeds) + 1
            inst = {
                "few_shot": few_shot,
                "next_no": next_no,
            }
            task_list = inst_gen.invoke(inst)

            instruction_data = post_process_response(num_few_shot, task_list)

            total = len(instruction_data)
            logger.info(f"total_instruction_data_num: {total}")

            keep = 0
            for instruction_data_entry in instruction_data:
                ''' 
                seed_tasks_japanese.jsonl の例題のうちどのinstruction(指示)に類似するか確認している 
                '''
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                with Pool(num_cpu) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,  # seed_tasks_japanese.jsonlのinstructionをtoken化してまとめたリスト(175個の要素)
                    )

                rouge_scores = [score.fmeasure for score in rouge_scores]

                # assert False, ''

                '''最も類似するInstructionのtop10?'''
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) > 0.7:
                    logger.error(f"Filter: 最大rouge_scoresが0.7を超えています。")
                    continue
                else:
                    keep += 1
                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)

            logger.info(f"Generated {total} instructions, kept {keep} instructions")
            jh.write(machine_instruction_data, output_file)


if __name__ == "__main__":
    """
    python -m self_inst.gen_manager
    """


    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)


    SEED = 46
    fix_seed(SEED)

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig

    basicConfig(level=WARN)

    # from model.groq_llm import GroqChatBase
    # llm_main = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # max_tokens=2048,
    #     requests_per_second=0.01,
    #     temperature=0.6,
    # )

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.5)

    test_cfg = dict(
        input_file="./data/input/self_inst/seed_tasks_jp.jsonl",
        output_dir="./data/output/self_inst",  # 入力ファイル名 + '-regen.json' として出力
        # num_instructions_to_generate=1000,  # トータルで生成する目標個数
        num_instructions_to_generate=10,  # 同時生成するタスクの個数
        num_tasks_to_generate=8,  # 1回で生成する総数
        num_few_shot=4,  # few-shot で参考として与えるシードの個数
        num_cpu=8,
    )

    em = GenerateManager(
        main_llm=llm,
    )

    em(GenerateConfig(**cast(Dict, test_cfg)))
