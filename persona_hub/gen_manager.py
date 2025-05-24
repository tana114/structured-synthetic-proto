import os.path
from dataclasses import dataclass
from pydoc_data.topics import topics
from typing import List, Dict, Tuple, Optional, final, Union, Any
import copy
import random

from tqdm.auto import tqdm

import numpy as np

from langchain_core.language_models import BaseChatModel

from client.concrete.persona_multiturn_gen import PersonaMultiGenerator

from util.file_tools import JsonHandler, JsonlHandler
from util.path_tools import OutputPathCreator

'''
参考にさせていただきました
https://github.com/tencent-ailab/persona-hub
https://github.com/matsuolab/nedo_project_code/tree/team_hatakeyama_phase2/team_hatakeyama_phase2/ota/topic-hub

'''


@dataclass
class ManageConfig:
    input_file: str
    output_dir: str
    num_instructions_for_train: int
    num_instructions_for_test: int
    num_of_evolve: int
    num_of_generation: int


RANDOM_SEED = 0


def fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


class PersonaHubManager(object):
    def __init__(
            self,
            main_llm: BaseChatModel,
            generate_batch_size: int = 10,
    ):
        """

        """
        self._model = main_llm
        self._pgen = PersonaMultiGenerator(
            chat_model=self._model,
            use_gen_num_check=True,
        )
        self._batch_size = generate_batch_size

    def gen_multi_conversation(self, task_seeds: List[Dict]) -> List[Dict]:
        """

        Parameters
        ----------
        task_seeds

        Returns
        -------

        """
        # few-shotの文字列を作成
        few_shot = self._pgen.encode_few_shot_prompt(task_seeds)

        instruction = {
            "persona_info": few_shot,
        }

        conversations = []
        results = self._pgen(instruction)

        # 暫定的なエラー処理
        if not results:
            return conversations

        for r in results:
            convs = dict(
                q1=r['first_generated_problem'],
                a1=r['first_response_to_problem'],
                q2=r['second_additional_problem'],
                a2=r['second_response_to_problem'],
            )
            conversations.append(convs)

        return conversations

    def __call__(
            self,
            seed_file: str,
            out_dir: str = './data/output',
    ) -> None:
        return self.file_handling(seed_file, out_dir)

    def file_handling(
            self,
            seed_set_file: str,
            output_dir: str,
    ) -> None:

        # This tool creates an output file name based on an input file name.
        op_gen = OutputPathCreator(output_dir=output_dir, out_suffix='.jsonl', add_stem='-gen')
        output_file = op_gen(seed_set_file)

        # file handle tools
        jlh = JsonlHandler()  # tool for *.jesonl files.

        seed_set = jlh.read(seed_set_file)

        def batch_processor(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield self.gen_multi_conversation(data[i:i + batch_size])

        gen_objects = jlh.read(output_file) if os.path.isfile(output_file) else []
        for processed in tqdm(batch_processor(seed_set, self._batch_size)):

            gen_objects.extend(processed)
            jlh.write(gen_objects, output_file)


if __name__ == "__main__":
    """
    python -m persona_hub.gen_manager
    """

    from logging import DEBUG, INFO, WARN, ERROR, basicConfig
    basicConfig(level=WARN)

    # from model.groq_llm import GroqChatBase
    # llm_main = GroqChatBase(
    #     model_name="llama-3.3-70b-versatile",
    #     # max_tokens=2048,
    #     # requests_per_second=0.06,
    #     requests_per_second=0.01,
    #     temperature=0.6,
    # )

    from langchain_experimental.llms.ollama_functions import OllamaFunctions

    # 構造化出力用のllm
    llm_main = OllamaFunctions(model="qwen3:4b", format="json", temperature=0.8)


    em = PersonaHubManager(
        main_llm=llm_main,
        # generate_batch_size = 10,
        generate_batch_size = 3,
    )

    seed_set_file = './data/input/persona_hub/math_seeds_0k.jsonl'
    output_dir = './data/output/persona_hub'

    em(seed_set_file, output_dir)
