import os.path
from typing import List, Dict, Tuple, Optional, final, Union, Any
import random
import numpy as np
from tqdm import tqdm

from util.file_tools import JsonHandler, JsonlHandler

'''
以下のようなpersonaとtask、topicの組み合わせを生成します
{"persona": "SF作家", "task": "math", "topic": "確率"}

参考にさせていただきました
https://github.com/matsuolab/nedo_project_code/tree/team_hatakeyama_phase2/team_hatakeyama_phase2/ota/topic-hub

'''

def fix_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


class PersonaHubSeedCollector(object):
    def __init__(
            self,
            seed_dir: str = './data/seed',
            output_dir: str = './data',
            batch_size: int = 10,
            remove_duplicate: bool = False,

    ):
        self._seed_dir = seed_dir
        self._output_dir = output_dir
        self._batch = batch_size
        self._remove_dup = remove_duplicate

    @staticmethod
    def remove_duplicate_dicts(list_of_dicts: List[Dict]) -> List[Dict]:
        unique_dicts = {}
        for d in list_of_dicts:
            # 辞書の項目をフローズンセットに変換
            frozen_d = frozenset(d.items())
            # フローズンセットをキーとして、元の辞書を値として保存
            unique_dicts[frozen_d] = d
        return list(unique_dicts.values())

    @staticmethod
    def task_seed_collector(
            tasks_file: str,
            topics_file: str,
            persona_file: str,
            num_to_collect: int = 1,
    ):
        """

        Parameters
        ----------
        tasks_file: *.jsonl
            {"task": "arithmetic"}
            ...
        topics_file: *.jsonl
            {"topic": "三角比"}
            ...
        persona_file: *.jsonl
            {"persona": "とび職"}
            ...
        num_to_collect

        Returns
        -------

        """
        jlh = JsonlHandler()
        task_list = [d["task"] for d in jlh.read(tasks_file)]
        topic_list = [d["topic"] for d in jlh.read(topics_file)]
        persona_list = [d["persona"] for d in jlh.read(persona_file)]
        for _ in tqdm(range(num_to_collect), desc="Generating seeds"):
            seed_set = dict(
                persona=random.choice(persona_list),
                task=random.choice(task_list),
                topic=random.choice(topic_list),
            )
            yield seed_set

    def __call__(
            self,
            task_type: str,  # 'math', 'arithmetic', 'coding', 'reasoning'
            num_to_generate: int,
    ) -> None:
        return self.file_handling(task_type, num_to_generate)

    def file_handling(
            self,
            task_type: str,  # 'math', 'arithmetic', 'coding', 'reasoning'
            num_to_generate: int,
    ) -> None:

        seed_dir = self._seed_dir
        output_dir = self._output_dir
        persona_file = seed_dir + "/persona_jp.jsonl"  # 職業を定義したファイル
        tasks_file = seed_dir + f"/{task_type}_task.jsonl"  # タスクを定義したファイル
        topics_file = seed_dir + f"/{task_type}_topic.jsonl"  # トピックを定義したファイル

        assert os.path.isfile(tasks_file), f"Need '{task_type}_task.jsonl' for seed file containing 'task' key."
        assert os.path.isfile(topics_file), f"Need '{task_type}_topic.jsonl' for seed file containing 'topic' key."
        assert os.path.isfile(persona_file), f"Need 'persona_jp.jsonl' for seed file containing 'persona' key."

        # 以下のようなpersonaとtask、topicの組み合わせを生成する出力先
        # {"persona": "SF作家", "task": "math", "topic": "確率"}
        seed_set_file = output_dir + f"/{task_type}_seeds_{num_to_generate // 1000}k.jsonl"

        # output directory check
        os.makedirs(output_dir, exist_ok=True)

        # file handle tools
        jlh = JsonlHandler()  # tool for *.jesonl files.

        seeds = jlh.read(seed_set_file) if os.path.isfile(seed_set_file) else []
        
        with tqdm(total=num_to_generate, desc="Generating seeds", initial=len(seeds)) as pbar:
            while len(seeds) < num_to_generate:
                ts = self.task_seed_collector(
                    tasks_file,
                topics_file,
                persona_file,
                num_to_collect=self._batch,
            )

                for s in ts:
                    seeds.append(s)
                    pbar.update(1)

            if self._remove_dup:
                seeds = self.remove_duplicate_dicts(seeds)

        jlh.write(seeds, seed_set_file)


if __name__ == "__main__":
    """
    python -m persona_hub.seed_set_gen
    """
    
    random_seed = 10
    fix_seeds(random_seed)

    sc = PersonaHubSeedCollector(
        seed_dir='./data/seed',
        output_dir='./data',
        # batch_size=100,
        batch_size=10,
        remove_duplicate=True,
    )

    task_type = 'math'  # 'math', 'arithmetic', 'coding', 'reasoning'
    # task_type = 'arithmetic'  # 'math', 'arithmetic', 'coding', 'reasoning'
    # task_type = 'coding'  # 'math', 'arithmetic', 'coding', 'reasoning'
    # task_type = 'reasoning'  # 'math', 'arithmetic', 'coding', 'reasoning'
    # num_to_gen = 50_000
    # num_to_gen = 10_000
    # num_to_gen = 1000
    num_to_gen = 100
    sc(task_type, num_to_gen)
