import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from typing import cast
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

BASE_SYSTEM_PROMPT_FORMAT = (
    "You are a helpful instruction writer.\n"
    "Create a #task# problem related to the #topic# and #persona#, and generate responses to the created problems.\n\n"
    "Here are the requirements:\n"
    "- Do not include personal information, such as individual names, in the problems you create."
    " Use information of the #persona# instead of the individual's name as the subject.\n"
    "- All outputs must be in Japanese, not in English.\n\n"
    "**STEP1**\n"
    "- Create a #first_generated_problem# using #persona#, #task#, and #topic# corresponding to the given #seed_id#."
    " Do not use information associated with any other #seed_id#.\n"
    "- You should make full use of the #persona# and the #topic# description to create the #first_generated_problem#"
    " to ensure that the #task# problem is unique.\n"
    "- The #first_generated_problem# must contain enough specific information to answer the question,"
    " but should not contain a solution to the problem.\n"
    "- The #first_generated_problem# should be simple and involve basic #task# skills and knowledge."
    " Any average grade school student can solve it correctly.\n"
    "- Think step by step to generate the #first_response_to_problem#. Do not hallucinate. Do not make up factual information.\n"
    "- The #first_response_to_problem# must be an answer derived solely from the #first_generated_problem#."
    " There must be no direct reference to #topic# or #task# in the answer to the #first_generated_problem#.\n\n"
    "**STEP2**\n"
    "- Create a brief additional question to better understand the #first_generated_problem#."
    " You can change part of the #first_generated_problem# or add additional conditions.\n"
    "- Think step by step to generate the #second_response_to_problem# and answer briefly."
    " Do not hallucinate. Do not make up factual information.\n"
    "- The #second_response_to_problem# must be an answer derived solely from the #first_generated_problem#,"
    " #first_response_to_problem#, and #second_additional_problem#."
    " There must be no direct reference to #topic# or #task# in the answer to the #second_additional_problem#.\n\n"
    "The user will give you a #seed_list# that containing several structured sets of #seed_id#,"
    " #task#, #persona# and #topic#.\n"
    "Repeat process STEP1 to STEP2 for the number of elements in the #seed_list# to generate a multiple structured list"
    " containing sets of #output_id#, #first_generated_problem#, #first_response_to_problem#,"
    " #second_additional_problem# and #second_response_to_problem#.\n\n"
)

HUMAN_PROMPT_FORMAT = (
    "#seed_list#: \n{persona_info}\n"
)


class TaskData(BaseModel):
    """The task record includes id and generated problem."""
    output_id: int = Field(description="List number of the generated problem")
    first_generated_problem: str = Field(description="Generated instruction statement.")
    first_response_to_problem: str = Field(description="Response to the generated instruction.")
    second_additional_problem: str = Field(
        description="Additional instruction to understand the first_generated_problem better.")
    second_response_to_problem: str = Field(description="Response to the additional instruction.")


class TaskList(BaseModel):
    """List for contains TaskData."""
    tasks: List[TaskData] = Field(description="A list containing multiple TaskData.")


class PersonaMultiGenerator(ConcreteChainBase):
    """
    I have referred to the following
    https://github.com/matsuolab/nedo_project_code/tree/team_hatakeyama_phase2/team_hatakeyama_phase2/ota/topic-hub

    This module can generate a multi-turn conversation given a seed of 'persona', 'task', and 'topic'.

    ＜例＞
    - シード
        'persona': "医者",
        'task': "math",
        'topic': "多項式の乗法",

    - 生成されるQアンドA
        医者のPersonaは患者さんの血液検査結果の解析を行うために多項式の乗法を使用しています。具体的には、ある特定の抗原が存在するかどうかを判定するために多項式 f(x) = x^3 + 4x + 1 を使います。もし、この多項式と g(x) = x^2 - x + 5 の積を求めるとどのような形になりますか？
        まず、f(x) と g(x) の各項をそれぞれ掛け合わせていきます。最初にx^3とx^2 を掛けると x^5 となります。
        この結果から、最も大きい次数の項は何になりますか？
        最も大きい次数の項は f(x) の最高次の項である x^3 と g(x) の最高次の項である x^2 を掛けた x^5 となります。
    """

    def __init__(
            self,
            chat_model: BaseChatModel,
            use_gen_num_check: bool = False,
    ):
        """
        Parameters
        ----------
        chat_model
        use_gen_num_check: bool
            生成するタスクの数が、与えたseedの要素数と一致しているかを確認し、一致しない場合は自動的に再試行する
        """
        super().__init__()
        self._llm = chat_model
        self._use_check = use_gen_num_check

    @staticmethod
    def encode_few_shot_prompt(
            seed_instructions: List[Dict[str, str]]
    ) -> str:
        # Add key 'seed_id' and give serial number to value.
        seeds = [{"seed_id": i + 1, **d} for i, d in enumerate(seed_instructions)]

        few_shot_prompt = ""
        # convert dict type to string.
        for d in seeds:
            few_shot_prompt += json.dumps(d, indent=2, ensure_ascii=False)
            few_shot_prompt += "\n"

        return few_shot_prompt

    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        system_prompt = BASE_SYSTEM_PROMPT_FORMAT
        human_prompt = HUMAN_PROMPT_FORMAT

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {persona_info}
            struct_type=TaskList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["persona_info"], str],
            **kwargs
    ) -> List[Dict]:

        if self._use_check:
            res = cast(TaskList, self._inst_num_check(input, **kwargs))

        else:
            chain_d = cast(ChainDirector, self._chain_director)
            res = cast(TaskList, chain_d.invoke(input, **kwargs, ))

        # TaskList型を辞書型にdumpしたものを返す#
        task_list = [d.model_dump() for d in res.tasks]

        return task_list

    def _inst_num_check(
            self,
            input: Dict[Literal["persona_info"], str],
            **kwargs
    ):
        chain_d = cast(ChainDirector, self._chain_director)
        res = cast(TaskList, chain_d.invoke(input, **kwargs))
        if not res:
            return cast(TaskList, self._inst_num_check(input, **kwargs))

        # 文字列に含まれている要素の数を確認
        persona_info_s = input['persona_info']
        # 文字列を個別のJSONオブジェクトに分割して要素数を数える
        objects_num = len(persona_info_s.strip().split('\n}\n'))
        # 要素の数が想定した個数になっているかチェックするためにListの中身を確認
        tasks_list = res.tasks
        # tasks_list = [e for e in tasks_list if e.generated_problem]  # 空文字は削除
        return self._inst_num_check(input, **kwargs) if len(tasks_list) != objects_num else res


if __name__ == "__main__":
    """
    python -m client.concrete.persona_multiturn_gen
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.3-70b-versatile",
        # requests_per_second=0.32,
        # temperature=0,
        temperature=0.6,
    )

    per_list = [
        {
            'persona': "カメラマン",
            'task': "数学",
            'topic': "二進法",
        },
        {
            'persona': "医者",
            'task': "math",
            'topic': "多項式の乗法",
        },
        {
            'persona': "電子計算機操作員",
            'task': "reasoning game",
            'topic': "数量推理",
        },
        {
            'persona': "会計士",
            'task': "推理クイズ",
            'topic': "命題と対偶の真偽関係",
        },
        {
            'persona': "酪農家",
            'task': "math",
            'topic': "二進法",
        },
    ]

    gen = PersonaMultiGenerator(
        chat_model=llm,
        use_gen_num_check=True,
    )

    # few-shotの文字列を作成
    few_shot = gen.encode_few_shot_prompt(per_list)

    inst = {
        "persona_info": few_shot,
    }

    result = gen(inst)

    for r in result:
        print('------', r['output_id'], '------')
        print(r['first_generated_problem'])
        print(r['first_response_to_problem'])
        print(r['second_additional_problem'])
        print(r['second_response_to_problem'])

'''
------ 1 ------
カメラマンのPersonaは、写真撮影時に必要な露出時間（シャッタースピード）を計算するために二進法を使用しています。彼が持っている2つのシャッタースピードは、0.5秒と1秒です。もし1秒と0.5秒を足したものを表す二進数で表現すると何になりますか？
まず、1秒の二進数表現は 1 となります。次に、0.5秒の二進数表現は 0.1 です。これらを足してみると 1.1 となります。
もし0.25秒と0.125秒を加えた場合、その合計時間を二進数で表すと何になりますか？
0.25秒は二進数では 0.01 となり、0.125秒は 0.001 となります。これらを足してみると 0.011 となります。
------ 2 ------
医者のPersonaは患者さんの血液検査結果の解析を行うために多項式の乗法を使用しています。具体的には、ある特定の抗原が存在するかどうかを判定するために多項式 f(x) = x^3 + 4x + 1 を使います。もし、この多項式と g(x) = x^2 - x + 5 の積を求めるとどのような形になりますか？
まず、f(x) と g(x) の各項をそれぞれ掛け合わせていきます。最初にx^3とx^2 を掛けると x^5 となります。
この結果から、最も大きい次数の項は何になりますか？
最も大きい次数の項は f(x) の最高次の項である x^3 と g(x) の最高次の項である x^2 を掛けた x^5 となります。
------ 3 ------
電子計算機操作員のPersonaは、ある問題を解決するために数量推理を使用します。例えば、あるバッテリーが10%から20%まで充電された場合、これに等しい割合は何パーセントですか？
10％から20％までの増加は10％の増加となります。
もしバッテリーが40％から50％まで充電された場合、これに等しい割合は何パーセントですか？
40％から50％までの増加も10％の増加となります。
------ 4 ------
会計士のPersonaは、財務報告書を作成する際に命題と対偶の真偽関係を使用します。ある商品が利益を生むとき（P）、「その商品の価格が高い」(Q)という主張は必ずしも真であるとは限らない。この場合、逆の場合「商品の価格が高い(Q)ならば、それが利益を生む(P)」と主張するのはどのように評価されますか？
主張P: 商品が利益を生むこと
主張Q: その商品の価格が高い
Pが真であるからといって必ずしもQが真とは限らないため、対偶「QならばP」は真であるとは限りません。
もし「商品の価格が高い(Q)ならば、それが利益を生む(P)」という主張が常に真であれば、「その商品が利益を生む(P)」と主張するのはどのように評価されますか？
対偶が真である場合、つまりQならばPが常に成り立つとすれば、その商品は必ず利益を生むことになります。これは「商品が利益を生む」という主張が常に正しいことを意味します。
------ 5 ------
酪農家のPersonaは牛乳の成分分析を行う際などに二進法を使用します。あるタンパク質の存在をチェックするための判定式が010101（これは2進数で表現された値です）であるとするとき、この数値は何ですか？
与えられた2進数の数値は010101です。これを10進数に変換すると5 + 2 = 7となります。
もしタンパク質の存在をチェックするための判定式が110101（これもまた2進数で表現された値です）であるとしたら、この数値は何ですか？
与えられた二進数は110101であり、これを十進数に変換すると1 + 4 + 32 = 37となります。

'''
