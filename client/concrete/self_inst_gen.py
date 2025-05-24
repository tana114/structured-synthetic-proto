import json
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any
from typing import cast

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from client.chain_base import ChainDirector, ConcreteChainBase

"""

 I have referred to the following
 
https://github.com/tatsu-lab/stanford_alpaca

"""

NO_SIGN = "<noinput>"

''' SYSTEM_PROMPT_FORMAT
{task_num}は生成するタスクの個数（few-shotの数も含まれる）
{no_input}はinputが必要ない場合に仮置きしておく文字列
'''
SYSTEM_PROMPT_FORMAT = (
    "You are a helpful instruction and response writer.\n"
    "You are asked to come up with a set of {tasks_num} structured and diverse list of task instructions."
    "These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.\n"
    "Here are the requirements:\n"
    "- Try not to repeat the verb for each #instruction# to maximize diversity.\n"
    "- The Wording used for the #instruction# also should be diverse."
    " For example, you should combine questions with imperative instructions.\n"
    "- The type of instructions should be diverse."
    " The list of task instructions should include diverse types of tasks like open-ended generation, classification, editing, etc.\n"
    "- A GPT model should be able to complete the #instruction#."
    " For example, do not ask the assistant to create any visual or audio output."
    " For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.\n"
    "- The #instruction# should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.\n"
    "- The #output# must be an appropriate response to the #instruction#,"
    " and make sure the #output# is as detailed and polite as possible.\n"
    "- The #output# should be helpful and informative, but do not hallucinate. Do not make up factual information.\n"
    "- Ensure that the #output# sentences is at least 300 to 500 characters long.\n"
    "- The #instruction# must not be a translation task.\n"
    "- The #instruction# and #output# mast be in Japanese.\n"
)

''' HUMAN_PROMPT_FORMAT
{{few_shot}}と{{next_no}}はinvoke時に与える
{{few_shot}} : few-shot文字列
{{next_no}} : few-shot以降に生成するタスクの開始no
'''
HUMAN_PROMPT_FORMAT = (
    "List of {tasks_num} structured tasks: \n\n{{few_shot}}\n"
    "Generate the list of remaining tasks after no: {{next_no}} onwards following the above.\n"
)


class TaskData(BaseModel):
    """The task record includes instruction, input, and output."""
    no: int = Field(description="List number of the generated task")
    instruction: str = Field(description="Instruction statement.")
    output: str = Field(description="500 to 1000 character output statement.")


class TaskList(BaseModel):
    """List for contains TaskData."""
    tasks: List[TaskData] = Field(description="A list containing multiple TaskData.")


class SelfInstructGenerator(ConcreteChainBase):
    def __init__(
            self,
            chat_model: BaseChatModel,
            num_task_to_generate: int = 10,
            use_gen_num_check: bool = False,
    ):
        """
        :param chat_model:
        :param num_task_to_generate: int
        生成するタスクの数（Few-Shotで例示した個数も含まれる）
        """
        super().__init__()
        self._llm = chat_model
        self._num_gen = num_task_to_generate
        self._use_check = use_gen_num_check

    @staticmethod
    def encode_few_shot_prompt(
            seed_instructions: List[Dict[str, str]]
    ) -> str:
        """
        このクラスでinvokeする際に渡すFew-Shotプロンプト用の文字を生成する

        :param seed_instructions:
        :return:

        seed_instructions =[
            {"instruction": "hoge1", "input": "", "output": "hogefuga1"},
            {"instruction": "hoge2", "input": "fuga2", "output": "hogefuga2"},
            {"instruction": "hoge3", "input": "fuga3, "output": "hogefuga3"},
        ]

        few_shot_prompts ='''
        {
          no: 1,
          instruction: hoge1,
          input: , "<noinput>"
          output: hogefuga1
        }
        {
          ...
          output: hogefuga3
        }
        """
        # Replace '""' of input  with '"<noinput>"' character.
        # seeds = [{k: NO_SIGN if k == 'input' and v == "" else v for k, v in d.items()} for d in seed_instructions]
        # Add key 'no' and give serial number to value.
        seeds = [{"no": i + 1, **d} for i, d in enumerate(seed_instructions)]

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
        system_prompt = SYSTEM_PROMPT_FORMAT.format(tasks_num=self._num_gen, no_input=NO_SIGN)
        human_prompt = HUMAN_PROMPT_FORMAT.format(tasks_num=self._num_gen)

        return ChainDirector(
            chat_model=self._llm,
            system_prompt=system_prompt,
            human_prompt=human_prompt,  # {few-shot}, {next_no}
            struct_type=TaskList,
        )

    def _invoke_handling(
            self,
            input: Dict[Literal["few_shot", "next_no"], str],
            **kwargs
    ) -> List[Dict]:

        if self._use_check:
            res = cast(TaskList, self._inst_num_check(input, **kwargs))
        else:
            chain_d = cast(ChainDirector, self._chain_director)
            res = cast(TaskList, chain_d.invoke(input, **kwargs, ))

            # TaskList型を辞書型にdumpしたものを返す#
        task_list = [d.model_dump() for d in res.tasks]
        # # 'input' の内容がNO_SIGNのままになっている場合は空文字の""に置き換える
        # tg_key = 'input'
        # for d in task_list:
        #     if d[tg_key].lower() == NO_SIGN.lower():
        #         d[tg_key] = ""

        return task_list

    def _inst_num_check(
            self,
            input: Dict[Literal["few_shot", "next_no"], str],
            **kwargs
    ):
        chain_d = cast(ChainDirector, self._chain_director)
        res = cast(TaskList, chain_d.invoke(input, **kwargs, ))
        if not res:
            return self._inst_num_check(input, **kwargs)
        # 要素の数が想定した個数になっているかチェックするためにListの中身を確認
        tasks_list = res.tasks
        tasks_list = [e for e in tasks_list if e.instruction]  # 空文字は削除
        counts = self._num_gen - int(input["next_no"])
        return self._inst_num_check(input, **kwargs) if len(tasks_list) != counts + 1 else res


if __name__ == "__main__":
    """
    python -m client.concrete.self_inst_gen
    """

    from model.groq_llm import GroqChatBase

    llm = GroqChatBase(
        model_name="llama-3.3-70b-versatile",
        # requests_per_second=0.32,
        temperature=0
    )

    seeds = [
        {
            "instruction": "アメリカの国務省が発表した「語学習得難易度ランキング」では日本語は「最も習得に時間がかかる言語」のカテゴリーに分類されています。\n「はやくいって」という言葉を例に、その理由を述べなさい。",
            "output": "日本語を母国語としない人々にとって、「日本語は非常に難しい言語である」と言われています。\nその理由として、\n・漢字、ひらがな、カタカナ、と表記が何種類もある。\n・同一の漢字でも音読みと訓読みがある\n・地名の読みが難しい\n・主語、述語が省略される\nなどが挙げられます。\nそして、やっと基本的な日本語を習得してもさらなる壁が立ちはだかっているのです。\n例えば、「はやくいって」、この言葉がすべて平仮名で書かれていたり会話の中で出てきた場合です。\n外国人のＡ君が大学の講義を終えてアルバイト先に向かっているとき、校門で日本人の友達Ｂ君に出会い進路相談をされ、１時間が経過してしまいます。さすがにもうアルバイトの開始時間に間に合わない！！そこでＡ君はＢ君に急いでいることを伝えると…Ｂ君から「はやくいって！」と言われました。\nＡ君はその言葉の意味を理解しかねてしばし立ち尽くしてしまいます。\n\n「はやくいって」は、「早く言って」と「早く行って」の両方の解釈が出来てしまうのです。\n\n１，アルバイトがあって急いでいるのなら、その旨をB君に「早く言って」ほしかった。\n２，アルバイトがあって急いでいるのなら、アルバイト先に「早く行って」\n\n上記のように、複数の解釈ができる日本語が多く存在しており、「習得に時間がかかる言語である」といわれる理由の一つです。",
        },
        {
            "instruction": "「1匹＝いっぴき」と数えるのに、「2匹→２ぴき」と数えない理由を答えなさい。",
            "output": "日本語には、「助数詞」と言って「数量を表現するのに用いる接尾語」があります。\nこの助数詞は、「種類の多さ」と「音の変化＝変音現象」が特徴です。\n\n紙→枚（まい）、車→台（だい）、列車→輌（りょう）、箸→膳（ぜん）、エンジン→基（き）\n\n種類の多さもさることながら、もう一つの難しさが変音現象です。\n例えば、カエルを数えるときに「1匹＝いっぴき」と読むのに対して「2匹≠にぴき」で2匹を「にぴき」とは読まずに「2匹＝にひき」となります。\nどうしてこうなるのか。\nここには、漢語の半濁音のルール、というものが発生します。\nまず、半濁音はハ行にだけしか付きません。（はひふへほ→ぱぴぷぺぽ）\nそして、連濁といって語と語が合体するときに後ろに来る語の頭の部分が清音から濁音に変化します。\nハ行音の連濁のうち、前の語の最後に「つ」がくると、後ろの語の頭が半濁音に変わります。\n\n例）切（せつ）　＋　腹（ふく）　＝　切腹（せっぷく）\n　\n助数詞も、数字の「一・六・八・十・百」が「ハ行音の助数詞」と結びつく時、促音便「っ（小さい”っ”）」を起こし、後ろの助数詞が半濁音となります。\n1杯（いち　＋　はい　→　いっぱい）、6泊（ろく　＋　はく　→　ろっぱく）、8袋（はち　＋　ふくろ　→　はっぷくろ）、\n10編（じゅう　＋　へん　→　じっぺん）、100本（ひゃく　＋　ほん　→ひゃっぽん）\n\nこういった理由から、「2匹→2ぴき」とはなりません。",
        },
        {
            "instruction": "「花の咲くころ」と「花の色は美しい」に共通して使われている助詞「の」の意味を説明しなさい。",
            "output": "「の」は、まず、助詞であることから「付属語」であり「活用はしない」。\n上記のことを踏まえて、自立語に付属した「の」は次の5通りの使われ方をします。\n１）主格「の」・・・ 主語「が」の意味を添える。「が」に置き換えることができる。\n秋桜の咲く季節が好きだ。　→　秋桜が咲く季節が好きだ。\n\n２）連体修飾格「の」・・・連体修飾語（助詞「の」の真下の体言を修飾する）。名詞と名詞をつなぐ。\n\n３）同格「の」・・・ A=Bの関係になるもの。「?で」と置き換えることができる。\n白き鳥の、嘴と脚と赤き鳥　　→　白い鳥で嘴と脚とが赤い鳥（白い鳥＝嘴と脚とが赤い鳥）\n\n４）体言の代用「の」・・・体言の代わりに用いるもので、 「?のもの」・「?のこと」に置き換えることができる。\n食べるのが好きです。　→　食べることが好きです。\n\n５）連用格「の」・・・連用修飾語をつくり、比喩表現となる。「?のように」に置き換えることができる。\nしだり尾の　ながながし夜を　ひとりかも寝む　→（鳥の）長い尾のように長い秋の夜を私は一人で寝るのであろうか。\n\n現代文では主に１）、２）、４）が用いられ、\n・「花の咲くころ」の「の」は「花が咲くころ」と言い換えることができ、助詞「の」の主格の働きをするものになります。\n・「花の色は美しい」の「の」は、「の」の下に体言である「色」が来ています。名詞＋名詞となり、名詞同士をつなぐ連体修飾格の働きをするものになります。",
        },
        {
            "instruction": "こそあど言葉は代名詞だと習ったのですが、この前の模試では不正解になりました。どうしてですか。",
            "output": "こそあど言葉＝指示語は、4つの品詞（名詞・連体詞・形容動詞・副詞）に分類することができるのです。\nこれら4つの品詞を見分けるには、以下の点に注目します。\n1）名詞（代名詞）・・・前に出てきた名詞を指す場合。\n美味しそうな林檎だね。それはいつ食べるの？【それ＝林檎】\n\n2）連体詞・・・指示語の下に名詞が来ている場合。\nその林檎は美味しそうだね。【その+林檎（名詞）】\n\n3）形容動詞・・・「こんなだ・そんなだ・あんなだ・どんなだ」の4語で、指示語の下に来る語の説明をしている場合。一番間違えやすい品詞になるので、注意が必要です。\nこんなに食べられない。\n\n4）副詞・・・指示語の下に動詞が来ている場合。\nそう言う君は誰だい。【そう+言う（動詞）】\n\nとなります。\nよって、こそあど言葉が代名詞だと習ったときは、1）のパターンで文章が成り立っていたのでしょう。\nそして、模試では2）～4）のいずれかのパターンで出題されていたと考えられます。"

        },
    ]

    gen = SelfInstructGenerator(
        chat_model=llm,
        num_task_to_generate=8,
        use_gen_num_check=True,
    )
    # few-shotの文字列を作成
    few_shot = gen.encode_few_shot_prompt(seeds)
    print(few_shot)
    next_no = len(seeds) + 1

    inst = {
        "few_shot": few_shot,
        "next_no": next_no,
    }

    result = gen(inst)
    # print(result)

    for r in result:
        print(r['no'])
        print(r['instruction'])
        print(len(r['output']))
        print(r['output'])
        print('------------------------')
