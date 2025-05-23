# structured synthetic prototype

## 概要

構造化出力を使用して大規模言語モデル（LLM）の生成形式を制御することを試みています。
構造化出力により、1度の生成で複数の合成データを生成する方法を示します。

### 機能

[`persona_hub/gen_manager.py`](persona_hub/manager.py:37)では、構造化出力を使用して、persona_hubに基づいたLLMとのマルチターンの会話を生成しています。
一度に複数のマルチターン回答を生成できるため生成効率が向上しますが、その分精度と安定性は悪化する傾向があるようです。

### 実行方法

[`persona_hub/gen_manager.py`](persona_hub/manager.py:37)を実行するには、次のコマンドを使用します。

```bash
python -m persona_hub.gen_manager
```

スクリプトが実行され、提供されたシードデータに基づいて会話が生成されます。

### 例

以下は、ローカル環境に構築した`qwen3:4b`モデルを使用して`persona_hub/manager.py`で生成されたマルチターンの会話の例です。生成時に与えたシードは
`{"persona": "キャラクターデザイナー", "task": "math", "topic": "データの相関"}`。

```json
{
  "q1": "キャラクターデザイナーは、2010年から2020年にかけて、毎年の人気キャラクターの数を記録した。このデータをもとに、キャラクターの数と年数の相関関係を調べるために、相関係数を計算しなさい。",
  "a1": "キャラクターの数と年数の相関係数を計算するためには、データの平均値、分散、共分散を求めます。まず、年数とキャラクターの数のデータをまとめ、それぞれの平均値を算出します。",
  "q2": "キャラクターデザイナーが2010年から2020年にかけて毎年記録したキャラクターの数のデータは以下の通りです：2010年: 100, 2011年: 120, 2012年: 150,  qualities: 2013年: 170, 2014年: 200, 2015年: 220, 2016年: 250, 2017年: 270, 2018年: 300, 2019年: 320, 2020年: 350。このデータを使って、相関係数を求める際、どのデータを用いるべきですか？",
  "a2": "相関係数を求めるには、年数とキャラクターの数のデータを対応させた対を用います。2010年から2020年にかけてのデータをすべて使用し、それぞれの年数とキャラクターの数の対を並べて計算します。"
}

```

### その他

Groq APIを用いる場合は`.env`ファイルを作成しkeyを設定してください:

   ```
   GROQ_API_KEY=your_api_key_here
   ```
