from abc import ABC
import re
import json
from pydantic import BaseModel
from tiny_chat.utils.llm_utils import convert_openai_response_format

# DeepSeek-R1/Qwen3向け
THINK_PATTERN = r"^<think>[\s\S]*?</think>"


class QueryPreprocessor(ABC):

    def transform(self, query=None):
        return None


class HypotheticalDocument(QueryPreprocessor):
    # Precise Zero-Shot Dense Retrieval without Relevance Labels
    # https://arxiv.org/abs/2212.10496

    def __init__(self, openai_client, model_name, temperature, top_p, prefix="検索文章: ", meta_prompt=None):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.prefix = prefix  # embeddings(ruri)都合で変更する
        self.meta_prompt = meta_prompt

    def transform(self, query=None):
        N = len(query) * 2
        messages_for_api = [
            {
                "role": "user",
                "content": "タスクは、与えられた元の検索クエリに対して、"
                           "該当する文書内容を例示することです。文書内容の例示はデータベースの類似度検索に利用されます。\n"
                           f"以後、「検索クエリ:」に対しての文書内容のみ{N}文字で簡潔に記述してください。返信や補足説明は不要です。"
            },
            {
                "role": "assistant",
                "content": "わかりました。適切に文書内容例を記述します。"
            },
            {
                "role": "user",
                "content": "検索クエリ: 茶道を体験したいです。京都で初心者が楽しめる場所はありますか？"
            },
            {
                "role": "assistant",
                "content": "京都で初めての茶道体験。初心者でも安心して参加できる、英語対応可能な教室を紹介します。"
                           "抹茶の点て方から和菓子の頂き方まで、基本的な作法を丁寧に学べます。"
                           "美しい庭園を眺めながら、静かなひとときをお過ごしください。"
            },
            {
                "role": "user",
                "content": "検索クエリ: 折り紙 鶴 簡単な折り方 子供向け"
            },
            {
                "role": "assistant",
                "content": "簡単に折れる『折り鶴』の折り方をステップバイステップで解説。分かりやすいイラストと写真付き。"
                           "平和の象徴でもある鶴を親子で一緒に作ってみましょう。必要な道具は折り紙一枚だけです。"
            },
            {
                "role": "user",
                "content": "検索クエリ: リアルタイムOSでタスクをスケジューリングするアルゴリズムが知りたいです。"
            },
            {
                "role": "assistant",
                "content": "組み込みシステム向けリアルタイムOS (RTOS) におけるタスクスケジューリングアルゴリズムを比較。"
                           "優先度ベースプリエンプティブ方式、ラウンドロビン方式、Rate Monotonic (RM)、Earliest Deadline First (EDF) の特徴、"
                           "応答性、スケジューラビリティ解析について解説。"
            },
        ]

        if self.meta_prompt is not None:
            messages_for_api.append(
                {
                    "role": "system",
                    "content": self.meta_prompt
                },
            )

        messages_for_api.append(
            {
                "role": "user",
                "content": "検索クエリ: " + query
            },
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=N,  # FIXME 出力長
                stream=False
            )

            # DeepSeek-R1/Qwen3
            response_text = re.sub(THINK_PATTERN, "", response.choices[0].message.content)

            return self.prefix + response_text
        except:
            # 例外時はそのまま返す
            return query


class StepBackQuery(QueryPreprocessor):
    # Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
    # https://arxiv.org/abs/2310.06117

    def __init__(self, openai_client, model_name, temperature, top_p, meta_prompt=None):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.meta_prompt = meta_prompt

    def transform(self, query=None):
        N = int(len(query) * 1.5)
        messages_for_api = [
            {
                "role": "user",
                "content": "タスクは、与えられた検索クエリを一歩下がって、"
                           "より一般的で、より高レベルで、回答しやすい「ステップバック質問」に言い換えることです。\n"
                           "ステップバック質問は、検索クエリに直接答えるために必要な全体的なコンテキストや基本的な情報、"
                           f"原則を取得するのに役立ちます。\n以後、「検索クエリ:」に対応するステップバック質問のみ{N}文字で簡潔に記述してください。返信や補足説明は不要です。"
            },
            {
                "role": "assistant",
                "content": "わかりました。適切に「ステップバック質問」を記述します。"
            },
            {
                "role": "user",
                "content": "検索クエリ: 明治時代に活躍した夏目漱石は、1905年から1907年の間にどのような作品を発表しましたか？"
            },
            {
                "role": "assistant",
                "content": "夏目漱石の主要な著作は何ですか？"
            },
            {
                "role": "user",
                "content": "検索クエリ: 2023年のWBCで優勝した国の決勝戦の対戦相手はどこですか？"
            },
            {
                "role": "assistant",
                "content": "2023年のWBCで優勝した国はどこですか？"
            },
            {
                "role": "user",
                "content": "検索クエリ: 京都にある清水寺が現在の形になったのは西暦何年ですか？"
            },
            {
                "role": "assistant",
                "content": "京都の清水寺の歴史を教えてください。"
            },
        ]

        if self.meta_prompt is not None:
            messages_for_api.append(
                {
                    "role": "system",
                    "content": self.meta_prompt
                },
            )

        messages_for_api.append(
            {
                "role": "user",
                "content": "検索クエリ: " + query
            },
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=N,  # FIXME 出力長
                stream=False
            )
            # DeepSeek-R1/Qwen3
            response_text = re.sub(THINK_PATTERN, "", response.choices[0].message.content)

            return response_text
        except:
            # 例外時はそのまま返す
            return query


class QueryResponse(BaseModel):
    query: str
    reason: str

class QueryResponseList(BaseModel):
    queries: list[QueryResponse]


class QueryPlanner:

    def __init__(self, openai_client, model_name, temperature, top_p, meta_prompt=None,
                 is_vllm=False, generate_queries=3):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.meta_prompt = meta_prompt
        self.is_vllm = is_vllm
        self.generate_queries = generate_queries if generate_queries > 1 else 3

    def transform(self, query=None):
        messages_for_api = [
            {
                "role": "user",
                "content": "タスクは与えられた質問を分解し、質問応答のために不足している情報を特定し、"
                           f"不足情報を取得するための具体的な検索クエリとそのクエリ作成理由を最大{self.generate_queries}個記述することです。"
                           "以後、「質問:」に対応する検索クエリ情報をjson形式で記述してください。返信や補足説明は不要です。"
            },
            {
                "role": "assistant",
                "content": "わかりました。適切に質問応答するための不足情報考慮し、クエリと作成理由をjson形式で記述します。"
            },
            {
                "role": "user",
                "content": "質問: 茶道を体験したいです。京都で初心者が楽しめる場所はありますか？"
            },
            {
                "role": "assistant",
                "content": """{
                    "queries": [
                        {
                            "query": "京都で茶道体験ができる場所はどこですか？",
                            "reason": "ユーザーの主な要望である「京都」「茶道体験」「初心者」を直接的に組み合わせた検索で、基本的な選択肢を得るため。",
                        },
                        {
                            "query": "京都の観光客向け茶道教室の予約が取りたいです。",
                            "reason": "「体験」よりも少し本格的な「教室」の情報や、観光客が利用しやすい予約システムの有無を確認するため。",
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": "質問: PythonでWebスクレイピングを始めるための基本的なステップと、おすすめのライブラリを教えてください。"
            },
            {
                "role": "assistant",
                "content": """{
                    "queries": [
                        {
                            "query": "Pythonで実現するWebスクレイピングのプログラミングガイドはありますか？",
                            "reason": "「基本的なステップ」という質問に答えるため、網羅的かつ初心者向けの解説記事やチュートリアルを探す。",
                        },
                        {
                            "query": "Pythonでスクレイピングする場合のライブラリについて比較したいです。Requests、BeautifulSoup、Seleniumどれが優れていますか？",
                            "reason": "「おすすめのライブラリ」という質問に対し、主要な選択肢の特徴や長所・短所を比較検討できる情報を得るため。",
                        },
                        {
                            "query": "Webスクレイピングでインターネットサイトをクロールするときのマナーや注意点を教えてください。",
                            "reason": "Webスクレイピングを実用する上で不可欠な、倫理的・法的な側面や技術的な配慮事項に関する情報を補足するため。",
                        }
                    ]
                }"""
            },
            {
                "role": "user",
                "content": "質問: 健康的な食生活を送るために、1日に摂取すべきタンパク質の量と、タンパク質を多く含む食品の例を教えてください。特に運動習慣がある成人男性の場合でお願いします。"
            },
            {
                "role": "assistant",
                "content": """{
                    "queries": [
                        {
                            "query": "成人男性が一日必要とするでタンパク質の摂取量は？",
                            "reason": "質問の主要な要素である「運動習慣のある成人男性」に特化した「1日のタンパク質摂取量」の具体的な数値や計算方法を調べるため。",
                        },
                        {
                            "query": "高タンパク質を含む食品の一覧を示してください。含有量が記載されているものがいいです。",
                            "reason": "「タンパク質を多く含む食品の例」という要求に応え、具体的な食品名とそれぞれのタンパク質量をリストアップするため。",
                        },
                        {
                            "query": "筋トレで効率的なタンパク質摂取タイミングを教えてください。",
                            "reason": "「運動習慣がある」という文脈を考慮し、タンパク質摂取のタイミングが運動効果（特に筋力トレーニング）にどう影響するかという補足情報を提供するため。",
                        }
                    ]
                }"""
            },
        ]
        if self.meta_prompt is not None:
            messages_for_api.append(
                {
                    "role": "system",
                    "content": self.meta_prompt
                },
            )

        messages_for_api.append(
            {
                "role": "user",
                "content": "質問: " + query
            }
        )

        try:
            if self.is_vllm:
                # vllmはOpenAI APIのresponse_formatに対応していないが、
                # openai-like api?の仕様のextra_bodyには対応している
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_for_api,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    extra_body={"guided_json": QueryResponseList.model_json_schema()},
                )
            else:
                # ollamaはresponse_formatにも対応している
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_for_api,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    response_format=convert_openai_response_format(QueryResponseList),
                )
            json_response = json.loads(response.choices[0].message.content)
            result = QueryResponseList(**json_response)
            return result
        except Exception as e:
            # 例外時はそのまま返す
            return QueryResponseList(queries=[QueryResponse(query=query, reason="")])

    @classmethod
    def result_merge(cls, full_result: list[list]):
        merged_result = []
        seen_keys = set()

        if not full_result:
            return merged_result

        max_inner_list_length = 0
        for result_list in full_result:
            if len(result_list) > max_inner_list_length:
                max_inner_list_length = len(result_list)

        for item_idx in range(max_inner_list_length):
            for result_list in full_result:
                # 入力リストの順序を維持し、なるべくスコアの良い結果をlist上位に持つ（後で切る可能性がある）
                if item_idx < len(result_list):
                    item = result_list[item_idx]

                    source = item.payload.get('source')
                    page = item.payload.get('page')
                    key = (source, page)

                    if key in seen_keys:
                        continue

                    merged_result.append(item)
                    seen_keys.add(key)

        return merged_result