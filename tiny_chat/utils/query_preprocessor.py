from abc import ABC
import re
import json
from typing import Optional
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
    reason: str
    query: str


class QueryResponseList(BaseModel):
    queries: list[QueryResponse]


class QueryEvaluateResponse(BaseModel):
    valid_index: list[int] = []
    knowledge: Optional[str] = ""
    search_needed: bool = False
    new_query: Optional[str] = None


class QueryPlanner:

    def __init__(self, openai_client, model_name, temperature, top_p, meta_prompt=None,
                 is_vllm=False, generate_queries=3, logger=None):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.meta_prompt = meta_prompt
        self.is_vllm = is_vllm
        self.generate_queries = generate_queries if generate_queries > 1 else 3
        self.logger = logger

    def _call_llm(self, messages_for_api, pydantic_class):
        if self.is_vllm:
            # vllmはOpenAI APIのresponse_formatに対応していないが、extra_bodyには対応している
            # FIXME 2025/05/11 現在　v0エンジンのみ対応
            # https://docs.vllm.ai/en/stable/features/reasoning_outputs.html#structured-output
            # It is only supported in v0 engine now.
            # VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
            #     --enable-reasoning --reasoning-parser deepseek_r1
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
                extra_body={"guided_json": pydantic_class.model_json_schema()},
            )
        else:
            # ollamaはresponse_formatにも対応している
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
                response_format=convert_openai_response_format(pydantic_class),
            )
        return response

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
                            "reason": "ユーザーの主な要望である「京都」「茶道体験」「初心者」を直接的に組み合わせた検索で、基本的な選択肢を得るため。",
                            "query": "京都で茶道体験ができる場所はどこですか？",
                        },
                        {
                            "reason": "「体験」よりも少し本格的な「教室」の情報や、観光客が利用しやすい予約システムの有無を確認するため。",
                            "query": "京都の観光客向け茶道教室の予約が取りたいです。",
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
                            "reason": "「基本的なステップ」という質問に答えるため、網羅的かつ初心者向けの解説記事やチュートリアルを探す。",
                            "query": "Pythonで実現するWebスクレイピングのプログラミングガイドはありますか？",
                        },
                        {
                            "reason": "「おすすめのライブラリ」という質問に対し、主要な選択肢の特徴や長所・短所を比較検討できる情報を得るため。",                        
                            "query": "Pythonでスクレイピングする場合のライブラリについて比較したいです。Requests、BeautifulSoup、Seleniumどれが優れていますか？",
                        },
                        {
                            "reason": "Webスクレイピングを実用する上で不可欠な、倫理的・法的な側面や技術的な配慮事項に関する情報を補足するため。",
                            "query": "Webスクレイピングでインターネットサイトをクロールするときのマナーや注意点を教えてください。",
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
                            "reason": "質問の主要な要素である「運動習慣のある成人男性」に特化した「1日のタンパク質摂取量」の具体的な数値や計算方法を調べるため。",
                            "query": "成人男性が一日必要とするでタンパク質の摂取量は？",
                        },
                        {
                            "reason": "「タンパク質を多く含む食品の例」という要求に応え、具体的な食品名とそれぞれのタンパク質量をリストアップするため。",
                            "query": "高タンパク質を含む食品の一覧を示してください。含有量が記載されているものがいいです。",
                        },
                        {
                            "reason": "「運動習慣がある」という文脈を考慮し、タンパク質摂取のタイミングが運動効果（特に筋力トレーニング）にどう影響するかという補足情報を提供するため。",
                            "query": "筋トレで効率的なタンパク質摂取タイミングを教えてください。",
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
            response = self._call_llm(messages_for_api, QueryResponseList)
            json_response = json.loads(response.choices[0].message.content)
            result = QueryResponseList(**json_response)
            return result
        except Exception as e:
            # 例外時はそのまま返す
            if self.logger:
                import traceback
                error_trace_str = traceback.format_exc()
                self.logger.error("クエリプラン変換エラー: " + error_trace_str)
            return QueryResponseList(queries=[QueryResponse(reason="クエリプラン変換エラー", query=query)])

    @classmethod
    def result_merge(cls, full_result: list[list], black_list=None):
        merged_result = []
        seen_keys = set()
        if black_list is None:
            black_list = set()

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
                    if key in black_list:
                        continue
                    if key in seen_keys:
                        continue

                    merged_result.append(item)
                    seen_keys.add(key)

        return merged_result

    def evaluate(self, question: str, query: QueryResponse, search_results=None,
                 knowledge="", exists_valid_list=None, black_list=None):
        if question is None or question == '':
            raise ValueError('require question')

        if query is None or query.query == '' or query.reason == '':
            raise ValueError('require query')

        original_indices = list(range(len(search_results)))
        if exists_valid_list is None:
            exists_valid_list = set()

        if black_list is None:
            black_list = set()

        white_search_result = []
        for i, result in enumerate(search_results):
            source = result.payload.get('source')
            page = result.payload.get('page')
            key = (source, page)
            if key in black_list:
                continue
            if key in exists_valid_list:
                continue
            white_search_result.append(result)
            original_indices.append(i)

        search_results_text = "検索結果はありません"

        if len(white_search_result) > 0:
            search_results_text = ""
            for i, r in enumerate(white_search_result):
                search_results_text +=f"[{i+1}] {r.payload.get('source')} page:{r.payload.get('page', '-')}\n```\n{r.payload.get('text')} \n```\n\n"

        request_content = f"""# タスク"
                           "タスクは与えられた質問と検索理由（reason）に対して、有効な検索結果があるか確認し、検索結果が妥当であれば質問回答に役立つ知識（knowledge）を更新することです。\n"
                           "有効な検索結果があった場合は、その検索結果のインデックスを有効なインデックス（valid_index）として列挙してください。"
                           "また、クエリは検索結果を得るためにデータベースに発行したクエリです。"
                           "検索理由を満たした検索結果が無い、あるいは追加で検索が必要な場合は、再検索フラグをtrueとして、検索理由に基づき新しいクエリを記述してください。"
                           "上記タスク結果をはjson形式で以下のように記述してください。"
                           "各要素は {{"}} 
                           "valid_index": "knowledge記述に有効だった検索結果のインデックス番号をリスト記載"
                           "knowledge": "質問回答に役立つ検索結果があった場合に、知識を検索結果に基づき抜粋して記述する。記述事実に基づき記載し、既存知識がある場合はそれも踏まえて更新すること"
                           "search_needed": "再検索フラグ。再検索する場合はtrueとする"
                           "new_query": "再検索が必要な場合は、検索理由からにクエリと重複しない新しい検索クエリを記述。疑問点や要望などの補足説明は記載しない"" {{"}} の形式であるべきです。"
                           ""
                           "## 質問"
                           "{question}"
                           ""
                           "## 知識(knowledge)"
                           "{question}"
                           ""
                           "## 検索理由(reason)"
                           "{query.reason}"
                           ""
                           "## クエリ(query)"
                           "{query.query}"
                           ""
                           "## 検索結果"
                           "{search_results_text}" 
                           """

        messages_for_api = []
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
                "content": request_content
            },
        )

        try:
            response = self._call_llm(messages_for_api, QueryEvaluateResponse)
            json_response = json.loads(response.choices[0].message.content)
            evaluate_result = QueryEvaluateResponse(**json_response)
            new_knowledge = evaluate_result.knowledge
            new_query = None
            if evaluate_result.search_needed:
                new_query = QueryResponse(reason=query.reason, query=evaluate_result.new_query)

            valid_results = []
            for i in original_indices:
                result = search_results[i]
                source = result.payload.get('source')
                page = result.payload.get('page')
                key = (source, page)
                if (i+1) in evaluate_result.valid_index:
                    valid_results.append(result)
                    exists_valid_list.add(key)
                else:
                    black_list.add(key)

            return new_knowledge, new_query, valid_results, exists_valid_list, black_list
        except Exception as e:
            if self.logger:
                import traceback
                error_trace_str = traceback.format_exc()
                self.logger.error("クエリ評価エラー: " + error_trace_str)
            return knowledge, None, search_results, exists_valid_list, black_list

