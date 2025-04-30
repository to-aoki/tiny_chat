from abc import ABC
import textwrap


class QueryPreprocessor(ABC):

    def transform(self, query=None):
        return None


class HypotheticalDocument(QueryPreprocessor):

    def __init__(self, openai_client, model_name, temperature, top_p, prefix="検索文章: "):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.prefix = prefix

    def transform(self, query=None):
        messages_for_api = [
            {
                "role": "user",
                "content": query
            }
        ]
        try:
            # LLMの返答をクエリとして採用する
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False
            )

            return self.prefix + response.choices[0].message.content
        except:
            # 例外時はそのまま返す
            return query


class StepBackQuery(QueryPreprocessor):

    def __init__(self, openai_client, model_name, temperature, top_p):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def transform(self, query=None):
        step_back_extract = textwrap.dedent(f"""\
            # ステップバック質問
            あなたは知識検索の専門家です。あなたのタスクは、与えられた元の検索クエリを一歩下がって、より一般的で、より高レベルで、回答しやすい「ステップバック質問」に言い換えることです。
            ステップバック質問は、元のクエリに直接答えるために必要な全体的なコンテキストや基本的な情報、原則を取得するのに役立ちます。

            ## 例を挙げます
            元の検索クエリ: 明治時代に活躍した夏目漱石は、1905年から1907年の間にどのような作品を発表しましたか？
            ステップバック質問: 夏目漱石の主要な著作は何ですか？

            元の検索クエリ: 2023年のWBCで優勝した国の決勝戦の対戦相手はどこですか？
            ステップバック質問: 2023年のWBCで優勝した国はどこですか？

            元の検索クエリ: 京都にある清水寺が現在の形になったのは西暦何年ですか？
            ステップバック質問: 京都の清水寺の歴史を教えてください。

            ## 以下のステップバック質問を記述しなさい。質問文のみ応答してください
            元の検索クエリ: {query}
            ステップバック質問:
            """
        )
        messages_for_api = [
            {
                "role": "user",
                "content": step_back_extract
            }
        ]
        try:
            # LLMの返答をクエリとして採用する
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False
            )

            return response.choices[0].message.content
        except:
            import traceback
            traceback.print_exc()
            # 例外時はそのまま返す
            return query
