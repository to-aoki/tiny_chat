from abc import ABC


class QueryPreprocessor(ABC):

    def transform(self, query=None):
        return None


class HypotheticalDocument(QueryPreprocessor):
    # Precise Zero-Shot Dense Retrieval without Relevance Labels
    # https://arxiv.org/abs/2212.10496

    def __init__(self, openai_client, model_name, temperature, top_p, prefix="検索文章: "):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.prefix = prefix  # embeddings(ruri)都合で変更する

    def transform(self, query=None):
        messages_for_api = [
            {
                "role": "user",
                "content": "あなたは知識検索の専門家です。\nあなたのタスクは、与えられた元の検索クエリに対して、"
                           "該当する文書内容を例示することです。文書内容の例示はデータベースの検索に利用されます。\n"
                           "以後、質問に対して文書内容の例示を応答してください。"
            },
            {
                "role": "assistant",
                "content": "わかりました。適切に文書内容の例示を応答します。"
            },
            {
                "role": "user",
                "content": "茶道を体験したいです。京都で初心者が楽しめる場所はありますか？"
            },
            {
                "role": "assistant",
                "content": "京都で初めての茶道体験。初心者でも安心して参加できる、英語対応可能な教室を紹介します。"
                           "抹茶の点て方から和菓子の頂き方まで、基本的な作法を丁寧に学べます。"
                           "美しい庭園を眺めながら、静かなひとときをお過ごしください。"
            },
            {
                "role": "user",
                "content": "折り紙 鶴 簡単な折り方 子供向け"
            },
            {
                "role": "assistant",
                "content": "簡単に折れる『折り鶴』の折り方をステップバイステップで解説。分かりやすいイラストと写真付き。"
                           "平和の象徴でもある鶴を親子で一緒に作ってみましょう。必要な道具は折り紙一枚だけです。"
            },
            {
                "role": "user",
                "content": "リアルタイムOSでタスクをスケジューリングするアルゴリズムが知りたいです。"
            },
            {
                "role": "assistant",
                "content": "組み込みシステム向けリアルタイムOS (RTOS) におけるタスクスケジューリングアルゴリズムを比較。"
                           "優先度ベースプリエンプティブ方式、ラウンドロビン方式、Rate Monotonic (RM)、Earliest Deadline First (EDF) の特徴、"
                           "応答性、スケジューラビリティ解析について解説。"
            },
            {
                "role": "user",
                "content": query
            },
        ]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=len(query) * 5,  # FIXME 出力長
                stream=False
            )
            return self.prefix + response.choices[0].message.content
        except:
            # 例外時はそのまま返す
            return query


class StepBackQuery(QueryPreprocessor):
    # Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
    # https://arxiv.org/abs/2310.06117

    def __init__(self, openai_client, model_name, temperature, top_p):
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

    def transform(self, query=None):
        messages_for_api = [
            {
                "role": "user",
                "content": "あなたは知識検索の専門家です。\nあなたのタスクは、与えられた元の検索クエリを一歩下がって、"
                           "より一般的で、より高レベルで、回答しやすい「ステップバック質問」に言い換えることです。\n"
                           "ステップバック質問は、元のクエリに直接答えるために必要な全体的なコンテキストや基本的な情報、"
                           "原則を取得するのに役立ちます。\n以後質問に対して、「ステップバック質問」を応答してください。"
            },
            {
                "role": "assistant",
                "content": "わかりました。適切に「ステップバック質問」に言い換え応答します。"
            },
            {
                "role": "user",
                "content": "明治時代に活躍した夏目漱石は、1905年から1907年の間にどのような作品を発表しましたか？"
            },
            {
                "role": "assistant",
                "content": "夏目漱石の主要な著作は何ですか？"
            },
            {
                "role": "user",
                "content": "2023年のWBCで優勝した国の決勝戦の対戦相手はどこですか？"
            },
            {
                "role": "assistant",
                "content": "2023年のWBCで優勝した国はどこですか？"
            },
            {
                "role": "user",
                "content": "京都にある清水寺が現在の形になったのは西暦何年ですか？"
            },
            {
                "role": "assistant",
                "content": "京都の清水寺の歴史を教えてください。"
            },
            {
                "role": "user",
                "content": query
            },
        ]
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=int(len(query) * 1.5),  # FIXME 出力長
                stream=False
            )
            return response.choices[0].message.content
        except:
            import traceback
            traceback.print_exc()
            # 例外時はそのまま返す
            return query
