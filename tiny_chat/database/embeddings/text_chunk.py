
class TextChunker:
    """テキストをチャンクに分割するクラス。

    Attributes:
        chunk_size (int): チャンクサイズ
        chunk_overlap (int): 連続するチャンク間の重複部分のサイズ
    """
    separators = [
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        "",
    ]

    def __init__(self, chunk_size: int = 768, chunk_overlap: int = 12):
        """TextChunkerをチャンクサイズと重複部分のサイズで初期化します。

        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): 連続するチャンク間の重複部分のサイズ
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_texts(self, texts: list[str]) -> list[str]:
        """テキストをチャンクに分割します。

        Args:
            texts (List[str]): 分割する入力テキストのリスト

        Returns:
            List[str]: テキストチャンクのリスト
        """
        chunk_texts = []

        for text in texts:
            chunks = self.split_text(text, self.separators)
            chunk_texts.extend(chunks)

        return chunk_texts

    def split_text(self, text: str, separators: list[str] = None) -> list[str]:
        """テキストを指定されたセパレータとチャンクサイズを使って分割します。

        Args:
            text (str): 分割するテキスト
            separators (List[str], optional): 分割に使用するセパレータのリスト

        Returns:
            List[str]: 分割されたチャンクのリスト
        """
        chunks = []
        # 最初のセパレータで試みる
        separator_idx = 0

        # テキストが十分に短い場合はそのまま返す
        if len(text) <= self.chunk_size:
            return [text]

        if separators is None:
            separators = self.separators

        while separator_idx < len(separators):
            separator = separators[separator_idx]

            # 現在のセパレータでテキストを分割
            segments = text.split(separator) if separator else list(text)

            # 分割後のセグメントをチャンクに結合
            current_chunk = ""
            for i, segment in enumerate(segments):
                # 現在のセグメントを追加すると chunk_size を超える場合
                if len(current_chunk) + len(segment) + len(separator) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    # 重複を考慮して次のチャンクの開始位置を設定
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + (separator if separator else "") + segment
                else:
                    if i > 0 and separator:  # 最初のセグメント以外でセパレータを追加
                        current_chunk += separator
                    current_chunk += segment

            # 最後のチャンクを追加
            if current_chunk:
                chunks.append(current_chunk)

            # チャンクができている場合は終了、そうでなければ次のセパレータを試す
            if chunks:
                return chunks
            separator_idx += 1

        # どのセパレータでもうまく分割できなかった場合、文字単位で強制的に分割
        if not chunks:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                end = min(i + self.chunk_size, len(text))
                chunks.append(text[i:end])

                # 最後まで処理したら終了
                if end == len(text):
                    break

        return chunks
