
class TextChunker:
    """テキストをチャンクに分割するクラス。
    
    静的メソッドを提供し、チャンクサイズと重複部分のサイズを引数として指定します。
    """
    # セパレータリスト（クラス変数として保持）
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

    @staticmethod
    def split_texts(texts: list[str], chunk_size: int = 1024, chunk_overlap: int = 24) -> list[str]:
        """テキストをチャンクに分割します。

        Args:
            texts (list[str]): 分割する入力テキストのリスト
            chunk_size (int, optional): チャンクサイズ。デフォルトは768。
            chunk_overlap (int, optional): 連続するチャンク間の重複部分のサイズ。デフォルトは12。

        Returns:
            list[str]: テキストチャンクのリスト
        """
        chunk_texts = []

        for text in texts:
            chunks = TextChunker.split_text(
                text, separators=TextChunker.separators,
                hunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunk_texts.extend(chunks)

        return chunk_texts

    @staticmethod
    def split_text(text: str, separators: list[str] = None,
                   chunk_size: int = 1024, chunk_overlap: int = 24) -> list[str]:
        """テキストを指定されたセパレータとチャンクサイズを使って分割します。

        Args:
            text (str): 分割するテキスト
            separators (list[str], optional): 分割に使用するセパレータのリスト。
                指定されない場合はデフォルトのセパレータを使用します。
            chunk_size (int, optional): チャンクサイズ。デフォルトは768。
            chunk_overlap (int, optional): 連続するチャンク間の重複部分のサイズ。デフォルトは12。

        Returns:
            list[str]: 分割されたチャンクのリスト
        """
        chunks = []
        # 最初のセパレータで試みる
        separator_idx = 0

        # テキストが十分に短い場合はそのまま返す
        if len(text) <= chunk_size:
            return [text]

        if separators is None:
            separators = TextChunker.separators

        while separator_idx < len(separators):
            separator = separators[separator_idx]

            # 現在のセパレータでテキストを分割
            segments = text.split(separator) if separator else list(text)

            # 分割後のセグメントをチャンクに結合
            current_chunk = ""
            for i, segment in enumerate(segments):
                # 現在のセグメントを追加すると chunk_size を超える場合
                if len(current_chunk) + len(segment) + len(separator) > chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    # 重複を考慮して次のチャンクの開始位置を設定
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
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
            for i in range(0, len(text), chunk_size - chunk_overlap):
                end = min(i + chunk_size, len(text))
                chunks.append(text[i:end])

                # 最後まで処理したら終了
                if end == len(text):
                    break

        return chunks
