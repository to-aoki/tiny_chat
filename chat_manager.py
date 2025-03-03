import json

class ChatManager:
    """
    LLMとのチャットを管理するクラス
    """
    def __init__(self):
        self.messages = []  # 表示用メッセージ履歴
        self.full_messages = []  # LLMへ送信する完全なメッセージ履歴
        self.attachments = []  # 添付ファイルリスト

    def add_user_message(self, content):
        """
        ユーザーメッセージを追加

        Args:
            content (str): メッセージの内容

        Returns:
            dict: 追加されたメッセージ
        """
        # 添付ファイル情報を追加
        attachments_info = ""
        if self.attachments:
            attachments_display = []
            for attachment in self.attachments:
                page_info = f"{attachment['num_pages']}ページ" if attachment['num_pages'] > 0 else ""
                attachments_display.append(f"{attachment['filename']} ({page_info})")
            attachments_info = "\n\n[添付ファイル: " + ", ".join(attachments_display) + "]"

        display_content = content + attachments_info

        # 表示用メッセージに追加
        message = {"role": "user", "content": display_content}
        self.messages.append(message)

        # 完全なメッセージ履歴に一時的なプレースホルダーを追加
        # IDを含めることでメッセージの対応関係を明確にする
        message_id = len(self.messages) - 1  # メッセージのインデックスをIDとして使用
        placeholder = {"role": "user", "content": f"placeholder_for_enhanced_prompt_{message_id}",
                       "message_id": message_id}
        self.full_messages.append(placeholder)

        return message
    def add_assistant_message(self, content):
        """
        アシスタントメッセージを追加

        Args:
            content (str): メッセージの内容

        Returns:
            dict: 追加されたメッセージ
        """
        message = {"role": "assistant", "content": content}
        self.messages.append(message)
        self.full_messages.append(message)
        return message

    def add_system_message(self, content):
        """
        システムメッセージを追加

        Args:
            content (str): メッセージの内容

        Returns:
            dict: 追加されたメッセージ
        """
        message = {"role": "system", "content": content}
        self.full_messages.append(message)
        # システムメッセージは通常表示用履歴には追加しない
        return message

    def get_latest_user_message(self):
        """
        最新のユーザーメッセージを取得

        Returns:
            dict or None: 最新のユーザーメッセージ（ない場合はNone）
        """
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message
        return None

    def get_enhanced_prompt(self, user_content, max_length=4000, uri_processor=None):
        """
        添付ファイルとURLの内容を含む拡張プロンプトを生成

        Args:
            user_content (str): 元のユーザーメッセージ内容
            max_length (int): 添付コンテンツの最大長
            uri_processor: URIを処理するためのプロセッサ

        Returns:
            str: 拡張されたプロンプト
        """
        # URLリファレンスを削除（添付ファイル記述があれば削除）
        clean_content = user_content.split("\n\n[添付ファイル:")[0]
        enhanced_prompt = clean_content

        # 添付ファイルの内容を追加
        for idx, attachment in enumerate(self.attachments):
            if len(attachment["content"]) > max_length:
                content_truncated = attachment["content"][:max_length] + f"\n...(テキストが長いため{max_length}文字で省略されました)"
            else:
                content_truncated = attachment["content"]

            enhanced_prompt += f"\n\n添付ファイル{idx + 1} '{attachment['filename']}' の内容:\n\n{content_truncated}"

        # URLコンテンツがあれば追加
        if uri_processor:
            uri_contents = self._process_uris_with_processor(clean_content, max_length, uri_processor)
        else:
            uri_contents = []

        if uri_contents:
            enhanced_prompt += "\n\n以下はURLから取得したコンテンツです:\n\n" + "\n\n---\n\n".join(uri_contents)

        return enhanced_prompt

    def _process_uris_with_processor(self, text, max_length, uri_processor, limit=1):
        """
        テキスト内のURIを処理してコンテンツを取得

        Args:
            text (str): 処理するテキスト
            max_length (int): 最大コンテンツ長
            uri_processor: URIプロセッサオブジェクト

        Returns:
            list: 取得したコンテンツリスト
        """
        # URIを検出
        detected_uris = uri_processor.detect_uri(text)
        detected_uris = detected_uris[:limit]
        uri_contents = []

        for uri in detected_uris:
            # 外部プロセッサを使用してURI処理
            content, message = uri_processor.process_uri(uri)
            if content:
                if len(content) > max_length:
                    content_truncated = content[:max_length] + f"...(テキストが長いため{max_length}文字で省略されました)"
                else:
                    content_truncated = content
                uri_contents.append(f"URL: {uri}\n\n{content_truncated}")

        return uri_contents

    def update_enhanced_prompt(self, enhanced_prompt):
        """
        最後のユーザーメッセージのプレースホルダーを
        拡張プロンプトで置き換える

        Args:
            enhanced_prompt (str): 拡張されたプロンプト内容
        """
        # 最後のメッセージのIDを取得
        latest_message_id = len(self.messages) - 1

        # 対応するプレースホルダーを探して更新
        for i in range(len(self.full_messages) - 1, -1, -1):
            msg = self.full_messages[i]
            if msg["role"] == "user" and "message_id" in msg and msg["message_id"] == latest_message_id:
                self.full_messages[i] = {"role": "user", "content": enhanced_prompt, "message_id": latest_message_id}
                break

    def prepare_messages_for_api(self, meta_prompt=""):
        """
        API呼び出し用のメッセージ配列を準備

        Args:
            meta_prompt (str): システムメッセージとして追加するメタプロンプト

        Returns:
            list: API用に準備されたメッセージリスト
        """
        messages_for_api = []

        # メタプロンプトがあれば最初に追加
        if meta_prompt.strip():
            messages_for_api.append({"role": "system", "content": meta_prompt})

        # 履歴を確認して、ユーザーメッセージとアシスタントメッセージが交互に来るようにする
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user":
                # ユーザーメッセージの場合は、拡張プロンプトをfull_messagesから探す
                for full_msg in self.full_messages:
                    if full_msg["role"] == "user" and full_msg["content"] != "placeholder_for_enhanced_prompt":
                        user_content_base = msg["content"].split("\n\n[添付ファイル:")[0].strip()
                        if user_content_base in full_msg["content"]:
                            messages_for_api.append({"role": "user", "content": full_msg["content"]})
                            break
                else:
                    # 見つからなければ、表示用のメッセージをそのまま使用
                    messages_for_api.append({"role": msg["role"], "content": msg["content"]})
            else:
                # アシスタントメッセージはそのまま追加
                messages_for_api.append({"role": msg["role"], "content": msg["content"]})

        return messages_for_api

    def clear_attachments(self):
        """添付ファイルをクリア"""
        self.attachments = []

    def add_attachment(self, filename, content, num_pages):
        """
        添付ファイルを追加

        Args:
            filename (str): ファイル名
            content (str): ファイルの内容（テキスト）
            num_pages (int): ページ数

        Returns:
            dict: 追加された添付ファイル情報
        """
        attachment = {
            "filename": filename,
            "content": content,
            "num_pages": num_pages
        }
        self.attachments.append(attachment)
        return attachment

    def to_json(self, include_system=False):
        """
        メッセージ履歴をJSON形式で出力

        Args:
            include_system (bool): システムメッセージを含めるかどうか

        Returns:
            str: JSON形式のメッセージ履歴
        """
        import json

        filtered_messages = []

        for msg in self.messages:
            if not include_system and msg["role"] == "system":
                continue

            # プレースホルダーチェック（ユーザーメッセージのみ）
            if msg["role"] == "user" and "placeholder_for_enhanced_prompt" in msg["content"]:
                # プレースホルダーが含まれる場合はスキップ
                continue

            # プレースホルダーがなければそのまま追加
            filtered_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return json.dumps(filtered_messages, ensure_ascii=False, indent=2)

    def apply_imported_history(self, json_str):
        """
        JSONからインポートした履歴を適用し、プレースホルダーなどを適切に処理する

        Args:
            json_str (str): JSON形式のメッセージ履歴

        Returns:
            bool: 適用の成功/失敗
        """
        import json

        try:
            # 現在の状態を一時保存（エラー時の復元用）
            current_messages = self.messages.copy()
            current_full_messages = self.full_messages.copy()

            # 一旦メッセージをクリア
            self.messages = []
            self.full_messages = []

            # JSONデータをパース
            messages = json.loads(json_str)

            # メッセージを1つずつ追加
            message_id = 0
            for msg in messages:
                if msg["role"] == "system":
                    self.add_system_message(msg["content"])
                elif msg["role"] == "user":
                    # プレースホルダーテキストをチェック
                    if "placeholder_for_enhanced_prompt" in msg["content"]:
                        # プレースホルダーを含む場合は、有効なコンテンツのみ抽出
                        clean_content = ""
                        if "[添付ファイル:" in msg["content"]:
                            # 添付ファイル情報があれば保持
                            attachment_part = msg["content"].split("[添付ファイル:")[1]
                            clean_content = "[添付ファイル:" + attachment_part
                    else:
                        # プレースホルダーがなければそのまま使用
                        clean_content = msg["content"]

                    # 処理済みコンテンツをメッセージに追加
                    self.messages.append({"role": "user", "content": clean_content})
                    self.full_messages.append({"role": "user", "content": clean_content, "message_id": message_id})
                    message_id += 1
                else:
                    self.add_assistant_message(msg["content"])

            return True

        except Exception as e:
            # エラー時は元の状態に戻す
            self.messages = current_messages
            self.full_messages = current_full_messages
            return False

    def load_from_json(self, json_str):
        """
        JSON文字列からメッセージ履歴を読み込み

        Args:
            json_str (str): JSON形式のメッセージ履歴

        Returns:
            bool: 読み込みの成功/失敗
        """
        import json

        try:
            messages = json.loads(json_str)

            # 基本的な検証
            if not isinstance(messages, list):
                return False

            for msg in messages:
                if not isinstance(msg, dict):
                    return False
                if "role" not in msg or "content" not in msg:
                    return False
                if msg["role"] not in ["user", "assistant", "system"]:
                    return False

            # 検証通過後にメッセージをクリア
            self.messages = []
            self.full_messages = []

            message_id = 0
            for msg in messages:
                if msg["role"] == "system":
                    self.add_system_message(msg["content"])
                elif msg["role"] == "user":
                    # プレースホルダーテキストを検出して置き換える
                    content = msg["content"]
                    if "placeholder_for_enhanced_prompt" in content:
                        # プレースホルダーが含まれている場合は、表示用のコンテンツを作成
                        # 添付ファイル情報だけを残す
                        if "[添付ファイル:" in content:
                            parts = content.split("[添付ファイル:")
                            # 元のメッセージ部分を空にして、添付ファイル情報だけ保持
                            content = "[添付ファイル:" + parts[1]
                        else:
                            # プレースホルダーを含むが添付ファイル情報がない場合、空のメッセージにする
                            content = ""

                    # ユーザーメッセージを直接追加
                    self.messages.append({"role": "user", "content": content})
                    # full_messagesにも同じ内容を追加 (JSONから復元する場合はプレースホルダーは不要)
                    self.full_messages.append({"role": "user", "content": content, "message_id": message_id})
                    message_id += 1
                else:
                    self.add_assistant_message(msg["content"])

            return True

        except json.JSONDecodeError:
            return False
        except Exception as e:
            print(f"JSON load error: {str(e)}")
            return False

    def check_total_message_length(self, meta_prompt=""):
        """
        すべてのメッセージ（システムプロンプト含む）の合計長をチェック

        Args:
            meta_prompt (str): メタプロンプト（システムメッセージとして含める）

        Returns:
            int: 合計文字数
        """
        total_length = 0

        # メタプロンプトがあれば加算
        if meta_prompt.strip():
            total_length += len(meta_prompt)

        # 全メッセージの長さを加算
        for msg in self.full_messages:
            total_length += len(msg["content"])

        return total_length

    def estimate_enhanced_prompt_length(self, user_content, max_length=4000, uri_processor=None):
        """
        拡張プロンプトの長さを推定（実際に生成せず）

        Args:
            user_content (str): 元のユーザーメッセージ内容
            max_length (int): 添付コンテンツの最大長
            uri_processor: URIを処理するためのプロセッサ

        Returns:
            int: 推定される拡張プロンプトの長さ
        """
        # ベースとなるユーザー入力
        estimated_length = len(user_content)

        # 添付ファイルの内容を考慮
        for attachment in self.attachments:
            attachment_content_length = min(len(attachment["content"]), max_length)
            # ヘッダーなども考慮
            attachment_header = f"\n\n添付ファイル '{attachment['filename']}' の内容:\n\n"
            estimated_length += len(attachment_header) + attachment_content_length

            # 長すぎる場合の省略メッセージも考慮
            if len(attachment["content"]) > max_length:
                truncate_msg = f"\n...(テキストが長いため{max_length}文字で省略されました)"
                estimated_length += len(truncate_msg)

        # URLの内容を考慮（URIプロセッサが提供されている場合）
        if uri_processor:
            estimated_length += 4000

        return estimated_length

    def would_exceed_message_length(self, user_content, max_message_length, max_context_length=4000, meta_prompt="",
                                    uri_processor=None):
        """
        新しいメッセージを追加した場合に、最大メッセージ長を超えるかどうか判定

        Args:
            user_content (str): 新しいユーザーメッセージ
            max_message_length (int): 最大許容メッセージ長
            max_context_length (int): コンテキスト長（添付ファイル/URLコンテンツの最大長）
            meta_prompt (str): メタプロンプト
            uri_processor: URIを処理するためのプロセッサ

        Returns:
            tuple: (超過するかどうか, 推定される合計長, 最大長)
        """
        # 現在のメッセージ長
        current_length = self.check_total_message_length(meta_prompt)

        # 新しいメッセージの推定長
        new_message_length = self.estimate_enhanced_prompt_length(
            user_content,
            max_length=max_context_length,
            uri_processor=uri_processor
        )

        # 合計の推定長
        total_estimated_length = current_length + new_message_length

        # 最大長と比較
        would_exceed = total_estimated_length > max_message_length

        return would_exceed, total_estimated_length, max_message_length