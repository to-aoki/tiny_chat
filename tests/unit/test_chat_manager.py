import pytest
from tiny_chat.chat.chat_manager import ChatManager

class TestChatManager:
    def test_initialization(self):
        """初期化時の状態を確認"""
        manager = ChatManager()
        assert manager.messages == []
        assert manager.full_messages == []
        assert manager.attachments == []

    def test_add_user_message(self):
        """ユーザーメッセージの追加を確認"""
        manager = ChatManager()
        message = manager.add_user_message("Hello")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello"
        
        # full_messagesも更新されていることを確認
        assert len(manager.full_messages) == 1
        assert manager.full_messages[0]["role"] == "user"
        assert manager.full_messages[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        """アシスタントメッセージの追加を確認"""
        manager = ChatManager()
        message = manager.add_assistant_message("Hi there")
        
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
        assert manager.messages[0]["content"] == "Hi there"
        
        # full_messagesも更新されていることを確認
        assert len(manager.full_messages) == 1
        assert manager.full_messages[0]["role"] == "assistant"
        assert manager.full_messages[0]["content"] == "Hi there"

    def test_delete_message_pair(self):
        """メッセージ対の削除を確認"""
        manager = ChatManager()
        manager.add_user_message("User 1")
        manager.add_assistant_message("Assistant 1")
        manager.add_user_message("User 2")
        manager.add_assistant_message("Assistant 2")
        
        assert len(manager.messages) == 4
        
        # 最初のペア(インデックス0)を削除
        result = manager.delete_message_pair(0)
        
        assert result is True
        assert len(manager.messages) == 2
        assert manager.messages[0]["content"] == "User 2"
        assert manager.messages[1]["content"] == "Assistant 2"

    def test_delete_message_pair_invalid_index(self):
        """無効なインデックスでの削除失敗を確認"""
        manager = ChatManager()
        manager.add_user_message("User 1")
        
        # ペアになっていない、または範囲外
        assert manager.delete_message_pair(0) is False
        assert manager.delete_message_pair(10) is False

    def test_add_attachment(self):
        """添付ファイルの追加とプロンプトへの反映を確認"""
        manager = ChatManager()
        manager.add_attachment("test.txt", "File content", 1)
        
        assert len(manager.attachments) == 1
        
        # 添付ファイルがある状態でメッセージを追加
        manager.add_user_message("Check this file")
        
        # 表示用メッセージには添付ファイル情報が含まれる
        assert "[添付ファイル: test.txt (1ページ)]" in manager.messages[0]["content"]
        
        # full_messagesにはプレースホルダーが入る
        assert "placeholder_for_enhanced_prompt" in manager.full_messages[0]["content"]
        
        # 拡張プロンプトの更新を確認
        enhanced_prompt = manager.get_enhanced_prompt("Check this file")
        assert "添付ファイル1 'test.txt' の内容" in enhanced_prompt
        assert "File content" in enhanced_prompt

    def test_estimate_enhanced_prompt_length(self):
        """拡張プロンプト長の推定を確認"""
        manager = ChatManager()
        content = "Hello"
        
        # 添付なし
        length = manager.estimate_enhanced_prompt_length(content)
        assert length == len(content)
        
        # 添付あり
        manager.add_attachment("test.txt", "World", 1)
        length_with_attachment = manager.estimate_enhanced_prompt_length(content)
        
        assert length_with_attachment > len(content) + len("World")
