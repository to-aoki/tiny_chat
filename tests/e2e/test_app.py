from streamlit.testing.v1 import AppTest
import os
import pytest

# アプリケーションのパス
APP_PATH = os.path.join(os.path.dirname(__file__), "../../tiny_chat/main.py")

from unittest.mock import patch

class TestApp:
    def test_app_starts(self):
        """アプリが正常に起動し、タイトルが表示されるか確認"""
        with patch("sys.argv", ["main.py"]), \
             patch("tiny_chat.chat.chat_config.ModelManager.fetch_available_models", return_value=([], True)), \
             patch("tiny_chat.utils.llm_utils.get_llm_client", return_value=None):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        
        assert not at.exception
        # 少なくとも1つの要素が表示されていることを確認
        assert len(at.main) > 0

    def test_sidebar_settings(self):
        """サイドバーの設定項目が表示されるか確認"""
        with patch("sys.argv", ["main.py"]), \
             patch("tiny_chat.chat.chat_config.ModelManager.fetch_available_models", return_value=([], True)), \
             patch("tiny_chat.utils.llm_utils.get_llm_client", return_value=None):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        
        # サイドバーに設定用の入力フィールドがあるか
        # 注意: 実際のサイドバーの構成に依存します。
        # ChatConfigのデフォルト値などが読み込まれているはずです。
        assert len(at.sidebar) > 0

    def test_chat_input_exists(self):
        """チャット入力欄が存在するか確認"""
        with patch("sys.argv", ["main.py"]), \
             patch("tiny_chat.chat.chat_config.ModelManager.fetch_available_models", return_value=([], True)), \
             patch("tiny_chat.utils.llm_utils.get_llm_client", return_value=None):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        
        # chat_inputは1つだけのはず
        assert len(at.chat_input) == 1
