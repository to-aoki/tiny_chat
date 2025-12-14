import pytest
import os
import sys

# プロジェクトルートをパスに追加して、tiny_chatモジュールをインポートできるようにする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def mock_openai_client(mocker):
    """OpenAIクライアントをモックするフィクスチャ"""
    mock_client = mocker.Mock()
    return mock_client

@pytest.fixture
def mock_streamlit_session(mocker):
    """Streamlitのセッションステートをモックするフィクスチャ"""
    mock_session = mocker.patch("streamlit.session_state")
    return mock_session
