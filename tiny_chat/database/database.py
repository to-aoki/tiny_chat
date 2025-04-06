import os

import streamlit as st

from tiny_chat.database.database_config import DatabaseConfig
from tiny_chat.database.components.search import show_search_component
from tiny_chat.database.components.registration import show_registration
from tiny_chat.database.components.deletion import show_delete_component
from tiny_chat.database.components.settings import show_settings


SUPPORT_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']

# プロセスレベルでQdrantManagerインスタンスを保持するためのグローバル変数
_qdrant_manager = None

# インスタンス生成のロックに使用
_qdrant_lock = None

# 設定ファイルのパス
CONFIG_FILE = "database_config.json"


def get_or_create_qdrant_manager(
        logger=None, config_file_path=CONFIG_FILE, reconnect=False):
    """
    QdrantManagerを取得または初期化する共通関数
    プロセスレベルで一つのインスタンスを共有するよう修正
    スレッドセーフな実装を使用

    Args:
        logger: ロガーオブジェクト（オプション）

    Returns:
        QdrantManager: 初期化されたQdrantManagerオブジェクト
    """
    global _qdrant_manager, _qdrant_lock

    # ロックオブジェクトがなければ作成
    if _qdrant_lock is None:
        import threading
        _qdrant_lock = threading.Lock()

    # ロックを取得して排他制御
    with _qdrant_lock:
        # プロセスレベルでQdrantManagerがまだ初期化されていない場合は初期化
        if _qdrant_manager is None:
            with st.spinner("データベースを初期化中..."):
                if "db_config" not in st.session_state:
                    # 外部設定ファイルから設定を読み込む
                    db_config = DatabaseConfig.load(config_file_path)
                    logger.info(f"DB設定ファイルを読み込みました: {config_file_path}")
                    # セッション状態に設定オブジェクトを初期化
                    st.session_state.db_config = db_config
                    logger.info("設定オブジェクトをセッション状態に初期化しました")
                    from tiny_chat.database.qdrant.qdrant_manager import QdrantManager

                if logger:
                    logger.info("QdrantManagerを初期化しています...")
                _qdrant_manager = QdrantManager(
                    **db_config.__dict__
                )
                if logger:
                    logger.info("QdrantManagerの初期化が完了しました")

        elif reconnect:
            db_config = st.session_state.db_config
            if _qdrant_manager.is_need_reconnect(**db_config.__dict__):
                with st.spinner("データベースを再接続中..."):
                    try:
                        if logger:
                            logger.info("QdrantManagerを再初期化しています...")
                        from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
                        _qdrant_manager = QdrantManager(
                            **db_config.__dict__
                        )
                        if logger:
                            logger.info("QdrantManagerの再初期化が完了しました")
                    except Exception as e:
                        if logger:
                            logger.error(f"QdrantManagerの再初期化が失敗しました: {str(e)}")
                        raise e
            else:
                try:
                    _qdrant_manager.set_collection_name(collection_name=db_config.collection_name)
                except Exception as e:
                    if logger:
                        logger.error(f"QdrantManagerの情報更新に失敗しました: {str(e)}")
                    raise e
            try:
                db_config.save(CONFIG_FILE)
            except Exception as e:
                if logger:
                    logger.error(f"DB設定情報の保存に失敗しました: {str(e)}")
                raise e

    return _qdrant_manager


@st.fragment
def show_database_component(logger, extensions=SUPPORT_EXTENSIONS):
    # 検索と文書登録のタブを作成
    search_tabs = st.tabs(["🔍 検索", "📁 登録", "🪣 管理", "⚙️ 設定"])

    # QdrantManagerを使用
    _qdrant_manager = get_or_create_qdrant_manager(logger)

    # 検索タブ
    with search_tabs[0]:
        show_search_component(_qdrant_manager)

    # 文書登録タブ
    with (search_tabs[1]):
        show_registration(_qdrant_manager, extensions=extensions)

    # 削除タブ
    with search_tabs[2]:
        show_delete_component(_qdrant_manager, logger=logger)
    # 設定タブ
    with search_tabs[3]:
        show_settings(logger=logger, config_file_path=CONFIG_FILE)


# 単独動作用LLMLL
def run_database_app():
    import logging
    from tiny_chat.utils.logger import get_logger
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # https://discuss.streamlit.io/t/message-error-about-torch/90886/9
    # RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
    import torch
    torch.classes.__path__ = []

    st.set_page_config(page_title="データベース", layout="wide")
    # ロガーの初期化
    LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
    LOGGER.info("単独データベースアプリケーションを起動しました")

    # 単独で起動した場合はQdrantManagerを初期化
    get_or_create_qdrant_manager(LOGGER)
    
    # コンポーネントの表示
    show_database_component(logger=LOGGER, extensions=SUPPORT_EXTENSIONS)
