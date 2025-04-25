import os
import threading

import streamlit as st

from tiny_chat.database.database_config import DatabaseConfig, DEFAULT_CONFIG_PATH
from tiny_chat.database.components.search import show_search_component
from tiny_chat.database.components.registration import show_registration
from tiny_chat.database.components.manage import show_manage_component
from tiny_chat.database.components.settings import show_settings


SUPPORT_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']

# プロセスレベルでQdrantManagerインスタンスを保持するためのグローバル変数
_qdrant_manager = None


@st.cache_resource
def get_lock():
    # インスタンス生成のロックに使用
    return threading.Lock()


my_lock = get_lock()


def get_or_create_qdrant_manager(logger=None, config_file_path=DEFAULT_CONFIG_PATH, reconnect=False):
    """
    QdrantManagerを取得または初期化する共通関数
    プロセスレベルで一つのインスタンスを共有するよう修正
    スレッドセーフな実装を使用

    Args:
        logger: ロガーオブジェクト（オプション）

    Returns:
        QdrantManager: 初期化されたQdrantManagerオブジェクト
    """
    if "db_config" not in st.session_state:
        # 外部設定ファイルから設定を読み込む
        db_config = DatabaseConfig.load(config_file_path)
        logger.info(f"DB設定ファイルを読み込みました: {config_file_path}")
        # セッション状態に設定オブジェクトを初期化
        st.session_state.db_config = db_config
        logger.info("設定オブジェクトをセッション状態に初期化しました")

    global _qdrant_manager

    with my_lock:
        # プロセスレベルでQdrantManagerがまだ初期化されていない場合は初期化
        if _qdrant_manager is None:
            with st.spinner("データベースを初期化中..."):
                from tiny_chat.database.qdrant.qdrant_manager import QdrantManager

                if logger:
                    logger.info("QdrantManagerを初期化しています...")
                try:
                    db_config = st.session_state.db_config
                    _qdrant_manager = QdrantManager(
                        **db_config.__dict__
                    )
                    from tiny_chat.database.qdrant.collection import Collection
                    collection = Collection(**db_config.__dict__)
                    collection.save(qdrant_manager=_qdrant_manager)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if logger:
                        logger.error(f"QdrantManagerの初期化が失敗しました: {str(e)}")
                    raise e
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
                        from tiny_chat.database.qdrant.collection import Collection
                        _qdrant_manager = QdrantManager(
                            **db_config.__dict__
                        )
                        collection = Collection(**db_config.__dict__)
                        collection.save(qdrant_manager=_qdrant_manager)

                        if logger:
                            logger.info("QdrantManagerの再初期化が完了しました")
                    except Exception as e:
                        if logger:
                            logger.error(f"QdrantManagerの再初期化が失敗しました: {str(e)}")
                        raise e
            else:
                try:
                    _qdrant_manager.set_collection_name(collection_name=db_config.selected_collection_name)
                except Exception as e:
                    if logger:
                        logger.error(f"QdrantManagerの情報更新に失敗しました: {str(e)}")
                    raise e
    if reconnect:
        try:
            db_config.save(DEFAULT_CONFIG_PATH)
        except Exception as e:
            if logger:
                logger.error(f"DB設定情報の保存に失敗しました: {str(e)}")
            raise e

    return _qdrant_manager


@st.fragment
def show_database_component(logger, extensions=SUPPORT_EXTENSIONS):

    mode = ["🔍 検索"]
    is_server_mode = True
    if st.session_state.get("config") is None or (st.session_state.get("config") is not None and st.session_state.get(
            "config").get("session_only_mode") is not True):
        mode.append("📑 登録")
        mode.append("📚 管理")
        mode.append("⚙️ 設定")
        is_server_mode = False

    if 'active_select_db' not in st.session_state:
        st.session_state.active_select_db = mode[0]

    # 検索と文書登録のタブを作成
    st.selectbox(
        "データベースナビゲーション",
        mode,
        key='active_select_db',
        label_visibility="collapsed",
    )

    # QdrantManagerを使用
    _qdrant_manager = get_or_create_qdrant_manager(logger)

    # 検索タブ
    if st.session_state.active_select_db == mode[0]:
        show_search_component(_qdrant_manager)
    if not is_server_mode:
        # 文書登録タブ
        if st.session_state.active_select_db == mode[1]:
            show_registration(_qdrant_manager, extensions=extensions)

        # 削除タブ
        if st.session_state.active_select_db == mode[2]:
            show_manage_component(_qdrant_manager, logger=logger)

        # 設定タブ
        if st.session_state.active_select_db == mode[3]:
            show_settings(logger=logger, config_file_path=DEFAULT_CONFIG_PATH)


# 単独動作用
def run_database_app():
    import logging
    from tiny_chat.utils.logger import get_logger
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # https://discuss.streamlit.io/t/message-error-about-torch/90886/9
    # RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
    import torch
    torch.classes.__path__ = []

    # ロガーの初期化
    LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
    LOGGER.info("単独データベースアプリケーションを起動しました")

    # 単独で起動した場合はQdrantManagerを初期化
    get_or_create_qdrant_manager(LOGGER)
    
    # コンポーネントの表示
    show_database_component(logger=LOGGER, extensions=SUPPORT_EXTENSIONS)
