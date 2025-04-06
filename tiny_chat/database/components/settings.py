import streamlit as st

CONFIG_FILE = "database_config.json"


def show_settings(
    logger=None,
    config_file_path=CONFIG_FILE,
):
    url = st.text_input("データベースURL", value=st.session_state.db_config.server_url)
    api_key = st.text_input("データベース接続API-Key（必要なデータベースの場合）", value=st.session_state.db_config.api_key)
    file_path = st.text_input("データベースファイルパス（URLと排他利用）", value=st.session_state.db_config.file_path)
    collection_name = st.text_input("デフォルトコレクション名", value=st.session_state.db_config.selected_collection_name)

    if st.button("設定変更・保存（URL、API-Key、ファイルパスを変更する場合は再接続します"):
        st.session_state.db_config.server_url = url
        st.session_state.db_config.api_key = api_key
        st.session_state.db_config.file_path = file_path
        st.session_state.db_config.selected_collection_name = collection_name
        from tiny_chat.database.database import get_or_create_qdrant_manager
        manager = get_or_create_qdrant_manager(
            logger=logger, config_file_path=config_file_path, reconnect=True)

        st.rerun()

