import streamlit as st
from tiny_chat.database.database_config import DEFAULT_CONFIG_PATH

def show_settings(
    logger=None,
    config_file_path=DEFAULT_CONFIG_PATH,
):
    url = st.text_input("データベースURL", value=st.session_state.db_config.server_url)
    api_key = st.text_input("データベース接続API-Key（必要なデータベースの場合）", value=st.session_state.db_config.api_key)
    file_path = st.text_input("データベースファイルパス（URLと排他利用）", value=st.session_state.db_config.file_path)
    collection_name = st.text_input("デフォルトコレクション名", value=st.session_state.db_config.selected_collection_name)

    top_k_input = st.text_input("LLMに与える検索結果数", value=st.session_state.db_config.top_k)
    try:
        top_k = int(top_k_input)
        if top_k < 0:
            st.error("検索結果数は0以上の整数を入力してください")
            top_k = st.session_state.db_config.top_k  # 不正な値の場合は元の値を維持
    except ValueError:
        st.error("検索結果数は整数を入力してください")
        top_k = st.session_state.db_config.top_k  # 数値に変換できない場合は元の値を維持

    score_threshold_input = st.text_input("検索結果として採用する類似度スコア下限",
                                          value=st.session_state.db_config.score_threshold)
    try:
        score_threshold = float(score_threshold_input)
        if score_threshold < 0.0 or score_threshold > 1.0:
            st.error("類似度スコア下限は0.0以上1.0以下の値を入力してください")
            score_threshold = st.session_state.db_config.score_threshold  # 不正な値の場合は元の値を維持
    except ValueError:
        st.error("類似度スコア下限は数値を入力してください")
        score_threshold = st.session_state.db_config.score_threshold  # 数値に変換できない場合は元の値を維持

    if st.button("設定変更・保存（URL、API-Key、ファイルパスを変更する場合は再接続します"):
        st.session_state.db_config.server_url = url
        st.session_state.db_config.api_key = api_key
        st.session_state.db_config.file_path = file_path
        st.session_state.db_config.selected_collection_name = collection_name
        st.session_state.db_config.top_k = top_k
        st.session_state.db_config.score_threshold = score_threshold
        from tiny_chat.database.database import get_or_create_qdrant_manager
        get_or_create_qdrant_manager(
            logger=logger, config_file_path=config_file_path, reconnect=True)
        st.rerun()

