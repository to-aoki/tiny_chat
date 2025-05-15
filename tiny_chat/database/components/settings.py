import streamlit as st
from tiny_chat.database.database_config import DEFAULT_CONFIG_PATH


def show_settings(
        logger=None,
        config_file_path=DEFAULT_CONFIG_PATH,
):
    with st.form(key="settings_form"):
        st.subheader("データベース接続設定")
        url = st.text_input("データベースURL", value=st.session_state.db_config.server_url)
        api_key = st.text_input(
            "データベース接続API-Key（必要なデータベースの場合）",
            value=st.session_state.db_config.api_key,
            type="password"  # APIキーなのでパスワード型にする
        )
        file_path = st.text_input(
            "データベースファイルパス（URLと排他利用）",
            value=st.session_state.db_config.file_path
        )

        st.subheader("検索設定")
        collection_name = st.text_input(
            "デフォルトコレクション名",
            value=st.session_state.db_config.selected_collection_name
        )
        top_k = st.number_input(
            "LLMに与える検索結果数",
            min_value=0,
            value=int(st.session_state.db_config.top_k),
            step=1,
            format="%d"
        )
        score_threshold = st.number_input(
            "検索結果として採用する類似度スコア下限",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.db_config.score_threshold),
            step=0.01,
            format="%.2f"
        )

        submitted = st.form_submit_button("設定を保存して再接続")

    if submitted:
        config_changed = False
        connection_params_changed = False

        # 接続パラメータの変更チェック
        if st.session_state.db_config.server_url != url:
            st.session_state.db_config.server_url = url
            config_changed = True
            connection_params_changed = True
        if st.session_state.db_config.api_key != api_key:
            st.session_state.db_config.api_key = api_key
            config_changed = True
            connection_params_changed = True
        if st.session_state.db_config.file_path != file_path:
            st.session_state.db_config.file_path = file_path
            config_changed = True
            connection_params_changed = True

        if st.session_state.db_config.selected_collection_name != collection_name:
            st.session_state.db_config.selected_collection_name = collection_name
            config_changed = True
        if int(st.session_state.db_config.top_k) != top_k:  # 比較時も型を合わせる
            st.session_state.db_config.top_k = top_k
            config_changed = True
        if float(st.session_state.db_config.score_threshold) != score_threshold:  # 比較時も型を合わせる
            st.session_state.db_config.score_threshold = score_threshold
            config_changed = True

        if config_changed:
            try:
                # 設定ファイルに保存 (DatabaseConfigクラスにsaveメソッドがあると仮定)
                st.session_state.db_config.save(config_file_path)
                st.success("設定を保存しました。")

                if connection_params_changed:
                    st.info("接続パラメータが変更されたため、データベースに再接続します...")
                    from tiny_chat.database.database import get_or_create_qdrant_manager
                    get_or_create_qdrant_manager(
                        logger=logger, config_file_path=config_file_path, reconnect=True)

                    if logger:
                        logger.info("データベース設定が変更されたため、再接続が要求されました。")
                else:
                    if logger:
                        logger.info("データベース設定が変更されました (再接続は不要)。")

                st.rerun()

            except Exception as e:
                st.error(f"設定の保存または再接続中にエラーが発生しました: {e}")
                if logger:
                    logger.error(f"設定保存/再接続エラー: {e}")
        else:
            st.info("設定に変更はありませんでした。")

