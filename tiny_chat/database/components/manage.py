import streamlit as st
import pandas as pd


def _manage_sources(qdrant_manager, logger):
    # コレクション名の入力
    collection_name = st.text_input(
        "コレクション名",
        value=qdrant_manager.collection_name,
        help="操作対象のコレクション名を指定します。",
        key="data_management_collection"
    )

    # 使用可能なソースを取得（常に最新の状態を取得、コレクション名を明示的に指定）
    sources = qdrant_manager.get_sources(collection_name=collection_name)

    if not sources:
        st.warning("データベースにソースが見つかりません。先にファイルを登録してください。")
    else:
        # ソースの選択（固定キーを使用）
        selected_source = st.selectbox(
            "削除するソースを選択",
            options=sources,
            help="指定したソースを持つすべてのチャンクが削除されます。",
            key="source_select_delete"
        )

        # 選択したソースをセッション状態に保存（削除確認時に使用）
        if "selected_source_to_delete" not in st.session_state:
            st.session_state.selected_source_to_delete = None

        if selected_source:
            st.session_state.selected_source_to_delete = selected_source

        # 削除ボタン
        delete_cols = st.columns([3, 3, 3])
        with delete_cols[1]:
            delete_pressed = st.button(
                "削除実行",
                key="delete_source_button",
                type="primary",
                use_container_width=True
            )

        # 削除確認状態の管理
        if "delete_confirmation_state" not in st.session_state:
            st.session_state.delete_confirmation_state = False

        # 削除実行ボタンが押されたら確認状態をONに
        if delete_pressed and st.session_state.selected_source_to_delete:
            st.session_state.delete_confirmation_state = True

        # 確認状態がONの場合に確認ダイアログを表示
        if st.session_state.delete_confirmation_state and st.session_state.selected_source_to_delete:
            selected_source_to_delete = st.session_state.selected_source_to_delete

            # 確認ダイアログ
            st.warning(
                f"ソース '{selected_source_to_delete}' に関連するすべてのチャンクを削除します。この操作は元に戻せません。"
            )
            confirm_cols = st.columns([2, 2, 2])
            with confirm_cols[0]:
                cancel_confirmed = st.button(
                    "キャンセル",
                    key="cancel_delete_source_button",
                    use_container_width=True
                )

            with confirm_cols[2]:
                confirmed = st.button(
                    "削除を確定",
                    key="confirm_delete_source_button",
                    type="primary",
                    use_container_width=True
                )

            # キャンセルボタンが押された場合は確認状態をOFFに
            if cancel_confirmed:
                st.session_state.delete_confirmation_state = False
                st.rerun()

            if confirmed:
                with st.spinner(f"ソース '{selected_source_to_delete}' のチャンクを削除中..."):
                    try:
                        # コレクション名を指定して削除処理を実行
                        # ソースでフィルタリングして削除（ソース名が単一でも配列として渡す）
                        filter_params = {"source": [selected_source_to_delete]}

                        # delete_by_filterメソッドに明示的にコレクション名を渡す
                        qdrant_manager.delete_by_filter(filter_params, collection_name=collection_name)

                        # 常に成功メッセージを表示
                        st.success(f"ソース '{selected_source_to_delete}' の削除が完了しました")
                        # 確認状態をリセット
                        st.session_state.delete_confirmation_state = False
                        st.session_state.selected_source_to_delete = None
                        # 削除後に画面を更新して、ソースリストを最新化
                        st.rerun()
                    except Exception as e:
                        st.error(f"削除処理中にエラーが発生しました: {str(e)}")
                        logger.error(f"削除処理エラー: {str(e)}")


def _manage_collections(qdrant_manager, logger):
    # 利用可能なコレクション一覧を取得
    collections = qdrant_manager.get_collections()

    if not collections:
        st.warning("データベースにコレクションが見つかりません。")
    else:
        # コレクション情報を表示
        st.write(f"利用可能なコレクション: {len(collections)}個")

        # 各コレクションの情報を表示
        collection_infos = []
        for col_name in collections:
            try:
                # 現在のコレクション名を保持
                original_collection = qdrant_manager.collection_name

                # 文書数を取得（コレクション名を明示的に指定）
                doc_count = qdrant_manager.count_documents(collection_name=col_name)

                # コレクションに関する情報を収集
                collection_infos.append({
                    "name": col_name,
                    "doc_count": doc_count,
                    "is_current": col_name == original_collection
                })
            except Exception as e:
                logger.error(f"コレクション情報取得エラー ({col_name}): {str(e)}")
                collection_infos.append({
                    "name": col_name,
                    "doc_count": "エラー",
                    "is_current": col_name == qdrant_manager.collection_name
                })

        # 表形式で表示
        df_collections = pd.DataFrame(collection_infos)
        st.dataframe(
            df_collections,
            column_config={
                "name": "コレクション名",
                "doc_count": "文書数",
                "is_current": "現在使用中"
            },
            hide_index=True,
            use_container_width=True
        )

        # コレクションの選択（固定キーを使用）
        selected_collection = st.selectbox(
            "削除するコレクションを選択",
            options=collections,
            help="選択したコレクションを完全に削除します。この操作は元に戻せません。",
            key="collection_select_delete"
        )

        # 選択したコレクションをセッション状態に保存（削除確認時に使用）
        if "selected_collection_to_delete" not in st.session_state:
            st.session_state.selected_collection_to_delete = None

        if selected_collection:
            st.session_state.selected_collection_to_delete = selected_collection

        # 削除ボタン
        delete_cols = st.columns([3, 3, 3])
        with delete_cols[1]:
            delete_collection_pressed = st.button(
                "削除実行",
                key="delete_collection_button",
                type="primary",
                use_container_width=True
            )

        # 削除確認状態の管理
        if "delete_collection_confirmation_state" not in st.session_state:
            st.session_state.delete_collection_confirmation_state = False

        # 削除実行ボタンが押されたら確認状態をONに
        if delete_collection_pressed and st.session_state.selected_collection_to_delete:
            st.session_state.delete_collection_confirmation_state = True

        # 削除実行
        if st.session_state.delete_collection_confirmation_state and st.session_state.selected_collection_to_delete:
            selected_collection_to_delete = st.session_state.selected_collection_to_delete

            # デフォルトコレクションの削除を防止
            if selected_collection_to_delete == "default":
                st.error("デフォルトコレクションは削除できません。")
                st.session_state.delete_collection_confirmation_state = False
            else:
                # 確認ダイアログ
                st.warning(
                    f"コレクション '{selected_collection_to_delete}' を完全に削除します。この操作は元に戻せません。"
                )
                confirm_cols = st.columns([2, 2, 2])
                with confirm_cols[0]:
                    cancel_confirmed = st.button(
                        "キャンセル",
                        key="cancel_delete_collection_button",
                        use_container_width=True
                    )

                with confirm_cols[2]:
                    confirmed = st.button(
                        "削除を確定",
                        key="confirm_delete_collection_button",
                        type="primary",
                        use_container_width=True
                    )

                # キャンセルボタンが押された場合は確認状態をOFFに
                if cancel_confirmed:
                    st.session_state.delete_collection_confirmation_state = False
                    st.rerun()

                if confirmed:
                    with st.spinner(f"コレクション '{selected_collection_to_delete}' を削除中..."):
                        try:
                            # コレクションを削除
                            qdrant_manager.delete_collection(selected_collection_to_delete)

                            # 常に成功メッセージを表示
                            st.success(f"コレクション '{selected_collection_to_delete}' の削除が完了しました")
                            # 確認状態をリセット
                            st.session_state.delete_collection_confirmation_state = False
                            st.session_state.selected_collection_to_delete = None
                            # コレクション一覧を再取得して表示を更新
                            st.rerun()
                        except Exception as e:
                            st.error(f"削除処理中にエラーが発生しました: {str(e)}")
                            logger.error(f"コレクション削除エラー: {str(e)}")


def show_manage_component(qdrant_manager, logger):

    # タブを作成
    data_management_tabs = st.tabs(["ソース", "コレクション"])
    
    # ソースタブ
    with data_management_tabs[0]:
        _manage_sources(qdrant_manager, logger)

    # コレクションタブ
    with data_management_tabs[1]:
        _manage_collections(qdrant_manager, logger)
