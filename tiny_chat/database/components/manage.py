import streamlit as st


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
        # 選択したソースをセッション状態に保存
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
        from tiny_chat.database.qdrant.collection import Collection
        # コレクション情報を表示
        st.write(f"利用可能なコレクション: {len(collections)}個")
        try:
            # scrollメソッドを使用してデータを直接取得
            results = []
            offset = None
            while True:
                batch, next_offset = qdrant_manager.client.scroll(
                    collection_name=Collection.STORED_COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                if not batch:
                    break
                for point in batch:
                    collection_name = point.payload.get("collection_name")
                    if collection_name:
                        results.append({
                            "id": point.id,
                            "collection_name": collection_name,
                            "description": point.payload.get("text", ""),
                            "metadata": {k: v for k, v in point.payload.items()
                                         if k not in ["text", "collection_name"]}
                        })
                if len(batch) < 100 or next_offset is None:
                    break
                offset = next_offset
            collection_descriptions = results
        except Exception as e:
            logger.error(f"コレクション説明の取得中にエラーが発生しました: {str(e)}")
            collection_descriptions = []
        # 説明情報をDict形式に変換して高速アクセスできるようにする
        description_map = {desc['collection_name']: desc for desc in collection_descriptions}
        # 各コレクションの情報を表示
        collection_infos = []
        for col_name in collections:
            try:
                # 現在のコレクション名を保持
                original_collection = qdrant_manager.collection_name
                # 文書数を取得（コレクション名を明示的に指定）
                doc_count = qdrant_manager.count_documents(collection_name=col_name)
                # コレクション説明を取得
                description = ""
                if col_name in description_map:
                    description = description_map[col_name].get('description', '')
                # コレクションに関する情報を収集
                collection_infos.append({
                    "name": col_name,
                    "description": description,
                    "doc_count": doc_count,
                    "is_current": col_name == original_collection
                })
            except Exception as e:
                logger.error(f"コレクション情報取得エラー ({col_name}): {str(e)}")
                collection_infos.append({
                    "name": col_name,
                    "description": "エラー",
                    "doc_count": "エラー",
                    "is_current": col_name == qdrant_manager.collection_name
                })
        # カスタムテーブルでコレクション一覧とチェックボックスを一緒に表示
        # テーブルヘッダー
        col1, col2, col3, col4, col5 = st.columns([1, 2, 4, 1, 1])
        with col1:
            st.write("**選択**")
        with col2:
            st.write("**コレクション名**")
        with col3:
            st.write("**説明**")
        with col4:
            st.write("**文書数**")
        with col5:
            st.write("**削除**")
        # 区切り線
        st.markdown("---")
        # 各コレクションの行を表示
        for col_info in collection_infos:
            col_name = col_info["name"]
            description = col_info["description"]
            is_current = col_info["is_current"]
            # セッション状態の初期化
            if f"collection_active_{col_name}" not in st.session_state:
                st.session_state[f"collection_active_{col_name}"] = is_current
            # 行の表示
            col1, col2, col3, col4, col5 = st.columns([1, 2, 4, 1, 1])
            # チェックボックスを表示
            with col1:
                checkbox_selected = st.checkbox(
                    f"選択 {col_name}",  # アクセシビリティのために非空のラベルを設定
                    value=st.session_state[f"collection_active_{col_name}"],
                    key=f"collection_checkbox_{col_name}",
                    disabled=is_current,  # 現在使用中のコレクションはチェックを外せないようにする
                    label_visibility="collapsed"  # ラベルを非表示にする
                )
            # コレクション名を表示
            with col2:
                st.write(f"**{col_name}**" + (" (現在使用中)" if is_current else ""))
            # 説明を表示
            with col3:
                st.write(f"{description}" if description else "")
            # 文書数
            with col4:
                st.write(f"{col_info['doc_count']}")
            # 削除ボタン
            with col5:
                # デフォルトコレクションは削除不可
                button_disabled = is_current or col_name == Collection.STORED_COLLECTION_NAME
                delete_button = st.button(
                    "削除",
                    key=f"delete_button_{col_name}",
                    disabled=button_disabled,
                    type="secondary",
                )
                # 削除ボタンが押された場合
                if delete_button:
                    st.session_state.selected_collection_to_delete = col_name
                    st.session_state.delete_collection_confirmation_state = True
            
            # チェックボックスの状態が変更された場合の処理
            if checkbox_selected != st.session_state[f"collection_active_{col_name}"]:
                st.session_state[f"collection_active_{col_name}"] = checkbox_selected
                # チェックされたコレクションをアクティブにする
                if checkbox_selected:
                    # 他のコレクションのチェックを外す
                    for other_col_info in collection_infos:
                        other_col_name = other_col_info["name"]
                        if other_col_name != col_name:
                            st.session_state[f"collection_active_{other_col_name}"] = False
                    # qdrant_managerのcollection_nameを変更
                    qdrant_manager.collection_name = col_name
                    st.success(f"コレクション '{col_name}' を使用するように設定しました。")
                    st.rerun()
        # コレクション新規作成
        with st.expander("コレクション新規作成", expanded=False):
            # コレクション名の入力
            new_collection_name = st.text_input(
                "コレクション名",
                value="",
                help="新しいコレクション名を入力してください。既存のコレクション名と重複しないようにしてください。",
                key="new_collection_name"
            )
            # 説明文の入力
            new_collection_description = st.text_area(
                "説明",
                value="",
                help="コレクションの説明を入力してください。",
                height=100,
                key="new_collection_description"
            )
            # 詳細設定
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    # RAG戦略の選択
                    rag_strategy_options = ["bm25_static"]
                    rag_strategy = st.selectbox(
                        "RAG戦略",
                        options=rag_strategy_options,
                        index=0,
                        help="検索・抽出戦略を選択してください。",
                        key="new_rag_strategy"
                    )
                    # GPU使用の選択
                    use_gpu = st.checkbox(
                        "GPU使用",
                        value=False,
                        help="GPUを使用する場合はチェックしてください。",
                        key="new_use_gpu"
                    )
                with col2:
                    # チャンクサイズの入力
                    chunk_size = st.number_input(
                        "チャンクサイズ",
                        min_value=128,
                        max_value=8192,
                        value=1024,
                        step=128,
                        help="テキストを分割するチャンクのサイズを指定してください。",
                        key="new_chunk_size"
                    )
                    # チャンクオーバーラップの入力
                    chunk_overlap = st.number_input(
                        "チャンクオーバーラップ",
                        min_value=0,
                        max_value=512,
                        value=24,
                        step=8,
                        help="チャンク間のオーバーラップするトークン数を指定してください。",
                        key="new_chunk_overlap"
                    )
                    # 上位K件の入力
                    top_k = st.number_input(
                        "検索結果件数",
                        min_value=1,
                        max_value=20,
                        value=3,
                        step=1,
                        help="検索結果として返す上位件数を指定してください。",
                        key="new_top_k"
                    )
                    # スコアしきい値の入力
                    score_threshold = st.number_input(
                        "スコアしきい値",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.4,
                        step=0.05,
                        format="%.2f",
                        help="検索結果に含めるしきい値を指定してください。",
                        key="new_score_threshold"
                    )
            # 作成ボタン
            create_cols = st.columns([3, 3, 3])
            with create_cols[1]:
                create_pressed = st.button(
                    "コレクション作成",
                    key="create_collection_button",
                    type="primary",
                    use_container_width=True
                )
            # コレクション作成処理
            if create_pressed:
                if not new_collection_name:
                    st.error("コレクション名を入力してください。")
                elif new_collection_name in collections:
                    st.error(
                        f"コレクション名 '{new_collection_name}' は既に使用されています。別の名前を指定してください。")
                else:
                    with st.spinner(f"コレクション '{new_collection_name}' を作成中..."):
                        try:
                            # Collectionクラスを使用してコレクションを作成
                            new_collection = Collection(
                                collection_name=new_collection_name,
                                description=new_collection_description,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k,
                                score_threshold=score_threshold,
                                rag_strategy=rag_strategy,
                                use_gpu=use_gpu
                            )
                            # コレクション情報を保存
                            new_collection.save(qdrant_manager=qdrant_manager)
                            st.success(f"コレクション '{new_collection_name}' を作成しました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"コレクション作成中にエラーが発生しました: {str(e)}")
                            logger.error(f"コレクション作成エラー: {str(e)}")
        # コレクション説明更新
        with st.expander("コレクション説明の更新", expanded=False):
            # コレクションの選択
            update_collection = st.selectbox(
                "コレクション",
                options=collections,
                help="説明を更新/作成するコレクションを選択してください。",
                key="collection_select_update"
            )
            # 選択したコレクションの現在の説明を取得
            current_description = ""
            if update_collection in description_map:
                current_description = description_map[update_collection].get('description', '')
            # 説明の編集
            new_description = st.text_area(
                "説明",
                value=current_description,
                help="コレクションの説明を入力してください。",
                height=100,
                key="collection_description_update"
            )
            # 更新ボタン
            update_cols = st.columns([3, 3, 3])
            with update_cols[1]:
                update_pressed = st.button(
                    "更新",
                    key="update_description_button",
                    type="primary",
                    use_container_width=True
                )
            if update_pressed and update_collection:
                with st.spinner(f"コレクション '{update_collection}' の説明を更新中..."):
                    try:
                        Collection.update_description(
                            collection_name=update_collection, description=new_description,
                            qdrant_manager=qdrant_manager)
                        st.success(f"コレクション '{update_collection}' の説明を更新しました")
                        # 更新後に画面を更新して表示を最新化
                        st.rerun()
                    except Exception as e:
                        st.error(f"説明の更新中にエラーが発生しました: {str(e)}")
                        logger.error(f"コレクション説明更新エラー: {str(e)}")
        # 削除確認状態の管理
        if "delete_collection_confirmation_state" not in st.session_state:
            st.session_state.delete_collection_confirmation_state = False
        if "selected_collection_to_delete" not in st.session_state:
            st.session_state.selected_collection_to_delete = None
        # 削除実行
        if st.session_state.delete_collection_confirmation_state and st.session_state.selected_collection_to_delete:
            selected_collection_to_delete = st.session_state.selected_collection_to_delete
            # デフォルトコレクションの削除を防止
            if selected_collection_to_delete == Collection.STORED_COLLECTION_NAME:
                st.error("コレクション管理は削除できません。")
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
                            # コレクション管理から削除
                            Collection.delete(selected_collection_to_delete)
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

    tab_object = ["🏷 ️ソース", "📗 コレクション"]
    if 'active_object' not in st.session_state:
        st.session_state.active_object = tab_object[0]

    st.radio(
        "対象オブジェクト",
        tab_object,
        key='active_object',
        horizontal=True,
        label_visibility="collapsed"
    )

    # ソースタブ
    if st.session_state.active_object == tab_object[0]:
        _manage_sources(qdrant_manager, logger)

    # コレクションタブ
    if st.session_state.active_object == tab_object[1]:
        _manage_collections(qdrant_manager, logger)