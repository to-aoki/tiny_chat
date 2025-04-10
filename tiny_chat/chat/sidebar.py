import streamlit as st


from tiny_chat.utils.llm_utils import get_llm_client
from tiny_chat.chat.chat_config import ChatConfig, ModelManager


def sidebar(config_file_path, logger):
    st.header("設定")

    server_mode = st.session_state.config["session_only_mode"]

    # サーバーモードの状態を表示
    if server_mode:
        st.info("サーバーモードが有効です")

    else:
        # モデル選択または入力
        # APIに接続できる場合はドロップダウンリストを表示し、できない場合はテキスト入力欄を表示
        if st.session_state.models_api_success and st.session_state.available_models:
            model_input = st.selectbox(
                "モデル",
                st.session_state.available_models,
                index=st.session_state.available_models.index(
                    st.session_state.config["selected_model"]
                ) if st.session_state.config["selected_model"] in st.session_state.available_models else 0,
                help="API から取得したモデル一覧から選択します",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )
        else:
            model_input = st.text_input(
                "モデル",
                value=st.session_state.config["selected_model"],
                help="使用するモデル名を入力してください",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

        # 入力されたモデルを使用
        if model_input != st.session_state.config["selected_model"] and not st.session_state.is_sending_message:
            st.session_state.config["selected_model"] = model_input
            logger.info(f"モデルを変更: {model_input}")
            # 設定を外部ファイルに保存
            config = ChatConfig(
                server_url=st.session_state.config["server_url"],
                api_key=st.session_state.config["api_key"],
                selected_model=model_input,
                meta_prompt=st.session_state.config["meta_prompt"],
                context_length=st.session_state.config["context_length"],
                message_length=st.session_state.config["message_length"],
                uri_processing=st.session_state.config["uri_processing"],
                is_azure=st.session_state.config["is_azure"],
                session_only_mode=st.session_state.config["session_only_mode"]
            )
            config.save(config_file_path)
            logger.info("設定を保存しました")

        server_url = st.text_input(
            "サーバーURL",
            value=st.session_state.config["server_url"],
            help="OpenAI APIサーバーのURLを入力してください",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        api_key = st.text_input(
            "API Key",
            value=st.session_state.config["api_key"],
            type="password",
            help="APIキーを入力してください",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        is_azure = st.checkbox(
            "Azure OpenAIを利用",
            value=st.session_state.config["is_azure"],
            help="APIはAzure OpenAIを利用します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        message_length = st.number_input(
            "メッセージ長",
            min_value=1000,
            max_value=500000,
            value=st.session_state.config["message_length"],
            step=500,
            help="入力最大メッセージ長を決定します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        context_length = st.number_input(
            "添付ファイル文字列長",
            min_value=100,
            max_value=100000,
            value=st.session_state.config["context_length"],
            step=500,
            help="添付ファイルやURLコンテンツの取得最大長（切り詰める）文字数を指定します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        uri_processing = st.checkbox(
            "メッセージURL取得",
            value=st.session_state.config["uri_processing"],
            help="メッセージの最初のURLからコンテキストを取得し、プロンプトを拡張します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        if uri_processing != st.session_state.config["uri_processing"] and not st.session_state.is_sending_message:
            st.session_state.config["uri_processing"] = uri_processing
            logger.info(f"URI処理設定を変更: {uri_processing}")

        if not server_mode:
            # サーバーURL変更または Azure フラグの変更があった場合の処理
            server_or_azure_changed = (server_url != st.session_state.config["server_url"] or
                                      is_azure != st.session_state.config["is_azure"])

        # APIキーの変更をチェック
        api_key_changed = api_key != st.session_state.config["api_key"]

    meta_prompt = st.text_area(
        "メタプロンプト",
        value=st.session_state.config["meta_prompt"],
        height=150,
        help="LLMへのsystem指示を入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )
    if meta_prompt != st.session_state.config["meta_prompt"]:
        logger.info("メタプロンプトを更新しました")
        st.session_state.config["meta_prompt"] = meta_prompt

    if not server_mode:
        # いずれかの設定変更があった場合
        if (server_or_azure_changed or api_key_changed) and not st.session_state.is_sending_message:
            # ログ記録
            if server_or_azure_changed:
                logger.info(f"サーバーURLを変更: {server_url}")
                if is_azure != st.session_state.config["is_azure"]:
                    logger.info(f"Azure設定を変更: {is_azure}")

            if api_key_changed:
                logger.info("APIキーを変更しました")
            
            # 各フィールドの更新
            st.session_state.config["server_url"] = server_url
            st.session_state.config["api_key"] = api_key
            st.session_state.config["is_azure"] = is_azure

            # サーバー変更の場合はモデルリストを更新
            if server_or_azure_changed:
                logger.info("サーバー変更に伴いモデルリストを更新中...")
                new_models, selected_model, api_success = ModelManager.update_models_on_server_change(
                    server_url,
                    api_key,
                    st.session_state.config["selected_model"],
                    is_azure=is_azure
                )

                st.session_state.available_models = new_models
                st.session_state.models_api_success = api_success

                # モデルの自動変更通知 (新しいサーバーで現在のモデルが利用できない場合)
                if selected_model != st.session_state.config["selected_model"] and new_models:
                    old_model = st.session_state.config["selected_model"]
                    st.session_state.config["selected_model"] = selected_model
                    logger.warning(f"モデルを自動変更: {old_model} → {selected_model}")
                    st.info(
                        f"選択したモデル '{old_model}' は新しいサーバーでは利用できません。"
                        f"'{selected_model}' に変更されました。")

            # APIキーだけ変更された場合
            elif api_key_changed:
                logger.info("APIキー変更に伴いモデルリストを更新中...")
                models, api_success = ModelManager.fetch_available_models(
                    server_url,
                    api_key,
                    st.session_state.openai_client,
                    is_azure=is_azure
                )
                st.session_state.available_models = models
                st.session_state.models_api_success = api_success

            # クライアントの再初期化
            try:
                logger.info("設定変更に伴いOpenAIクライアントを初期化中...")
                st.session_state.openai_client = get_llm_client(
                    server_url=server_url,
                    api_key=api_key,
                    is_azure=is_azure
                )
                logger.info("OpenAIクライアント初期化完了")
            except Exception as e:
                error_msg = f"OpenAI クライアントの初期化に失敗しました: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.openai_client = None

        # 設定を反映ボタン
        if st.button("設定を反映", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
            logger.info("設定反映ボタンがクリックされました")

            # メタプロンプト、メッセージ長、コンテキスト長の変更を記録
            settings_changed = False

            if not server_mode:
                if message_length != st.session_state.config["message_length"]:
                    logger.info(f"メッセージ長を更新: {message_length}")
                    st.session_state.config["message_length"] = message_length
                    settings_changed = True

                if context_length != st.session_state.config["context_length"]:
                    logger.info(f"コンテキスト長を更新: {context_length}")
                    st.session_state.config["context_length"] = context_length
                    settings_changed = True

            if uri_processing != st.session_state.config["uri_processing"]:
                logger.info(f"URI処理設定を更新: {uri_processing}")
                st.session_state.config["uri_processing"] = uri_processing
                settings_changed = True

            # 設定をファイルに保存
            config = ChatConfig(
                server_url=server_url,
                api_key=api_key,
                selected_model=st.session_state.config["selected_model"],
                meta_prompt=meta_prompt,
                message_length=message_length,
                context_length=context_length,
                uri_processing=uri_processing,
                is_azure=is_azure,
                session_only_mode=server_mode
            )

            if st.session_state.config["session_only_mode"]:
                st.success("設定を更新しました (サーバーモード)")
            elif config.save(config_file_path):
                logger.info("設定をファイルに保存しました")
                st.success("設定を更新し、ファイルに保存しました")
            else:
                logger.warning("設定ファイルへの保存に失敗しました")
                st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

            if settings_changed or server_or_azure_changed or api_key_changed:
                st.rerun()

            # モデルリスト更新ボタン
            if st.button("モデルリスト更新", disabled=st.session_state.is_sending_message):

                # 現在の設定値を使用
                current_server = st.session_state.config["server_url"]
                current_api_key = st.session_state.config["api_key"]
                current_is_azure = st.session_state.config["is_azure"]

                try:
                    # モデルリストを再取得
                    models, api_success = ModelManager.fetch_available_models(
                        current_server,
                        current_api_key,
                        st.session_state.openai_client,
                        is_azure=current_is_azure
                    )

                    # 状態を更新
                    st.session_state.available_models = models
                    st.session_state.models_api_success = api_success

                    if not api_success:
                        logger.warning("モデルリスト取得に失敗しました")
                        st.error("モデルリストの取得に失敗しました。APIキーとサーバー設定を確認してください。")

                    # APIクライアントの再初期化
                    st.session_state.openai_client = get_llm_client(
                        server_url=current_server,
                        api_key=current_api_key,
                        is_azure=current_is_azure
                    )

                    st.rerun()
                except Exception as e:
                    error_msg = f"モデルリスト更新中にエラーが発生しました: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

    uploaded_json = st.file_uploader(
        "チャット履歴をインポート",
        type=["json"],
        help="以前に保存したチャット履歴JSONファイルをアップロードします",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    if uploaded_json is not None:
        logger.info(f"JSONファイルがアップロードされました: {uploaded_json.name}")
        content = uploaded_json.getvalue().decode("utf-8")

        if st.button("インポートした履歴を適用", disabled=st.session_state.is_sending_message):
            logger.info("履歴インポート適用ボタンがクリックされました")
            success = st.session_state.chat_manager.apply_imported_history(content)
            if success:
                logger.info("メッセージ履歴のインポートに成功しました")
                st.success("メッセージ履歴を正常にインポートしました")
                st.rerun()
            else:
                logger.error("メッセージ履歴のインポートに失敗しました: 無効なフォーマット")
                st.error("JSONのインポートに失敗しました: 無効なフォーマットです")

    # RAGモードがオンの場合は常にデータベース検索機能を表示する
    if st.session_state.rag_mode:
        try:
            from tiny_chat.database.database import get_or_create_qdrant_manager
            from tiny_chat.database.qdrant.collection import Collection

            # 検索用サイドバー設定
            st.sidebar.markdown("検索")

            # コレクション名の選択（サイドバーに表示）
            manager = get_or_create_qdrant_manager(logger)

            # コレクション一覧をQdrantManagerから取得
            available_collections = manager.get_collections()

            # コレクションがなければデフォルトのものを表示
            if not available_collections:
                available_collections = ["default"]

            available_collections = [collection for collection in available_collections if collection != Collection.STORED_COLLECTION_NAME]

            # コレクション選択の状態管理
            if "selected_collection" not in st.session_state:
                st.session_state.selected_collection = manager.collection_name

            # コレクション変更を検出するための一時フラグ
            if "collection_changing" not in st.session_state:
                st.session_state.collection_changing = False

            # サイドバーにコレクション選択を表示（メインエリアではなく）
            st.sidebar.markdown("コレクション選択", help="Qdrantデータベースで利用するコレクション（DB空間）を選択します")
            _ = st.sidebar.selectbox(
                "コレクション",  # 空のラベルから有効なラベルに変更
                available_collections,
                index=available_collections.index(
                    manager.collection_name
                ) if manager.collection_name in available_collections else 0,
                label_visibility="collapsed",  # ラベルを視覚的に非表示にする
                disabled=st.session_state.is_sending_message,
                key="collection_select",  # 固定のキーを使用
                on_change=lambda: setattr(st.session_state, "selected_collection", st.session_state.collection_select)
            )
            search_collection = st.session_state.collection_select

            # 選択されたコレクションに切り替え
            if (search_collection != manager.collection_name and 
                not st.session_state.collection_changing):
                # 変更中フラグを設定
                st.session_state.collection_changing = True
                st.sidebar.info(f"コレクション変更: {manager.collection_name} → {search_collection}")
                logger.info(f"コレクション変更: {manager.collection_name} → {search_collection}")
                
                # コレクション名を変更
                manager.set_collection_name(search_collection)
                st.session_state.selected_collection = search_collection

            elif st.session_state.collection_changing:
                # 再実行後、フラグをリセット
                st.session_state.collection_changing = False

            doc_count = manager.count_documents()
            st.sidebar.code(f"現在のコレクション: {search_collection}\n登録ドキュメント数: {doc_count}")

        except Exception as e:
            # データベース接続時のエラーを表示
            logger.error(f"データベース接続エラー: {str(e)}")
            st.sidebar.error("データベース接続エラー")
