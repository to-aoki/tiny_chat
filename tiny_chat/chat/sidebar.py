import streamlit as st


from tiny_chat.utils.llm_utils import get_llm_client, reset_ollama_model, identify_server
from tiny_chat.chat.chat_config import ChatConfig, ModelManager


def sidebar(config_file_path, logger):
    st.header("設定")

    server_mode = st.session_state.config["session_only_mode"]
    # 設定変更フラグ - すべての変更を追跡する統一変数
    settings_changed = False
    # サーバーURL、API Key、Azureフラグの変更を特別に追跡
    server_settings_changed = False

    if not server_mode:
        with st.expander("モデル設定", expanded=False):
            st.markdown('<span style="font-size: 12px;">変更時は「設定を反映」してください</span>', unsafe_allow_html=True)

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
                if st.session_state.infer_server_type == 'ollama':
                    # ollama向け操作 （他用途がある場合はアンロードしないほうがいいかもしれない）
                    reset_ollama_model(server_url=st.session_state.config["server_url"],
                                       model=st.session_state.config["selected_model"] )
                st.session_state.config["selected_model"] = model_input
                logger.info(f"モデルを変更: {model_input}")
                settings_changed = True

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
                max_value=2000000,
                value=st.session_state.config["message_length"],
                step=1000,
                help="入力最大メッセージ長を決定します",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

            max_completion_tokens = st.number_input(
                "生成トークン長",
                min_value=100,
                max_value=100000,
                value=st.session_state.config["max_completion_tokens"],
                step=100,
                help="出力最大トークン長を決定します",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

            context_length = st.number_input(
                "添付ファイル文字列長",
                min_value=500,
                max_value=1000000,
                value=st.session_state.config["context_length"],
                step=500,
                help="添付ファイルやURLコンテンツの取得最大長（切り詰める）文字数を指定します",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

            temperature = st.number_input(
                "温度",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.config["temperature"]),
                step=0.1,
                help="LLMの応答単語の確率分布を制御します（値が大きいと創造的/ハルシーネションが起こりやすいです）",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

            top_p = st.number_input(
                "確率累積値（top_p）",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.config["top_p"]),
                step=0.1,
                help="LLMの応答単語の生起確率累積値を制御します（値が大きくすると多様性をある程度維持します）",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )
            if st.session_state.config["use_web"]:
                uri_processing = st.checkbox(
                    "メッセージURL取得",
                    value=st.session_state.config["uri_processing"],
                    help="メッセージの最初のURLからコンテキストを取得し、プロンプトを拡張します",
                    disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
                )

                # 設定値変更の検出（フラグの一元管理）
                if uri_processing != st.session_state.config["uri_processing"] and not st.session_state.is_sending_message:
                    st.session_state.config["uri_processing"] = uri_processing
                    logger.info(f"URI処理設定を変更: {uri_processing}")
                    settings_changed = True

            # サーバー関連設定の変更をチェック
            server_url_changed = server_url != st.session_state.config["server_url"]
            api_key_changed = api_key != st.session_state.config["api_key"]
            is_azure_changed = is_azure != st.session_state.config["is_azure"]
            
            # サーバー設定変更フラグ
            server_settings_changed = server_url_changed or api_key_changed or is_azure_changed

            # モデルリスト更新ボタン
            if st.button("モデルリスト更新", disabled=st.session_state.is_sending_message):
                # 現在の設定値を使用
                current_server = st.session_state.config["server_url"]
                current_api_key = st.session_state.config["api_key"]
                current_is_azure = st.session_state.config["is_azure"]

                try:
                    st.session_state.openai_client = get_llm_client(
                        server_url=current_server,
                        api_key=current_api_key,
                        is_azure=current_is_azure
                    )

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

                    st.rerun()
                except Exception as e:
                    error_msg = f"モデルリスト更新中にエラーが発生しました: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

    meta_prompt = st.text_area(
        "メタプロンプト",
        value=st.session_state.config["meta_prompt"],
        height=150,
        help="LLMのsystem指示文字列を入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    if meta_prompt != st.session_state.config["meta_prompt"]:
        st.session_state.config["meta_prompt"] = meta_prompt
        settings_changed = True

    if not server_mode and server_settings_changed and not st.session_state.is_sending_message:
        # ログ記録
        if server_url_changed:
            logger.info(f"サーバーURLを変更: {server_url}")
            
        if is_azure_changed:
            logger.info(f"Azure設定を変更: {is_azure}")

        if api_key_changed:
            logger.info("APIキーを変更しました")


        # サーバー変更の場合はモデルリストを更新
        if server_url_changed or is_azure_changed:
            logger.info("サーバー変更に伴いモデルリストを更新中...")
            new_models, selected_model, api_success = ModelManager.update_models_on_server_change(
                server_url,
                api_key,
                st.session_state.config["selected_model"],
                is_azure=is_azure
            )

            st.session_state.available_models = new_models
            st.session_state.models_api_success = api_success
            st.session_state.config["is_azure"] = is_azure

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
            st.session_state.config["api_key"] = api_key
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

            st.session_state.config["api_key"] = api_key
            st.session_state.config["server_url"] = server_url
            st.session_state.infer_server_type = identify_server(
                server_url) if not st.session_state.config["is_azure"] else "azure"

            logger.info("OpenAIクライアント初期化完了")
            settings_changed = True
        except Exception as e:
            error_msg = f"OpenAI クライアントの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.openai_client = None

    # 設定を反映ボタン
    if not server_mode and st.button("設定を反映", disabled=st.session_state.is_sending_message):
        # 数値設定の変更を検出して反映
        if message_length != st.session_state.config["message_length"]:
            st.session_state.config["message_length"] = message_length
            settings_changed = True

        if max_completion_tokens != st.session_state.config["max_completion_tokens"]:
            st.session_state.config["max_completion_tokens"] = max_completion_tokens
            settings_changed = True

        if context_length != st.session_state.config["context_length"]:
            st.session_state.config["context_length"] = context_length
            settings_changed = True

        if temperature != st.session_state.config["temperature"]:
            st.session_state.config["temperature"] = temperature
            settings_changed = True

        if top_p != st.session_state.config["top_p"]:
            st.session_state.config["top_p"] = top_p
            settings_changed = True

        # サーバーモードでなければ設定ファイルに保存
        if not server_mode:
            # 設定をファイルに保存
            config = ChatConfig(
                server_url=st.session_state.config["server_url"],
                api_key=st.session_state.config["api_key"],
                selected_model=st.session_state.config["selected_model"],
                meta_prompt=meta_prompt,
                message_length=st.session_state.config["message_length"],
                max_completion_tokens=st.session_state.config["max_completion_tokens"],
                context_length=st.session_state.config["context_length"],
                uri_processing=st.session_state.config["uri_processing"],
                is_azure=st.session_state.config["is_azure"],
                session_only_mode=server_mode,
                rag_process_prompt=st.session_state.config["rag_process_prompt"],
                use_hyde=st.session_state.config["use_hyde"],
                use_step_back=st.session_state.config["use_step_back"],
                use_web=st.session_state.config["use_web"]
            )

            if config.save(config_file_path):
                logger.info("設定をファイルに保存しました")
                st.success("設定を更新し、ファイルに保存しました")
            else:
                logger.warning("設定ファイルへの保存に失敗しました")
                st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

        # 設定変更があった場合は画面を再読み込み
        if settings_changed:
            st.rerun()

    uploaded_json = st.file_uploader(
        "チャット履歴をインポート",
        type=["json"],
        help="以前に保存したチャット履歴JSONファイルをアップロードします",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # クエリ変換方式をラジオボタンで選択
    query_options = ["変換なし", "クエリ汎化(Step Back)", "仮クエリ回答(HYDE)"]

    # コールバック関数 - ラジオボタン選択時に即時反映する
    def on_query_conversion_change():
        """ラジオボタン選択変更時のコールバック関数"""
        # 選択値を取得して数値インデックスに変換
        option_name = st.session_state.query_conversion_radio
        if option_name == query_options[0]:
            new_mode = 0
        elif option_name == query_options[1]:
            new_mode = 1
        elif option_name == query_options[2]:
            new_mode = 2
        else:
            return

        # 前回値と比較して変更があれば設定を更新
        if "query_conversion_mode" not in st.session_state or st.session_state.query_conversion_mode != new_mode:
            # モードインデックスを更新
            st.session_state.query_conversion_mode = new_mode

            # 変更前の値を記録
            old_hyde = st.session_state.config["use_hyde"]
            old_step_back = st.session_state.config["use_step_back"]

            # モードに応じて設定値を更新
            if new_mode == 0:
                st.session_state.config["use_hyde"] = False
                st.session_state.config["use_step_back"] = False
            elif new_mode == 1:
                st.session_state.config["use_hyde"] = False
                st.session_state.config["use_step_back"] = True
            elif new_mode == 2:
                st.session_state.config["use_hyde"] = True
                st.session_state.config["use_step_back"] = False
            else:
                pass

            # 設定変更フラグを更新
            nonlocal settings_changed
            if old_hyde != st.session_state.config["use_hyde"] or \
                    old_step_back != st.session_state.config["use_step_back"]:
                settings_changed = True


    # セッション状態に初期値を設定
    if "query_conversion_mode" not in st.session_state:
        if not (st.session_state.config["use_hyde"] or st.session_state.config["use_step_back"]):
            current_mode = 0
        elif st.session_state.config["use_hyde"]:
            current_mode = 1
        else:
            current_mode = 2
        st.session_state.query_conversion_mode = current_mode

    # モード値を安全に取得（範囲チェック）
    current_mode = st.session_state.query_conversion_mode
    if current_mode < 0 or current_mode >= len(query_options):
        current_mode = 0
        st.session_state.query_conversion_mode = current_mode

    # 単一のラジオボタンコントロールを表示
    query_conversion_mode = st.radio(
        "RAGクエリ変換方式",
        options=query_options,
        index=current_mode,
        help="RAG利用時のクエリ変換方式を選択します。",
        disabled=st.session_state.is_sending_message,
        key="query_conversion_radio",
        on_change=on_query_conversion_change
    )

    if uploaded_json is not None:
        content = uploaded_json.getvalue().decode("utf-8")

        if st.button("インポートした履歴を適用", disabled=st.session_state.is_sending_message):
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
            st.sidebar.markdown("RAG（データベース）")

            with st.expander("検索設定", expanded=False):
                # top_kの設定
                rag_top_k = st.slider(
                    "最大検索件数 (top_k)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.db_config.top_k,
                    help="RAG検索で取得する最大文書数を設定します",
                    disabled=st.session_state.is_sending_message,
                )

                if rag_top_k != st.session_state.db_config.top_k:
                    st.session_state.db_config.top_k = rag_top_k

                # score_thresholdの設定
                rag_score_threshold = st.slider(
                    "スコアしきい値",
                    min_value=0.0,
                    max_value=5.0,
                    value=st.session_state.db_config.score_threshold,
                    step=0.01,
                    help="RAG検索で取得する文書の最小類似度スコアを設定します（高いほど関連性の高い文書のみ取得）",
                    disabled=st.session_state.is_sending_message,
                )

                if rag_score_threshold != st.session_state.db_config.score_threshold:
                    st.session_state.db_config.score_threshold = rag_score_threshold

                rag_process_prompt = st.text_area(
                    "検索結果指示",
                    value=st.session_state.config["rag_process_prompt"],
                    height=150,
                    help="検索後の情報活用をLLM指示する文字列を入力してください",
                    disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
                )

                if rag_process_prompt != st.session_state.config["rag_process_prompt"]:
                    st.session_state.config["rag_process_prompt"] = rag_process_prompt
                    settings_changed = True

                st.markdown('<span style="font-size: 12px;">保存する場合は「設定を反映」してください</span>',
                            unsafe_allow_html=True)

            # コレクション名の選択（サイドバーに表示）
            manager = get_or_create_qdrant_manager(logger)

            # コレクション一覧をQdrantManagerから取得
            available_collections = manager.get_collections()

            # コレクションがなければデフォルトを作成・表示
            if not available_collections:
                available_collections = ["default"]
                collection = Collection(collection_name="default")
                collection.top_k = rag_top_k
                collection.score_threshold = rag_score_threshold
                collection.save(qdrant_manager=manager)

            available_collections = [collection for collection in available_collections
                                     if collection != Collection.STORED_COLLECTION_NAME]

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
            if search_collection != manager.collection_name and not st.session_state.collection_changing:
                # 変更中フラグを設定
                st.session_state.collection_changing = True
                st.sidebar.info(f"コレクション変更: {manager.collection_name} → {search_collection}")

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