import streamlit as st

from config_manager import Config, ModelManager
from llm_utils import get_llm_client


def sidebar(config_file_path, logger):
    with st.sidebar:
        st.header("設定")

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
            config = Config(
                server_url=st.session_state.config["server_url"],
                api_key=st.session_state.config["api_key"],
                selected_model=model_input,
                meta_prompt=st.session_state.config["meta_prompt"],
                context_length=st.session_state.config["context_length"],
                message_length=st.session_state.config["message_length"],
                uri_processing=st.session_state.config["uri_processing"],
                is_azure=st.session_state.config["is_azure"]
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

        meta_prompt = st.text_area(
            "メタプロンプト",
            value=st.session_state.config["meta_prompt"],
            height=150,
            help="LLMへのsystem指示を入力してください",
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
            "メッセージURLコンテンツ取得",
            value=st.session_state.config["uri_processing"],
            help="メッセージの最初のURLからコンテキストを取得し、プロンプトを拡張します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

        if uri_processing != st.session_state.config["uri_processing"] and not st.session_state.is_sending_message:
            st.session_state.config["uri_processing"] = uri_processing
            logger.info(f"URI処理設定を変更: {uri_processing}")

        if (server_url != st.session_state.config["server_url"] or
                is_azure != st.session_state.config["is_azure"]) and not st.session_state.is_sending_message:
            logger.info(f"サーバーURLを変更: {server_url}")
            st.session_state.config["previous_server_url"] = st.session_state.config["server_url"]
            st.session_state.config["server_url"] = server_url
            st.session_state.config["api_key"] = api_key
            st.session_state.config["is_azure"] = is_azure

            logger.info("サーバー変更に伴いモデルリストを更新中...")
            new_models, selected_model, api_success = ModelManager.update_models_on_server_change(
                server_url,
                api_key,
                st.session_state.config["selected_model"],
                is_azure=st.session_state.config["is_azure"]
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

            try:
                logger.info("新しいサーバーでOpenAIクライアントを初期化中...")
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

        else:
            if api_key != st.session_state.config["api_key"] and not st.session_state.is_sending_message:
                # サーバURLは同じだがAPIキーだけ変更された場合（かつメッセージ送信中でない場合）
                logger.info("APIキーを変更しました")
                st.session_state.config["api_key"] = api_key
                logger.info("APIキー変更に伴いモデルリストを更新中...")
                models, api_success = ModelManager.fetch_available_models(
                    server_url,
                    api_key,
                    st.session_state.openai_client,
                    is_azure=is_azure
                )
                st.session_state.available_models = models
                st.session_state.models_api_success = api_success

                try:
                    logger.info("APIキー変更に伴いOpenAIクライアントを再初期化中...")
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

        if st.button("設定を反映", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
            logger.info("設定反映ボタンがクリックされました")
            # サーバURLが変更された場合はモデルリストを更新
            if server_url != st.session_state.config["server_url"] or is_azure != st.session_state.config["is_azure"]:
                logger.info(f"サーバーURLを変更: {server_url}")
                st.session_state.config["previous_server_url"] = st.session_state.config["server_url"]
                st.session_state.config["server_url"] = server_url
                st.session_state.config["api_key"] = api_key
                st.session_state.config["is_azure"] = is_azure

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
            else:
                # サーバURLは同じだがAPIキーだけ変更された場合
                if api_key != st.session_state.config["api_key"]:
                    logger.info("APIキーを変更しました")
                st.session_state.config["api_key"] = api_key
                logger.info("モデルリストを更新中...")
                models, api_success = ModelManager.fetch_available_models(
                    server_url,
                    api_key,
                    st.session_state.openai_client,
                    is_azure=is_azure
                )
                st.session_state.available_models = models
                st.session_state.models_api_success = api_success

            try:
                logger.info("OpenAIクライアントを再初期化中...")
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

            if meta_prompt != st.session_state.config["meta_prompt"]:
                logger.info("メタプロンプトを更新しました")
            st.session_state.config["meta_prompt"] = meta_prompt

            if message_length != st.session_state.config["message_length"]:
                logger.info(f"メッセージ長を更新: {message_length}")
            st.session_state.config["message_length"] = message_length

            if context_length != st.session_state.config["context_length"]:
                logger.info(f"コンテキスト長を更新: {context_length}")
            st.session_state.config["context_length"] = context_length

            config = Config(
                server_url=server_url,
                api_key=api_key,
                selected_model=st.session_state.config["selected_model"],
                meta_prompt=meta_prompt,
                message_length=message_length,
                context_length=context_length,
                uri_processing=uri_processing,
                is_azure=is_azure
            )

            if config.save(config_file_path):
                logger.info("設定をファイルに保存しました")
                st.success("設定を更新し、ファイルに保存しました")
            else:
                logger.warning("設定ファイルへの保存に失敗しました")
                st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

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
