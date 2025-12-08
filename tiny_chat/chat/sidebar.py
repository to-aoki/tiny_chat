import streamlit as st

from tiny_chat.utils.llm_utils import get_llm_client, reset_ollama_model, identify_server
from tiny_chat.chat.chat_config import ChatConfig, ModelManager


def sidebar(config_file_path, logger):
    st.header("設定")

    # 設定変更フラグと関連フラグの初期化
    if "settings_changed" not in st.session_state:
        st.session_state.settings_changed = False
    if "server_reinit_needed" not in st.session_state:
        st.session_state.server_reinit_needed = False
    if "just_changed_config_keys" not in st.session_state:
        st.session_state.just_changed_config_keys = set()

    server_mode = st.session_state.config["session_only_mode"]

    def set_server_reinit_flag():
        st.session_state.server_reinit_needed = True

    def update_config_value(widget_key, config_key, is_float=False, is_int=False, is_bool=False,
                            log_message_prefix=None, post_update_callback=None):
        if widget_key not in st.session_state:
            return

        new_value = st.session_state[widget_key]
        current_config_value = st.session_state.config.get(config_key)

        original_new_value_for_log = new_value  # ログ用

        # 型変換と値の正規化
        if is_float:
            new_value = float(new_value)
            if current_config_value is not None:
                try:
                    current_config_value = float(current_config_value)
                except (ValueError, TypeError):
                    pass  # 比較のために変換できない場合はそのまま
        elif is_int:
            new_value = int(new_value)
            if current_config_value is not None:
                try:
                    current_config_value = int(current_config_value)
                except (ValueError, TypeError):
                    pass
        elif is_bool:  # チェックボックス用
            new_value = bool(new_value)
            if current_config_value is not None:
                try:
                    current_config_value = bool(current_config_value)
                except (ValueError, TypeError):
                    pass

        # 変更があった場合のみ更新
        if new_value != current_config_value:
            old_value_for_log = st.session_state.config.get(config_key, "N/A")
            st.session_state.config[config_key] = new_value
            st.session_state.settings_changed = True

            # 変更された設定キーを記録
            st.session_state.just_changed_config_keys.add(config_key)

            _log_prefix = log_message_prefix if log_message_prefix else f"設定変更 ({config_key})"
            logger.info(f"{_log_prefix}: {old_value_for_log} -> {original_new_value_for_log}")

            if post_update_callback:
                post_update_callback()

    def handle_model_change(widget_key_to_read):
        if widget_key_to_read not in st.session_state:
            return
        
        # メッセージ送信中は処理をスキップ
        if st.session_state.is_sending_message:
            # UIの値を元に戻す
            st.session_state[widget_key_to_read] = st.session_state.config["selected_model"]
            return

        new_model = st.session_state[widget_key_to_read]

        if new_model != st.session_state.config["selected_model"]:
            if st.session_state.infer_server_type == 'ollama':
                # Ollamaの場合、旧モデルをアンロードする（必要な場合）
                reset_ollama_model(server_url=st.session_state.config["server_url"],
                                   model=st.session_state.config["selected_model"])
            
            # 古いモデル名を保存（ログ用）
            old_model = st.session_state.config["selected_model"]
            
            st.session_state.config["selected_model"] = new_model
            logger.info(f"モデルを変更: {old_model} → {new_model}")
            st.session_state.settings_changed = True
            st.session_state.just_changed_config_keys.add("selected_model")

    with st.expander("モデル設定", expanded=False):
        if not server_mode:
            st.markdown('<span style="font-size: 12px;">変更時は「設定を反映」してください</span>',
                        unsafe_allow_html=True)

        # モデル選択または入力 (コールバック対応)
        model_widget_key = ""
        if st.session_state.models_api_success and st.session_state.available_models:
            model_widget_key = "model_select_widget"
            current_model_idx = 0
            
            # 現在のモデルがリストにない場合の処理を改善
            if st.session_state.config["selected_model"] in st.session_state.available_models:
                current_model_idx = st.session_state.available_models.index(st.session_state.config["selected_model"])
            elif st.session_state.available_models:  # 利用可能なモデルがあるが、選択中のモデルがリストにない場合
                # モデルがリストにない場合、最初のモデルに自動的に変更
                logger.warning(
                    f"現在の選択モデル '{st.session_state.config['selected_model']}' は利用可能なモデルリストにありません。"
                    f"'{st.session_state.available_models[0]}' に自動変更します。"
                )
                # 設定を実際に更新
                st.session_state.config["selected_model"] = st.session_state.available_models[0]
                st.session_state.settings_changed = True
                st.session_state.just_changed_config_keys.add("selected_model")
            
            st.selectbox(
                "モデル",
                st.session_state.available_models,
                index=current_model_idx,
                help="API から取得したモデル一覧から選択します",
                disabled=st.session_state.is_sending_message,
                key=model_widget_key,
                on_change=handle_model_change,
                args=(model_widget_key,)
            )
        else:
            model_widget_key = "model_text_input_widget"
            st.text_input(
                "モデル",
                value=st.session_state.config["selected_model"],
                help="使用するモデル名を入力してください",
                disabled=st.session_state.is_sending_message,
                key=model_widget_key,
                on_change=handle_model_change,
                args=(model_widget_key,)
            )

        # サーバー設定 (APIキー、サーバーURL、Azure設定はクライアントモードのみ)
        if not server_mode:
            st.text_input(
                "サーバーURL",
                value=st.session_state.config["server_url"],
                help="OpenAI APIサーバーのURLを入力してください",
                disabled=st.session_state.is_sending_message,
                key="server_url_widget",
                on_change=update_config_value,
                args=("server_url_widget", "server_url", False, False, False, "サーバーURL", set_server_reinit_flag)
            )
            st.text_input(
                "API Key",
                value=st.session_state.config["api_key"],
                type="password",
                help="APIキーを入力してください",
                disabled=st.session_state.is_sending_message,
                key="api_key_widget",
                on_change=update_config_value,
                args=("api_key_widget", "api_key", False, False, False, "APIキー", set_server_reinit_flag)
            )
            st.checkbox(
                "Azure OpenAIを利用",
                value=st.session_state.config["is_azure"],
                help="APIはAzure OpenAIを利用します",
                disabled=st.session_state.is_sending_message,
                key="is_azure_widget",
                on_change=update_config_value,
                args=("is_azure_widget", "is_azure", False, False, True, "Azure設定", set_server_reinit_flag)
            )

        # タイムアウト設定
        st.number_input(
            "クライアントタイムアウト (秒)",
            min_value=1.0,
            max_value=600.0,  # 適宜調整
            value=float(st.session_state.config["timeout"]),  # ChatConfigでデフォルトが設定されている想定
            step=1.0,
            format="%.1f",
            help="LLMの接続・応答のタイムアウト値を設定します",
            disabled=st.session_state.is_sending_message,
            key="client_timeout_widget",
            on_change=update_config_value,
            args=(
                "client_timeout_widget",
                "timeout",
                True,  # is_float
                False,  # is_int
                False,  # is_bool
                "タイムアウト",
                set_server_reinit_flag
            )
        )

        st.number_input("メッセージ長", min_value=1000, max_value=2000000,
                        value=int(st.session_state.config["message_length"]), step=1000, help="入力最大メッセージ長",
                        disabled=st.session_state.is_sending_message, key="message_length_widget",
                        on_change=update_config_value,
                        args=("message_length_widget", "message_length", False, True, False, "メッセージ長"))
        st.number_input("生成トークン長", min_value=100, max_value=100000,
                        value=int(st.session_state.config["max_completion_tokens"]), step=100,
                        help="出力最大トークン長", disabled=st.session_state.is_sending_message,
                        key="max_completion_tokens_widget", on_change=update_config_value,
                        args=("max_completion_tokens_widget", "max_completion_tokens", False, True, False,
                              "生成トークン長"))
        st.number_input("添付ファイル文字列長", min_value=500, max_value=1000000,
                        value=int(st.session_state.config["context_length"]), step=500,
                        help="添付ファイルやURLコンテンツの取得最大長", disabled=st.session_state.is_sending_message,
                        key="context_length_widget", on_change=update_config_value,
                        args=("context_length_widget", "context_length", False, True, False, "添付ファイル文字列長"))
        st.number_input("最大添付ファイル数", min_value=1, max_value=30,
                        value=int(st.session_state.config.get("max_attachment_files", 2)), step=1,
                        help="一度に添付できるファイルの最大数", disabled=st.session_state.is_sending_message,
                        key="max_attachment_files_widget", on_change=update_config_value,
                        args=("max_attachment_files_widget", "max_attachment_files", False, True, False, "最大添付ファイル数"))
        st.number_input("温度", min_value=0.0, max_value=2.0, value=float(st.session_state.config["temperature"]),
                        step=0.1, help="LLMの応答単語の確率分布制御", disabled=st.session_state.is_sending_message,
                        key="temperature_widget", on_change=update_config_value,
                        args=("temperature_widget", "temperature", True, False, False, "温度"))
        st.number_input("確率累積値（top_p）", min_value=0.0, max_value=1.0,
                        value=float(st.session_state.config["top_p"]), step=0.1,
                        help="LLMの応答単語の生起確率累積値制御", disabled=st.session_state.is_sending_message,
                        key="top_p_widget", on_change=update_config_value,
                        args=("top_p_widget", "top_p", True, False, False, "top_p"))

        if st.session_state.config["use_web"]:
            st.checkbox("メッセージURL取得", value=st.session_state.config["uri_processing"],
                        help="メッセージの最初のURLからコンテキストを取得",
                        disabled=st.session_state.is_sending_message, key="uri_processing_widget",
                        on_change=update_config_value,
                        args=("uri_processing_widget", "uri_processing", False, False, True, "URI処理設定"))

        # モデルリスト更新ボタン
        if st.button("モデルリスト更新", disabled=st.session_state.is_sending_message):
            current_server = st.session_state.config["server_url"]
            current_api_key = st.session_state.config["api_key"]
            current_is_azure = st.session_state.config["is_azure"]
            current_timeout = float(st.session_state.config["timeout"])
            
            # 現在選択されているモデルを保存
            previous_selected_model = st.session_state.config["selected_model"]
            
            try:
                st.session_state.openai_client = get_llm_client(server_url=current_server, api_key=current_api_key,
                                                                is_azure=current_is_azure, timeout=current_timeout)
                models, api_success = ModelManager.fetch_available_models(current_server, current_api_key,
                                                                          st.session_state.openai_client,
                                                                          is_azure=current_is_azure)
                st.session_state.available_models = models
                st.session_state.models_api_success = api_success
                
                if not api_success:
                    logger.warning("モデルリスト取得に失敗しました")
                    st.error("モデルリストの取得に失敗しました。APIキーとサーバー設定を確認してください。")
                else:
                    # モデルリストが正常に取得できた場合
                    if models and previous_selected_model not in models:
                        # 以前のモデルが新しいリストにない場合
                        new_selected_model = models[0]
                        st.session_state.config["selected_model"] = new_selected_model
                        st.session_state.settings_changed = True
                        st.session_state.just_changed_config_keys.add("selected_model")
                        
                        st.info(f"選択されていたモデル '{previous_selected_model}' は利用できません。"
                               f"'{new_selected_model}' に変更されました。")
                        logger.info(f"モデルリスト更新によりモデルを自動変更: {previous_selected_model} → {new_selected_model}")
                    elif models:
                        st.success(f"モデルリストを更新しました。{len(models)}個のモデルが利用可能です。")
                
                st.rerun()
            except Exception as e:
                error_msg = f"モデルリスト更新中にエラー: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

    st.text_area("メタプロンプト", value=st.session_state.config["meta_prompt"], height=150,
                 help="LLMのsystem指示文字列を入力してください", disabled=st.session_state.is_sending_message,
                 key="meta_prompt_widget", on_change=update_config_value,
                 args=("meta_prompt_widget", "meta_prompt", False, False, False, "メタプロンプト"))

    # プロンプトインポート機能
    prompt_file = st.file_uploader("プロンプトのインポート", type=["md", "txt", "prompt"],
                                   help="テキストファイルをインポートしてチャット入力欄に挿入します",
                                   disabled=st.session_state.is_sending_message)
    if prompt_file is not None:
        encodings_to_try = ['utf-8-sig', 'utf-8', 'cp932', 'euc_jp', 'shift_jis']
        prompt_content = None
        for encoding in encodings_to_try:
            try:
                prompt_content = prompt_file.getvalue().decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not prompt_content:
            logger.error("プロンプトファイルの読み込みに失敗したか、空のファイルです")
            st.error("プロンプトファイルの読み込みに失敗したか、空のファイルです")
        else:
            try:
                prompt_content_js = prompt_content.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
                direct_js = f"""<script>
                    function insert_chat_text_direct() {{
                        var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                        nativeInputValueSetter.call(chatInput, "{prompt_content_js}");
                        var event = new Event('input', {{ bubbles: true}}); chatInput.dispatchEvent(event);
                    }} insert_chat_text_direct(); </script>"""
                st.components.v1.html(direct_js, height=0)
                st.success(f"ファイル '{prompt_file.name}' の内容をチャット入力欄に挿入しました")
            except Exception as e:
                logger.error(f"プロンプトファイルの読み込みに失敗しました: {str(e)}")
                st.error(f"ファイルの読み込みに失敗しました: {str(e)}")

    # サーバー設定変更時の処理 (クライアント再初期化など)
    if not server_mode and st.session_state.get("server_reinit_needed",
                                                False) and not st.session_state.is_sending_message:
        logger.info("サーバー設定変更を検知。クライアント再初期化処理を開始します。")

        current_server_url = st.session_state.config["server_url"]
        current_api_key = st.session_state.config["api_key"]
        current_is_azure = st.session_state.config["is_azure"]
        current_timeout = float(st.session_state.config["timeout"])
        # モデルリスト更新前の選択モデル名を保持（更新後に比較するため）
        current_selected_model_before_reinit = st.session_state.config["selected_model"]

        changed_keys = st.session_state.get("just_changed_config_keys", set())

        # サーバーURLまたはAzure設定が変更された場合、モデルリストも更新
        if "server_url" in changed_keys or "is_azure" in changed_keys:
            logger.info("サーバーURLまたはAzure設定の変更に伴いモデルリストを更新中...")
            try:
                new_models, selected_model_after_update, api_success = ModelManager.update_models_on_server_change(
                    current_server_url, current_api_key,
                    current_selected_model_before_reinit,  # 更新前のモデル名を渡す
                    is_azure=current_is_azure
                )
                st.session_state.available_models = new_models
                st.session_state.models_api_success = api_success

                if not api_success:
                    logger.warning("クライアント再初期化中のモデルリスト更新に失敗しました。")

                if selected_model_after_update != current_selected_model_before_reinit and new_models:
                    st.session_state.config["selected_model"] = selected_model_after_update
                    logger.warning(
                        f"モデルを自動変更: {current_selected_model_before_reinit} → {selected_model_after_update}")
                    st.info(
                        f"選択したモデル '{current_selected_model_before_reinit}' は新しいサーバー/設定では利用できません。"
                        f"'{selected_model_after_update}' に変更されました。"
                    )
                elif not new_models and api_success:  # API成功だがモデルリストが空
                    logger.warning("モデルリストが空です。サーバー設定を確認してください。")
                    st.session_state.config["selected_model"] = ""  # 適切なデフォルト値に
            except Exception as e:
                logger.error(f"モデルリスト更新中にエラー（サーバー再初期化プロセス）: {str(e)}")
                st.warning(f"モデルリストの更新に失敗しました: {str(e)}")

        try:
            logger.info("設定変更に伴いOpenAIクライアントを初期化中...")
            st.session_state.openai_client = get_llm_client(
                server_url=current_server_url, api_key=current_api_key,
                is_azure=current_is_azure, timeout=current_timeout,
            )
            st.session_state.infer_server_type = identify_server(
                current_server_url) if not current_is_azure else "azure"
            logger.info("OpenAIクライアント初期化完了")
            st.success("サーバー設定が更新され、クライアントが再接続されました。")
        except Exception as e:
            error_msg = f"OpenAI クライアントの初期化に失敗: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.openai_client = None  # クライアント初期化失敗
        finally:
            st.session_state.server_reinit_needed = False  # フラグをリセット
            st.session_state.just_changed_config_keys.clear()  # 変更キー記録もリセット
            st.rerun()  # UIを更新して反映

    # 「設定を反映」ボタン
    if not server_mode and st.button("設定保存", disabled=st.session_state.is_sending_message,
                                     help="チャット設定をセッションに反映し、設定値をファイル保存します"):
        # ChatConfigインスタンス作成時にst.session_state.configから最新の値を取得し、型を保証
        config_to_save = ChatConfig(
            server_url=str(st.session_state.config["server_url"]),
            api_key=str(st.session_state.config["api_key"]),
            selected_model=str(st.session_state.config["selected_model"]),
            meta_prompt=str(st.session_state.config["meta_prompt"]),
            message_length=int(st.session_state.config["message_length"]),
            max_completion_tokens=int(st.session_state.config["max_completion_tokens"]),
            context_length=int(st.session_state.config["context_length"]),
            uri_processing=bool(st.session_state.config["uri_processing"]),
            is_azure=bool(st.session_state.config["is_azure"]),
            session_only_mode=server_mode,  # bool
            rag_process_prompt=str(st.session_state.config["rag_process_prompt"]),
            use_hyde=bool(st.session_state.config["use_hyde"]),
            use_step_back=bool(st.session_state.config["use_step_back"]),
            use_web=bool(st.session_state.config["use_web"]),
            web_top_k=int(st.session_state.config["web_top_k"]),
            use_multi=bool(st.session_state.config["use_multi"]),
            use_deep=bool(st.session_state.config["use_deep"]),
            temperature=float(st.session_state.config["temperature"]),
            top_p=float(st.session_state.config["top_p"]),
            timeout=float(st.session_state.config["timeout"]),
            max_attachment_files=int(st.session_state.config.get("max_attachment_files", 5)))

        if config_to_save.save(config_file_path):
            logger.info("設定をファイルに保存しました")
            st.success("設定を更新し、ファイルに保存しました")
        else:
            logger.warning("設定ファイルへの保存に失敗しました")
            st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

        if st.session_state.settings_changed:
            st.session_state.settings_changed = False  # rerun前にリセット
            st.rerun()

    # Ollamaモデルリセットボタン
    if st.session_state.infer_server_type == "ollama":
        st.caption("応答異常の場合にリセット")
        if st.button("モデルリセット", disabled=st.session_state.is_sending_message,
                     help="LLM(Ollama)の応答が異常な際にクリックし、LLMの再ロード"):
            reset_ollama_model(server_url=st.session_state.config["server_url"],
                               model=st.session_state.config["selected_model"])

    # クエリ変換設定
    query_options = ["変換なし", "クエリ汎化(Step Back)", "仮クエリ回答(HYDE)"]
    if st.session_state.infer_server_type != 'other': query_options.extend(["マルチクエリ生成", "DeepSearch"])

    def on_query_conversion_change_callback():
        option_name = st.session_state.query_conversion_radio
        new_mode = -1
        # (元のロジックのまま)
        if option_name == query_options[0]:
            new_mode = 0
        elif option_name == query_options[1]:
            new_mode = 1
        elif option_name == query_options[2]:
            new_mode = 2
        if st.session_state.infer_server_type != 'other' and new_mode < 0:
            if len(query_options) > 3 and option_name == query_options[3]: new_mode = 3
            if len(query_options) > 4 and option_name == query_options[4]: new_mode = 4
        if new_mode < 0: return

        if "query_conversion_mode" not in st.session_state or st.session_state.query_conversion_mode != new_mode:
            st.session_state.query_conversion_mode = new_mode
            old_vals = (st.session_state.config["use_step_back"], st.session_state.config["use_hyde"],
                        st.session_state.config["use_multi"], st.session_state.config["use_deep"])
            st.session_state.config["use_step_back"] = (new_mode == 1)
            st.session_state.config["use_hyde"] = (new_mode == 2)
            st.session_state.config["use_multi"] = (new_mode == 3)
            st.session_state.config["use_deep"] = (new_mode == 4)
            new_vals = (st.session_state.config["use_step_back"], st.session_state.config["use_hyde"],
                        st.session_state.config["use_multi"], st.session_state.config["use_deep"])
            if old_vals != new_vals:
                st.session_state.settings_changed = True
                # 変更されたキーを記録
                changed_rag_keys = {"use_step_back", "use_hyde", "use_multi", "use_deep"}
                st.session_state.just_changed_config_keys.update(changed_rag_keys)
                logger.info(f"クエリ変換モード変更: {option_name} (mode={new_mode})")

    if "query_conversion_mode" not in st.session_state:
        current_mode = 0
        if st.session_state.config["use_step_back"]:
            current_mode = 1
        elif st.session_state.config["use_hyde"]:
            current_mode = 2
        if st.session_state.infer_server_type != "other":
            if st.session_state.config["use_multi"] and len(query_options) > 3:
                current_mode = 3
            elif st.session_state.config["use_deep"] and len(query_options) > 4:
                current_mode = 4
        st.session_state.query_conversion_mode = current_mode

    current_display_mode_idx = st.session_state.query_conversion_mode
    if not (0 <= current_display_mode_idx < len(query_options)):  # 不正なインデックスの場合0にフォールバック
        current_display_mode_idx = 0
        st.session_state.query_conversion_mode = current_display_mode_idx

    if st.session_state.config["use_web"] or st.session_state.rag_mode:
        st.sidebar.markdown("RAG設定")
        with st.expander("クエリ変換設定", expanded=False):
            st.radio("クエリ変換方式", options=query_options, index=current_display_mode_idx,
                     help="RAG利用時のクエリ変換方式を選択します。", disabled=st.session_state.is_sending_message,
                     key="query_conversion_radio", on_change=on_query_conversion_change_callback)

    if st.session_state.config["use_web"]:
        with st.expander("DDGS検索設定", expanded=False):
            st.slider("最大検索件数 (top_k)", min_value=1, max_value=20,
                      value=int(st.session_state.config["web_top_k"]),
                      help="DuckDuckGo検索で取得する最大文書数を設定します",
                      disabled=st.session_state.is_sending_message, key="web_top_k_widget",
                      on_change=update_config_value,
                      args=("web_top_k_widget", "web_top_k", False, True, False, "DDGS検索最大件数"))

    if st.session_state.rag_mode:
        try:
            from tiny_chat.database.database import get_or_create_qdrant_manager
            from tiny_chat.database.qdrant.collection import Collection
            with st.expander("DB検索設定", expanded=False):
                def sync_db_top_k_from_config():
                    new_val = st.session_state.config["db_top_k"]
                    if st.session_state.db_config.top_k != new_val:
                        st.session_state.db_config.top_k = new_val

                def sync_db_score_threshold_from_config():
                    new_val = st.session_state.config["db_score_threshold"]
                    if st.session_state.db_config.score_threshold != new_val:
                        st.session_state.db_config.score_threshold = new_val

                st.slider("最大検索件数 (top_k)", min_value=1, max_value=20,
                          value=st.session_state.db_config.top_k,  # Initial value from the true source (db_config)
                          help="DB検索で取得する最大文書数を設定します",
                          disabled=st.session_state.is_sending_message,
                          key="db_top_k_widget",
                          on_change=update_config_value,
                          args=(
                              "db_top_k_widget",
                              "db_top_k",
                              False,
                              True,
                              False,
                              "DB検索 最大件数 (top_k)",
                              sync_db_top_k_from_config
                          ))


                st.slider("スコアしきい値", min_value=0.0, max_value=5.0,
                          value=st.session_state.db_config.score_threshold,
                          step=0.01,
                          help="DB検索で取得する文書の最小類似度スコア",
                          disabled=st.session_state.is_sending_message,
                          key="db_score_threshold_widget",
                          on_change=update_config_value,
                          args=(
                              "db_score_threshold_widget",
                              "db_score_threshold",
                              True,
                              False,
                              False,
                              "DB検索 スコアしきい値",
                              sync_db_score_threshold_from_config
                          ))

                st.text_area("検索結果指示", value=st.session_state.config["rag_process_prompt"], height=150,
                             help="検索後の情報活用をLLM指示する文字列を入力してください",
                             disabled=st.session_state.is_sending_message, key="rag_process_prompt_widget",
                             on_change=update_config_value,
                             args=("rag_process_prompt_widget", "rag_process_prompt", False, False, False,
                                   "検索結果指示"))
                if not server_mode: st.markdown(
                    '<span style="font-size: 12px;">設定値を保存する場合は「設定保存」してください</span>',
                    unsafe_allow_html=True)

            manager = get_or_create_qdrant_manager(logger)
            available_collections = [c for c in manager.get_collections() if c != Collection.STORED_COLLECTION_NAME]
            if not available_collections:
                available_collections = ["default"]
                collection = Collection(collection_name="default")
                collection.top_k = st.session_state.db_config.top_k
                collection.score_threshold = st.session_state.db_config.score_threshold
                collection.save(qdrant_manager=manager)

            if "selected_collection" not in st.session_state: st.session_state.selected_collection = manager.collection_name
            if "collection_changing" not in st.session_state: st.session_state.collection_changing = False

            st.sidebar.markdown("コレクション選択", help="Qdrantデータベースで利用するコレクションを選択します")
            current_collection_in_manager = manager.collection_name
            idx_collection = 0  # デフォルト
            if current_collection_in_manager in available_collections:
                idx_collection = available_collections.index(current_collection_in_manager)
            elif available_collections:  # マネージャーのコレクションがリストにないが、リスト自体は空でない
                logger.warning(
                    f"現在のマネージャーコレクション '{current_collection_in_manager}' がリストにありません。リストの先頭を表示します。")

            def on_collection_select_change_callback():
                st.session_state.selected_collection = st.session_state.collection_select_key

            st.sidebar.selectbox("コレクション", available_collections, index=idx_collection,
                                 label_visibility="collapsed", disabled=st.session_state.is_sending_message,
                                 key="collection_select_key", on_change=on_collection_select_change_callback)

            if st.session_state.selected_collection != manager.collection_name and not st.session_state.collection_changing:
                st.session_state.collection_changing = True
                logger.info(f"コレクション変更開始: {manager.collection_name} → {st.session_state.selected_collection}")
                manager.set_collection_name(st.session_state.selected_collection)
                st.sidebar.info(f"コレクションを '{st.session_state.selected_collection}' に変更しました。")
                st.rerun()

            if st.session_state.collection_changing and st.session_state.selected_collection == manager.collection_name:
                st.session_state.collection_changing = False
                logger.info(f"コレクション変更完了: {manager.collection_name}")

            doc_count = manager.count_documents()
            st.sidebar.code(f"現在のコレクション: {manager.collection_name}\n登録ドキュメント数: {doc_count}")
        except ImportError:
            st.sidebar.warning("データベース機能の依存関係がインストールされていません。")
        except Exception as e:
            logger.error(f"データベース接続エラー: {str(e)}")
            st.sidebar.error("データベース接続エラーが発生しました。")

    # チャット履歴インポート機能
    uploaded_json = st.file_uploader("チャット履歴をインポート", type=["json"],
                                     help="以前に保存したチャット履歴JSONファイルをインポートします。",
                                     disabled=st.session_state.is_sending_message)
    if uploaded_json is not None:
        content = uploaded_json.getvalue().decode("utf-8")
        if st.button("インポートした履歴を適用", disabled=st.session_state.is_sending_message):
            success = st.session_state.chat_manager.apply_imported_history(content)
            if success:
                logger.info("メッセージ履歴のインポートに成功しました");
                st.success("メッセージ履歴を正常にインポートしました");
                st.rerun()
            else:
                logger.error("メッセージ履歴のインポートに失敗しました: 無効なフォーマット");
                st.error("JSONのインポートに失敗しました: 無効なフォーマットです")