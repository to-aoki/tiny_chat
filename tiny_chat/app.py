import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import logging
import streamlit as st
from config_manager import Config, ModelManager
from file_processor import URIProcessor, FileProcessorFactory
from chat_manager import ChatManager
from logger import get_logger
from llm_utils import get_llm_client
from sidebar import sidebar
from wait_view import spinner
from copy_botton import copy_button
from database import show_database_component, search_documents, get_or_create_qdrant_manager

# https://discuss.streamlit.io/t/message-error-about-torch/90886/9
# RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
import torch
torch.classes.__path__ = []


LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
st.set_page_config(page_title="チャット", layout="wide")

# 設定ファイルのパス
CONFIG_FILE = "chat_app_config.json"

# サポートする拡張子
SUPPORT_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']


def initialize_session_state(config_file_path=CONFIG_FILE, logger=LOGGER):
    if "config" not in st.session_state:
        # 外部設定ファイルから設定を読み込む
        file_config = Config.load(config_file_path)
        logger.info(f"設定ファイルを読み込みました: {config_file_path}")

        # セッション状態に設定オブジェクトを初期化
        st.session_state.config = {
            "server_url": file_config.server_url,
            "api_key": file_config.api_key,
            "selected_model": file_config.selected_model,
            "meta_prompt": file_config.meta_prompt,
            "message_length": file_config.message_length,
            "context_length": file_config.context_length,
            "uri_processing": file_config.uri_processing,
            "is_azure": file_config.is_azure,
            "previous_server_url": file_config.server_url
        }
        logger.info("設定オブジェクトをセッション状態に初期化しました")

    # その他のセッション状態を初期化
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
        logger.info("ChatManagerを初期化しました")

    # メッセージ送信中フラグ
    if "is_sending_message" not in st.session_state:
        st.session_state.is_sending_message = False

    # 処理ステータスメッセージ
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""

    # モデル情報を初期化
    if "available_models" not in st.session_state:
        logger.info("利用可能なモデルを取得しています...")
        models, success = ModelManager.fetch_available_models(
            st.session_state.config["server_url"],
            st.session_state.config["api_key"],
            None,
            st.session_state.config["is_azure"]
        )
        st.session_state.available_models = models
        st.session_state.models_api_success = success
        if success:
            logger.info(f"利用可能なモデル: {', '.join(models)}")
        else:
            logger.warning("モデル取得に失敗しました")

    if "openai_client" not in st.session_state:
        try:
            logger.info("OpenAIクライアントを初期化しています...")
            st.session_state.openai_client = get_llm_client(
                server_url=st.session_state.config["server_url"],
                api_key=st.session_state.config["api_key"],
                is_azure=st.session_state.config["is_azure"]
            )
            logger.info("OpenAIクライアント初期化完了")
        except Exception as e:
            error_msg = f"OpenAI クライアントの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.openai_client = None

    # RAGモードのフラグ
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = False

    # RAG参照ソース情報を保存するリスト
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []

    # データベースタブが選択されたことを記録するフラグ
    if "database_tab_selected" not in st.session_state:
        st.session_state.database_tab_selected = False


# セッション状態の初期化
initialize_session_state(config_file_path=CONFIG_FILE, logger=LOGGER)

# サイドバー
with st.sidebar:
    sidebar(config_file_path=CONFIG_FILE, logger=LOGGER)

# タブの作成
tabs = st.tabs(["💬 チャット", "🔍 データベース"])


def show_chat_component(logger):

    # チャット履歴の表示
    for i, message in enumerate(st.session_state.chat_manager.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                copy_button(message["content"])

    # 添付ファイル一覧を表示
    if st.session_state.chat_manager.attachments:
        with st.expander(f"添付ファイル ({len(st.session_state.chat_manager.attachments)}件)", expanded=True):
            for idx, attachment in enumerate(st.session_state.chat_manager.attachments):
                cols = st.columns([4, 1])
                with cols[0]:
                    filename = attachment['filename']
                    _, ext = os.path.splitext(filename)

                    # ファイルタイプと表示単位
                    file_types = {
                        '.pdf': ("PDF", "ページ"),
                        '.xlsx': ("Excel", "シート"),
                        '.xls': ("Excel", "シート"),
                        '.docx': ("Word", "段落"),
                        '.pptx': ("PowerPoint", "スライド"),
                        '.txt': ("テキスト", ""),
                        '.csv': ("CSV", ""),
                        '.json': ("JSON", ""),
                        '.md': ("Markdown", ""),
                        '.html': ("HTML", ""),
                    }

                    file_type = "ファイル"
                    count_type = ""

                    # ファイルタイプとカウントタイプを取得
                    if ext.lower() in file_types:
                        file_type, count_type = file_types[ext.lower()]

                    # カウント表示
                    count_text = ""
                    if attachment['num_pages'] > 0 and count_type:
                        count_text = f"（{attachment['num_pages']}{count_type}）"

                    st.text(f"{idx + 1}. [{file_type}] {filename} {count_text}")
                    logger.debug(f"添付ファイル表示: {filename} {count_text}")

    with st.container():
        cols = st.columns([3, 2, 3])
        with cols[0]:
            if st.button(
                    "チャットクリア",
                    disabled=st.session_state.is_sending_message,
                    use_container_width=True,
                    key="clear_chat_history_button"):
                st.session_state.chat_manager = ChatManager()
                st.rerun()

        with cols[2]:
            if not st.session_state.chat_manager.messages:
                if st.button(
                        "チャット保存",
                        disabled=st.session_state.is_sending_message,
                        use_container_width=True,
                        key="export_chat_history_button"):
                    st.warning("保存するメッセージ履歴がありません")
            else:
                chat_history = st.session_state.chat_manager.to_json()
                st.download_button(
                    label="チャット保存",
                    data=chat_history,
                    file_name="chat_history.json",
                    mime="application/json",
                    disabled=st.session_state.is_sending_message,  # メッセージ送信中は無効化
                    use_container_width=True,
                    key="export_chat_history_button"
                )
                logger.info("メッセージ履歴のJSONエクスポート機能を提供しました")

        # RAGモードのチェックボックス
        use_rag = st.checkbox("RAG (データベースを利用した回答)", value=st.session_state.rag_mode,
                              key="rag_mode_checkbox")
        # RAGモードが変更された場合、状態を更新
        if use_rag != st.session_state.rag_mode:
            st.session_state.rag_mode = use_rag
            if use_rag:
                # RAGモードが有効になった場合
                st.info("RAGが有効になりました。データベースを準備しています...")
                get_or_create_qdrant_manager(logger)
                st.info("RAGが有効です：メッセージ内容で文書を検索し、関連情報を回答に活用します")
            else:
                st.info("RAGが無効です")
        elif use_rag:
            st.info("RAGが有効です：メッセージ内容で文書を検索し、関連情報を回答に活用します")

    # ユーザー入力
    # 添付ファイルは streamlit v1.43.2 以降
    prompt = st.chat_input(
        "メッセージを入力してください...",
        disabled=st.session_state.is_sending_message,
        accept_file=True,
        file_type=[ext.lstrip(".") for ext in SUPPORT_EXTENSIONS]
    )

    if prompt:
        if prompt and prompt["files"]:
            uploaded_file = prompt["files"][0]  # INFO 先頭1件のみ処理
            filename = uploaded_file.name
            _, file_extension = os.path.splitext(filename)
            processor_class = FileProcessorFactory.get_processor(file_extension)
            if processor_class is None:
                st.error(f"エラー: サポートされていないファイル形式です: {file_extension}")
                logger.error(f"未サポートのファイル形式: {file_extension}")

            else:
                # 各ファイルタイプに応じた処理方法と結果表示の設定
                count_value = 1
                count_type = ""

                # ファイルタイプに応じた処理
                if file_extension.lower() == '.pdf':
                    extracted_text, count_value, error = processor_class.extract_pdf_text(uploaded_file)
                    count_type = "ページ"
                elif file_extension.lower() in ['.xlsx', '.xls']:
                    extracted_text, count_value, error = processor_class.extract_excel_text(uploaded_file)
                    count_type = "シート"
                elif file_extension.lower() == '.pptx':
                    extracted_text, count_value, error = processor_class.extract_pptx_text(uploaded_file)
                    count_type = "スライド"
                elif file_extension.lower() == '.docx':
                    extracted_text, count_value, error = processor_class.extract_word_text(uploaded_file)
                    count_type = "段落"
                else:  # テキスト、HTMLなど
                    extracted_text, error = processor_class.extract_text(uploaded_file)

                # エラー処理
                if error:
                    # Display error message to the user
                    st.error(f"ファイル処理エラー: {error}")
                    logger.error(f"ファイル処理エラー ({filename}): {error}")
                else:
                    # ファイル名の重複チェックと処理
                    existing_files = [a["filename"] for a in st.session_state.chat_manager.attachments]
                    if filename in existing_files:
                        base_name, ext = os.path.splitext(filename)
                        counter = 1
                        new_name = f"{base_name}_{counter}{ext}"
                        while new_name in existing_files:
                            counter += 1
                            new_name = f"{base_name}_{counter}{ext}"
                        filename = new_name
                        logger.info(f"ファイル名重複を検出: {prompt['files'][0].name} → {filename}")

                    # 添付ファイルリストに追加
                    st.session_state.chat_manager.add_attachment(
                        filename=filename,
                        content=extracted_text,
                        num_pages=count_value
                    )
                    st.success(f"ファイル '{filename}' を添付しました")
                    logger.info(f"ファイルを添付: {filename} ({count_value}{count_type})")

        # メッセージ長チェック
        would_exceed, estimated_length, max_length = st.session_state.chat_manager.would_exceed_message_length(
            prompt.text,
            st.session_state.config["message_length"],
            st.session_state.config["context_length"],
            st.session_state.config["meta_prompt"],
            uri_processor=URIProcessor()
        )

        if would_exceed:
            st.error(f"エラー: メッセージ長が上限を超えています（推定: {estimated_length}文字、上限: {max_length}文字）。\n"
                     f"- メッセージを短くするか\n"
                     f"- 添付ファイルを減らすか\n"
                     f"- サイドバー設定のメッセージ長制限を引き上げてください。")
        else:
            # ユーザーメッセージを追加（RAG情報はこの時点ではまだ含まれていない）
            user_message = st.session_state.chat_manager.add_user_message(prompt.text)

            # UIに表示 (UIには元のメッセージだけを表示)
            with st.chat_message("user"):
                st.write(user_message["content"])

            # メッセージ送信中フラグをON
            st.session_state.is_sending_message = True
            st.session_state.status_message = "メッセージを処理中..."
            st.rerun()  # 状態を更新してUIを再描画

    if st.session_state.is_sending_message:
        # 処理待機用描画
        spinner()

    if st.session_state.is_sending_message and st.session_state.chat_manager.messages and \
            st.session_state.chat_manager.messages[-1]["role"] != "assistant":
        try:
            # 最後のユーザーメッセージを取得
            last_user_message = st.session_state.chat_manager.get_latest_user_message()
            prompt_content = last_user_message["content"].split("\n\n[添付ファイル:")[0]  # 添付ファイル情報を削除

            # 処理ステータスを更新
            st.session_state.status_message = "メッセージを処理中..."

            # URIプロセッサ
            detects_urls = []
            if st.session_state.config["uri_processing"]:
                uri_processor = URIProcessor()
                detects_urls = uri_processor.detect_uri(prompt_content)

            # 処理ステータスを更新
            st.session_state.status_message = "LLMにプロンプトを入力中..."

            # 拡張プロンプトの取得
            enhanced_prompt = None
            if st.session_state.chat_manager.attachments or (
                    st.session_state.config["uri_processing"] and len(detects_urls) > 0):
                # 処理ステータスを更新
                if st.session_state.chat_manager.attachments:
                    st.session_state.status_message = "添付ファイルの内容を解析中..."
                elif len(detects_urls) > 0:
                    st.session_state.status_message = "URLからコンテンツを取得中..."

                # 拡張プロンプトを生成
                enhanced_prompt = st.session_state.chat_manager.get_enhanced_prompt(
                    prompt_content,
                    max_length=st.session_state.config["context_length"],
                    uri_processor=uri_processor if st.session_state.config["uri_processing"] else None
                )

            # RAGモードが有効な場合、検索を実行
            if st.session_state.rag_mode:
                st.session_state.status_message = "関連文書を検索中..."
                # 最新のユーザーメッセージで検索（共通関数を使用）
                search_results = search_documents(prompt_content, top_k=5, logger=logger)

                if search_results:
                    # 検索結果を整形
                    search_context = "以下は検索システムから取得した関連情報です:\n\n"

                    # 参照情報をリセット（後でアシスタント出力に表示するため）
                    st.session_state.rag_sources = []

                    for i, result in enumerate(search_results):
                        filename = result.payload.get('filename', '文書')
                        source = result.payload.get('source', '')
                        text = result.payload.get('text', '')[:st.session_state.config["context_length"]]  # テキスト内容を取得

                        # 参照情報を保存
                        source_info = {
                            "index": i + 1,
                            "filename": filename,
                            "source": source,
                            "text": text  # テキスト内容も保存
                        }
                        st.session_state.rag_sources.append(source_info)

                        search_context += f"[{i + 1}] {filename}:\n"
                        search_context += f"{text}\n\n"

                    # 検索結果を含めた拡張プロンプトを作成
                    if enhanced_prompt:
                        enhanced_prompt += f"\n\n{search_context}"
                    else:
                        enhanced_prompt = prompt_content + f"\n\n{search_context}"

                    st.session_state.status_message = "検索結果を追加しました。LLMにプロンプトを入力中..."

            # 拡張プロンプトがあれば更新
            if enhanced_prompt:
                st.session_state.chat_manager.update_enhanced_prompt(enhanced_prompt)

            messages_for_api = st.session_state.chat_manager.prepare_messages_for_api(
                st.session_state.config["meta_prompt"])

            if not messages_for_api:
                # システムメッセージがあれば追加
                if st.session_state.config["meta_prompt"]:
                    messages_for_api.append({"role": "system", "content": st.session_state.config["meta_prompt"]})

                # 通常メッセージ
                content_to_send = prompt_content

                # RAGが有効で検索結果がある場合は、検索結果を含める
                if st.session_state.rag_mode and "rag_sources" in st.session_state and st.session_state.rag_sources:
                    search_context = "\n\n以下は検索システムから取得した関連情報です:\n\n"
                    for source in st.session_state.rag_sources:
                        search_context += f"[{source['index']}] {source['filename']}:\n"
                        if 'text' in source:
                            search_context += f"{source['text']}\n\n"
                    content_to_send += search_context

                messages_for_api.append({"role": "user", "content": content_to_send})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    # クライアントインスタンスが存在しない場合は初期化
                    if "openai_client" not in st.session_state or st.session_state.openai_client is None:
                        st.session_state.openai_client = get_llm_client(
                            server_url=st.session_state.config["server_url"],
                            api_key=st.session_state.config["api_key"],
                            is_azure=st.session_state.config["is_azure"]
                        )

                    # 既存のクライアントインスタンスを使用
                    client = st.session_state.openai_client

                    # ストリーミングモードでリクエスト
                    response = client.chat.completions.create(
                        model=st.session_state.config["selected_model"],
                        messages=messages_for_api,
                        stream=True
                    )

                    # ストリーミング応答をリアルタイムで処理
                    full_response = ""
                    message_placeholder.markdown("応答を生成中..._")

                    for chunk in response:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                full_response += delta.content
                                message_placeholder.markdown(full_response)

                    # RAGモードで検索結果がある場合、参照情報を追加
                    final_response = full_response
                    if st.session_state.rag_mode and "rag_sources" in st.session_state and st.session_state.rag_sources:
                        # 参照情報を表示用に整形
                        sources_md = "\n\n---\n**参照情報:**\n"
                        unique_sources = set()
                        refer = 0
                        for source in st.session_state.rag_sources:
                            # ソースパスを表示用テキストとリンク先URLに分ける
                            source_path = source["source"]
                            filename = source["filename"]

                            if not source_path:
                                continue

                            # /tmp/で始まるパスはスキップ（参照情報として表示しない）
                            if source_path.startswith('/tmp/'):
                                continue

                            # 重複チェック - 既に同じソースが存在する場合はスキップ
                            if source_path in unique_sources:
                                continue

                            # URLの処理
                            # - httpで始まる場合は、そのままURLとして使用
                            # - それ以外の場合は、file://プロトコルを付加（絶対パスの処理に注意）
                            if source_path.startswith('http'):
                                source_url = source_path
                            else:
                                # file://プロトコルの正しい形式：file:///path/to/file（スラッシュ3つ）
                                # 絶対パスの場合は先頭の / を維持するために注意
                                if source_path.startswith('/'):
                                    source_url = f"file://{source_path}"  # '/path/to/file' → 'file:///path/to/file'
                                else:
                                    source_url = f"file:///{source_path}"  # 相対パスの場合

                            # インデックス付きの参照リンクを追加
                            sources_md += f"- [{refer+1}] [{filename}]({source_url})\n"
                            refer += 1

                        # マークダウンで最終出力を表示（出力＋参照情報）
                        if refer > 0:
                            final_response = full_response + sources_md
                        message_placeholder.markdown(final_response)
                    else:
                        # 通常の出力
                        message_placeholder.markdown(full_response)

                    # 応答をメッセージ履歴に追加（参照情報を含む）
                    st.session_state.chat_manager.add_assistant_message(final_response)

                    # 送信後に添付ファイルを削除
                    st.session_state.chat_manager.clear_attachments()

                    # rag_sourcesをクリア
                    if "rag_sources" in st.session_state:
                        st.session_state.rag_sources = []

                except Exception as e:
                    error_message = f"APIエラー: {str(e)}"
                    message_placeholder.error(error_message)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

        st.session_state.is_sending_message = False
        st.session_state.status_message = "処理完了"
        st.rerun()


# チャット機能タブ
with tabs[0]:
    show_chat_component(logger=LOGGER)


# データベース機能タブ
with tabs[1]:
    # データベースタブが選択されたことを記録
    st.session_state.database_tab_selected = True
    # データベース機能の表示
    show_database_component(logger=LOGGER, extensions=SUPPORT_EXTENSIONS)
