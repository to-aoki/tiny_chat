import os
import logging
import tempfile
import functools
import webbrowser
import urllib.parse

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
import streamlit as st

from tiny_chat.chat.chat_config import ChatConfig, ModelManager
from tiny_chat.chat.chat_manager import ChatManager
from tiny_chat.utils.file_processor import URIProcessor, FileProcessorFactory
from tiny_chat.utils.logger import get_logger
from tiny_chat.utils.llm_utils import get_llm_client
from tiny_chat.chat.sidebar import sidebar
from tiny_chat.chat.copy_botton import copy_button

# https://discuss.streamlit.io/t/message-error-about-torch/90886/9
# RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
import torch
torch.classes.__path__ = []


# 設定ファイルのパス
CONFIG_FILE = "chat_app_config.json"

# サポートする拡張子
SUPPORT_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']

# ファイルタイプと表示単位のキャッシュ
FILE_TYPES = {
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


def initialize_session_state(config_file_path=CONFIG_FILE, logger=None, session_only_mode=False):
    if "config" not in st.session_state:
        # 外部設定ファイルから設定を読み込む
        file_config = ChatConfig.load(config_file_path)
        if logger is not None:
            logger.info(f"設定ファイルを読み込みました: {config_file_path}")

        # サーバーモードが指定されていれば上書き
        file_config.session_only_mode = session_only_mode
        
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
            "session_only_mode": file_config.session_only_mode,
        }

    # その他のセッション状態を初期化
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    # メッセージ送信中フラグ
    if "is_sending_message" not in st.session_state:
        st.session_state.is_sending_message = False

    # 処理ステータスメッセージ
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""

    # モデル情報を初期化
    if "available_models" not in st.session_state:
        models, success = ModelManager.fetch_available_models(
            st.session_state.config["server_url"],
            st.session_state.config["api_key"],
            None,
            st.session_state.config["is_azure"]
        )
        st.session_state.available_models = models
        st.session_state.models_api_success = success
        if not success:
            if logger is not None:
                logger.warning("モデル取得に失敗しました")

    if "openai_client" not in st.session_state:
        try:
            st.session_state.openai_client = get_llm_client(
                server_url=st.session_state.config["server_url"],
                api_key=st.session_state.config["api_key"],
                is_azure=st.session_state.config["is_azure"]
            )
        except Exception as e:
            if logger is not None:
                error_msg = f"OpenAI クライアントの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.openai_client = None

    # RAGモードのフラグ
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = False
        
    # RAGモードが一度でも有効になったことがあるかを追跡するフラグ
    if "rag_mode_ever_enabled" not in st.session_state:
        st.session_state.rag_mode_ever_enabled = False

    # RAG参照ソース情報を保存するリスト
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []

    # 現在の回答で参照されたファイル情報
    if "reference_files" not in st.session_state:
        st.session_state.reference_files = []

    # 初回チャットメッセージ送信フラグ
    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False


# チャットをクリアするコールバック関数
def clear_chat():
    st.session_state.chat_manager = ChatManager()
    # 参照ファイル情報もクリア
    st.session_state.reference_files = []


# RAGモード切り替え用の関数
def toggle_rag_mode(logger):
    # チェックボックスの状態を取得
    current_state = st.session_state.rag_mode_checkbox
    # セッション状態を更新
    st.session_state.rag_mode = current_state
    
    if current_state:
        # RAGモードが有効になった場合
        try:
            # RAGモードが一度でも有効になったことを記録（この値は保持される）
            st.session_state.rag_mode_ever_enabled = True

            # DBに接続
            from tiny_chat.database.database import get_or_create_qdrant_manager
            get_or_create_qdrant_manager(logger)

        except Exception as e:
            st.error(f"RAGモード有効化中にエラーが発生しました: {str(e)}")
            # エラーが発生したらRAGモードを無効化（ただしever_enabledは維持）
            st.session_state.rag_mode = False
            st.session_state.rag_mode_checkbox = False
    else:
        # RAGモードが無効になった場合、参照情報のみクリア（データベース表示は維持）
        st.session_state.rag_sources = []
        st.session_state.reference_files = []


# キャッシュ可能な検索関数 - RAGモード専用
@functools.lru_cache(maxsize=32)
def cached_search_documents(prompt_content, logger):
    if not st.session_state.rag_mode:
        return []
    
    # RAGモードが有効な場合のみ検索関数をインポートして実行
    from tiny_chat.database.database import get_or_create_qdrant_manager
    from tiny_chat.database.components.search import search_documents
    qdrant_manager = get_or_create_qdrant_manager(logger=logger)
    return search_documents(prompt_content, qdrant_manager=qdrant_manager)


def show_chat_component(logger):
    # チャット履歴の表示
    messages = st.session_state.chat_manager.messages
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                copy_button(message["content"])
                
                # 参照ファイルがある場合、最後の応答に対してのみボタンを表示する
                if i == len(messages) - 1 and st.session_state.reference_files:
                    with st.container():
                        st.write("参照情報を開く:")
                        for idx, file_info in enumerate(st.session_state.reference_files):
                            if not file_info["path"].startswith(('http://', 'https://')):
                                if st.button(
                                        f"[{file_info['index']}] {file_info['path']}", key=f"open_ref_{i}_{idx}"):
                                    try:
                                        webbrowser.open(file_info["path"])
                                    except Exception as e:
                                        st.error(f"ファイルを開けませんでした: {str(e)}")
                            else:
                                st.markdown(
                                    f"[\\[{file_info['index']}\\] {file_info['path']}]"
                                    f"({urllib.parse.quote(file_info['path'], safe=':/')})")

    # 添付ファイル一覧を表示
    if st.session_state.chat_manager.attachments:
        with st.expander(f"添付ファイル ({len(st.session_state.chat_manager.attachments)}件)", expanded=True):
            attachments_grid = []
            # 3カラムのグリッドに表示するためのデータ準備
            for idx, attachment in enumerate(st.session_state.chat_manager.attachments):
                filename = attachment['filename']
                _, ext = os.path.splitext(filename)
                ext = ext.lower()

                # ファイルタイプと表示単位を取得
                file_type, count_type = FILE_TYPES.get(ext, ("ファイル", ""))

                # カウント表示
                count_text = ""
                if attachment['num_pages'] > 0 and count_type:
                    count_text = f"（{attachment['num_pages']}{count_type}）"

                attachments_grid.append({
                    "index": idx + 1,
                    "file_type": file_type,
                    "filename": filename,
                    "count_text": count_text
                })
            
            # 3カラムで表示
            cols = st.columns(3)
            for idx, attachment_info in enumerate(attachments_grid):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.text(
                        f"{attachment_info['index']}. [{attachment_info['file_type']}] {attachment_info['filename']} "
                        f"{attachment_info['count_text']}")

    with st.container():
        cols = st.columns([3, 2, 3])
        with cols[0]:
            st.button(
                "チャットクリア",
                disabled=st.session_state.is_sending_message,
                use_container_width=True,
                key="clear_chat_history_button",
                on_click=clear_chat
            )

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

        # RAGモードのチェックボックス
        st.checkbox("RAG (データベースを利用した回答)", 
                    value=st.session_state.rag_mode,
                    key="rag_mode_checkbox", 
                    on_change=toggle_rag_mode,
                    args=(logger,))
        
        # 現在のRAG状態に基づいてメッセージを表示
        if st.session_state.rag_mode:
            st.info("RAGが有効です：メッセージ内容で文書を検索し、関連情報を回答に活用します")
        else:
            st.info("RAGが無効です")

    # ユーザー入力
    # 添付ファイルは streamlit v1.43.2 以降
    prompt = st.chat_input(
        "メッセージを入力してください...",
        disabled=st.session_state.is_sending_message,
        accept_file=True,
        file_type=[ext.lstrip(".") for ext in SUPPORT_EXTENSIONS]
    )

    # ファイル処理関数
    def process_uploaded_file(uploaded_file):
        filename = uploaded_file.name
        _, file_extension = os.path.splitext(filename)
        file_extension = file_extension.lower()  # 小文字に変換
        
        processor_class = FileProcessorFactory.get_processor(file_extension)
        if processor_class is None:
            st.error(f"エラー: サポートされていないファイル形式です: {file_extension}")
            logger.error(f"未サポートのファイル形式: {file_extension}")
            return False
        
        # 各ファイルタイプに応じた処理方法と結果表示の設定
        count_value = 1
        count_type = ""

        # ファイルタイプに応じた処理
        try:
            if file_extension == '.pdf':
                extracted_text, count_value, error = processor_class.extract_pdf_text(uploaded_file)
                count_type = "ページ"
            elif file_extension in ['.xlsx', '.xls']:
                extracted_text, count_value, error = processor_class.extract_excel_text(uploaded_file)
                count_type = "シート"
            elif file_extension == '.pptx':
                extracted_text, count_value, error = processor_class.extract_pptx_text(uploaded_file)
                count_type = "スライド"
            elif file_extension == '.docx':
                extracted_text, count_value, error = processor_class.extract_word_text(uploaded_file)
                count_type = "段落"
            else:  # テキスト、HTMLなど
                extracted_text, error = processor_class.extract_text(uploaded_file)
        except Exception as e:
            st.error(f"ファイル処理エラー: {str(e)}")
            logger.error(f"ファイル処理中に例外が発生 ({filename}): {str(e)}")
            return False

        # エラー処理
        if error:
            # Display error message to the user
            st.error(f"ファイル処理エラー: {error}")
            logger.error(f"ファイル処理エラー ({filename}): {error}")
            return False
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

            # 添付ファイルリストに追加
            st.session_state.chat_manager.add_attachment(
                filename=filename,
                content=extracted_text,
                num_pages=count_value
            )
            st.success(f"ファイル '{filename}' を添付しました")
            logger.info(f"ファイルを添付: {filename} ({count_value}{count_type})")
            return True
    
    # メッセージ送信処理関数
    def process_and_send_message():
        try:
            # 最後のユーザーメッセージを取得
            last_user_message = st.session_state.chat_manager.get_latest_user_message()
            prompt_content = last_user_message["content"].split("\n\n[添付ファイル:")[0]  # 添付ファイル情報を削除

            # URIプロセッサ
            detects_urls = []
            uri_processor = None
            if st.session_state.config["uri_processing"]:
                uri_processor = URIProcessor()
                detects_urls = uri_processor.detect_uri(prompt_content)

            # 拡張プロンプトの取得
            enhanced_prompt = None
            if st.session_state.chat_manager.attachments or (
                    st.session_state.config["uri_processing"] and detects_urls):
                # 拡張プロンプトを生成
                enhanced_prompt = st.session_state.chat_manager.get_enhanced_prompt(
                    prompt_content,
                    max_length=st.session_state.config["context_length"],
                    uri_processor=uri_processor
                )

            # RAGモードが有効な場合のみ、DBに接続して検索を実行
            if st.session_state.rag_mode:
                try:
                    # RAGモードが有効な場合のみQdrantマネージャを取得・初期化
                    from tiny_chat.database.database import get_or_create_qdrant_manager
                    get_or_create_qdrant_manager(logger)
                    
                    # 最新のユーザーメッセージで検索
                    search_results = cached_search_documents(prompt_content, logger)
                    
                    # 検索結果の状態をログに記録
                    logger.info(f"RAG検索結果: {len(search_results) if search_results else 0}件")

                    if search_results:
                        # 検索結果を整形
                        search_context = "関連文書が有効な場合は回答に役立ててください。\n関連文書:\n"

                        # 参照情報をリセット
                        st.session_state.rag_sources = []
                        
                        # 既存のパスを追跡して重複を避ける
                        exist_path = set()

                        for i, result in enumerate(search_results):
                            source = result.payload.get('source', '')

                            # 重複チェック
                            if source in exist_path:
                                continue
                                
                            exist_path.add(source)
                            
                            # テキスト内容を取得（長さ制限あり）
                            text = result.payload.get('text', '')[:st.session_state.config["context_length"]]  

                            # 参照情報を保存
                            source_info = {
                                "index": i + 1,
                                "source": source
                            }
                            st.session_state.rag_sources.append(source_info)

                            search_context += f"[{i + 1}] {source}:\n{text}\n\n"

                        # 検索結果を含めた拡張プロンプトを作成
                        if enhanced_prompt:
                            enhanced_prompt += f"\n\n{search_context}"
                        else:
                            enhanced_prompt = prompt_content + f"\n\n{search_context}"
                except Exception as e:
                    logger.error(f"RAG検索処理中にエラー: {str(e)}")
                    # エラーが発生しても続行、ただしRAG検索なしで

            # 拡張プロンプトがあれば更新
            if enhanced_prompt:
               st.session_state.chat_manager.update_enhanced_prompt(enhanced_prompt)

            # APIに送信するメッセージの準備
            messages_for_api = st.session_state.chat_manager.prepare_messages_for_api(
                st.session_state.config["meta_prompt"])

            if not messages_for_api:
                # システムメッセージがあれば追加
                if st.session_state.config["meta_prompt"]:
                    messages_for_api.append({"role": "system", "content": st.session_state.config["meta_prompt"]})

                if enhanced_prompt:
                    messages_for_api.append({"role": "user", "content": enhanced_prompt})
                else:
                    messages_for_api.append({"role": "user", "content": prompt_content})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("応答を生成中..._")

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

                    response = client.chat.completions.create(
                        model=st.session_state.config["selected_model"],
                        messages=messages_for_api,
                        stream=True
                    )

                    # ストリーミング応答をリアルタイムで処理
                    full_response = ""

                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # 過度な再描画を防ぐため、10文字ごとに更新
                            if len(full_response) % 10 == 0:
                                message_placeholder.markdown(full_response)

                    # 最終応答を表示
                    message_placeholder.markdown(full_response)

                    # RAGモードで検索結果がある場合のみ、参照情報を追加
                    if st.session_state.rag_mode and st.session_state.rag_sources:
                        # 参照ファイル情報を保存するが、マークダウンには表示しない
                        reference_files = []
                        refer = 0
                        exist_path = set()
                        
                        for source in st.session_state.rag_sources:
                            source_path = source["source"]

                            if source_path in exist_path:
                                continue

                            if not source_path or source_path.startswith(tempfile.gettempdir()):
                                continue

                            # URLの場合とローカルファイルの場合、両方とも参照ボタンとして表示できるようにする
                            reference_files.append({
                                "index": refer+1,
                                "path": source_path
                            })
                            refer += 1
                            exist_path.add(source_path)
                        
                        # セッション状態に参照ファイル情報を保存
                        st.session_state.reference_files = reference_files

                    # 応答をメッセージ履歴に追加
                    st.session_state.chat_manager.add_assistant_message(full_response)

                    # 送信後に添付ファイルを削除
                    st.session_state.chat_manager.clear_attachments()

                    # rag_sourcesをクリア
                    st.session_state.rag_sources = []

                except Exception as e:
                    error_message = f"APIエラー: {str(e)}"
                    logger.error(f"APIエラー: {str(e)}")
                    message_placeholder.error(error_message)

        except Exception as e:
            logger.error(f"エラーが発生しました: {str(e)}")
            st.error(f"エラーが発生しました: {str(e)}")

    # ユーザーがメッセージを送信した場合の処理
    if prompt:
        with st.spinner("応答中..."):
            # ファイルアップロードの処理
            file_processed = False
            if prompt["files"]:
                uploaded_file = prompt["files"][0]  # 先頭1件のみ処理
                file_processed = process_uploaded_file(uploaded_file)
                # ファイル処理後、テキストがなければ再描画して終了
                if not file_processed or not prompt.text:
                    st.rerun()
                    return

            # テキストメッセージ処理（ファイルアップロードがあってもなくても続行）
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
                st.session_state.initial_message_sent = True

                # 処理を実行
                process_and_send_message()

                # 処理終了フラグを設定
                st.session_state.is_sending_message = False
                st.session_state.status_message = "処理完了"
                st.session_state.initial_message_sent = False
                st.rerun()  # 再描画


def run_chat_app(server_mode=False):
    LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
    
    # サーバーモードが有効な場合はログに記録
    if server_mode:
        LOGGER.info("サーバーモードで起動しました。設定はファイルに保存されません。")

    # セッション状態の初期化
    initialize_session_state(config_file_path=CONFIG_FILE, logger=LOGGER, session_only_mode=server_mode)

    # サイドバー
    with st.sidebar:
        sidebar(config_file_path=CONFIG_FILE, logger=LOGGER)

    tab_items = ["💬 チャット", "🛢️ データベース"]
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tab_items[0]

    st.radio(
        "トップナビゲーション",
        tab_items,
        key='active_tab',
        horizontal=True,
        label_visibility="collapsed"
    )

    # チャット機能タブ
    if st.session_state.active_tab == tab_items[0]:
        show_chat_component(logger=LOGGER)

    # データベース機能タブ
    if st.session_state.active_tab == tab_items[1]:
        # データベース機能の表示
        if st.session_state.rag_mode_ever_enabled:
            try:
                from tiny_chat.database.database import get_or_create_qdrant_manager, show_database_component

                # チャットアプリからの呼び出しなので、ページ設定を重複して行わない（set_config=False）
                get_or_create_qdrant_manager(LOGGER)
                show_database_component(logger=LOGGER, extensions=SUPPORT_EXTENSIONS)

            except Exception as e:
                LOGGER.error(f"データベース接続エラー: {str(e)}")
                st.error(f"データベース接続中にエラーが発生しました: {str(e)}")
        else:
            st.warning("データベース機能を有効にするには以下のボタンをクリックしてください。")

            def enable_database():
                st.session_state.rag_mode_ever_enabled = True

            st.button("データベースを有効にする", on_click=enable_database, use_container_width=True)
