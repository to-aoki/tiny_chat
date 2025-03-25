import os
import logging
import streamlit as st
import pandas as pd
import tempfile
from typing import List, Dict, Any, Tuple

from config_manager import Config, ModelManager
from file_processor import URIProcessor, FileProcessorFactory
from chat_manager import ChatManager
from logger import get_logger
from llm_utils import get_llm_client
from sidebar import sidebar
from wait_view import spinner
from copy_botton import copy_button
from qdrant_manager import QdrantManager

# https://discuss.streamlit.io/t/message-error-about-torch/90886/9
# RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
import torch
torch.classes.__path__ = []

LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
st.set_page_config(page_title="チャット", layout="wide")

# 設定ファイルのパス
CONFIG_FILE = "chat_app_config.json"


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
    
    # 検索機能のためのQdrantManagerを初期化
    if 'manager' not in st.session_state:
        st.session_state.manager = QdrantManager(
            collection_name="default",
            path="./qdrant_data"
        )
        logger.info("QdrantManagerを初期化しました")
        
    # RAGモードのフラグ
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = False


# 検索機能のための関数
def process_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    ファイルを処理し、テキストとメタデータを抽出します
    
    Args:
        file_path: 処理するファイルのパス
        
    Returns:
        (extracted_text, metadata): 抽出されたテキストとメタデータの辞書
    """
    # ファイル拡張子を取得
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # ファイルプロセッサを取得
    processor = FileProcessorFactory.get_processor(file_ext)
    
    if not processor:
        st.warning(f"非対応の形式です: {file_ext}")
        return None, {}
    
    # ファイルを読み込む
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    # メタデータの初期化
    metadata = {
        "source": file_path,
        "filename": os.path.basename(file_path),
        "file_type": file_ext[1:],  # 拡張子の.を除去
    }
    
    # ファイルタイプに応じた処理
    if file_ext == '.pdf':
        text, page_count, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"PDFの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["page_count"] = page_count
        
    elif file_ext in ['.xlsx', '.xls']:
        text, sheet_count, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"Excelの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["sheet_count"] = sheet_count
        
    elif file_ext == '.docx':
        text, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"Wordの処理中にエラーが発生しました: {error}")
            return None, {}
            
    elif file_ext == '.pptx':
        text, slide_count, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"PowerPointの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["slide_count"] = slide_count
        
    elif file_ext in ['.txt', '.csv', '.json', '.md']:
        text, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"テキストファイルの処理中にエラーが発生しました: {error}")
            return None, {}
            
    elif file_ext in ['.html', '.htm']:
        text, message = processor.extract_text_from_bytes(file_bytes)
        if not text:
            st.warning(f"HTMLの処理中にエラーが発生しました: {message}")
            return None, {}
    
    else:
        st.warning(f"対応していないファイル形式です: {file_ext}")
        return None, {}
    
    # ファイルサイズを追加
    metadata["file_size"] = len(file_bytes)
    
    return text, metadata


def process_directory(directory_path: str, extensions: List[str] = None) -> List[Tuple[str, Dict]]:
    """
    ディレクトリ内のファイルを処理します
    
    Args:
        directory_path: 処理するディレクトリのパス
        extensions: 処理対象のファイル拡張子リスト (None の場合はすべてのサポートされる形式)
        
    Returns:
        [(text, metadata), ...]: 抽出されたテキストとメタデータのリスト
    """
    results = []
    
    # サポートされるすべての拡張子を取得
    if extensions is None:
        extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']
    
    # ファイルを検索
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in extensions:
                text, metadata = process_file(file_path)
                if text:
                    # 相対パスをメタデータに追加
                    rel_path = os.path.relpath(file_path, directory_path)
                    metadata["rel_path"] = rel_path
                    
                    results.append((text, metadata))
    
    return results


def add_files_to_qdrant(texts: List[str], metadatas: List[Dict]) -> List[str]:
    """
    テキストとメタデータをQdrantに追加します
    
    Args:
        texts: テキストのリスト
        metadatas: メタデータのリスト
        
    Returns:
        added_ids: 追加されたドキュメントのIDリスト
    """
    added_ids = st.session_state.manager.add_documents(texts, metadatas)
    return added_ids


def search_documents(query: str, top_k: int = 10, filter_params: Dict = None) -> List:
    """
    ドキュメントを検索します
    
    Args:
        query: 検索クエリ
        top_k: 返す結果の数
        filter_params: 検索フィルタ
        
    Returns:
        results: 検索結果のリスト
    """
    results = st.session_state.manager.query_points(query, top_k=top_k, filter_params=filter_params)
    return results


# セッション状態の初期化
initialize_session_state(config_file_path=CONFIG_FILE, logger=LOGGER)

# サイドバー
sidebar(config_file_path=CONFIG_FILE, logger=LOGGER)

# 検索用サイドバー設定
st.sidebar.title("検索")

# コレクション名の選択（サイドバーに表示）
# Qdrantデータディレクトリからコレクション一覧を取得
collections_path = os.path.join("./qdrant_data", "collection")
available_collections = []
if os.path.exists(collections_path):
    available_collections = [d for d in os.listdir(collections_path) if os.path.isdir(os.path.join(collections_path, d))]

# コレクションがなければデフォルトのものを表示
if not available_collections:
    available_collections = [st.session_state.manager.collection_name]


# コレクション選択UIをサイドバーに配置
st.sidebar.markdown('<p class="small-font">コレクション選択</p>', unsafe_allow_html=True)
search_collection = st.sidebar.selectbox(
    "コレクション",  # 空のラベルから有効なラベルに変更
    available_collections,
    index=available_collections.index(st.session_state.manager.collection_name) if st.session_state.manager.collection_name in available_collections else 0,
    label_visibility="collapsed"  # ラベルを視覚的に非表示にする
)

# 選択されたコレクションに切り替え
if search_collection != st.session_state.manager.collection_name:
    st.session_state.manager.get_collection(search_collection)

# サイドバーに現在のコレクション情報を表示
doc_count = st.session_state.manager.count_documents()
st.sidebar.markdown('<p class="small-font">現在のコレクション</p>', unsafe_allow_html=True)
st.sidebar.code(st.session_state.manager.collection_name)
st.sidebar.markdown('<p class="small-font">登録ドキュメント数</p>', unsafe_allow_html=True)
st.sidebar.code(f"{doc_count}")

# タブの作成
tabs = st.tabs(["💬 チャット", "🔍 データベース"])

# チャット機能タブ
with tabs[0]:
    # RAGモードのチェックボックス（検索を使用するか）
    if st.checkbox("RAG (検索システムを利用した回答)", value=st.session_state.rag_mode, key="rag_mode_checkbox"):
        st.session_state.rag_mode = True
        st.info("RAGモードがオンです：メッセージ内容で文書を検索し、関連情報を回答に活用します")
    else:
        st.session_state.rag_mode = False
    
    # 処理中ステータス表示エリア
    status_area = st.empty()

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
                        '.docx': ("Word", ""),
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
                    LOGGER.debug(f"添付ファイル表示: {filename} {count_text}")

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
            # チャット履歴がある場合はダウンロードボタンを表示、なければ通常ボタンを表示
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
                LOGGER.info("メッセージ履歴のJSONエクスポート機能を提供しました")

    # ユーザー入力
    # 添付ファイルは streamlit v1.43.2 以降
    prompt = st.chat_input(
        "メッセージを入力してください...",
        disabled=st.session_state.is_sending_message,
        accept_file=True,
        file_type=["pdf", "xlsx", "xls", "docx", "pptx", "txt", "csv", "json", "md", "html"],
    )

    if prompt:

        if prompt and prompt["files"]:
            uploaded_file = prompt["files"][0]  # 先頭１件のみ処理
            filename = uploaded_file.name
            _, file_extension = os.path.splitext(filename)
            processor_class = FileProcessorFactory.get_processor(file_extension)
            if processor_class is None:
                # Display error for unsupported file type
                st.error(f"エラー: サポートされていないファイル形式です: {file_extension}")
                LOGGER.error(f"未サポートのファイル形式: {file_extension}")

            else:
                # 各ファイルタイプに応じた処理方法と結果表示の設定
                extracted_text = None
                error = None
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
                    extracted_text, error = processor_class.extract_word_text(uploaded_file)
                else:  # テキスト、HTMLなど
                    extracted_text, error = processor_class.extract_text(uploaded_file)

                # エラー処理
                if error:
                    # Display error message to the user
                    st.error(f"ファイル処理エラー: {error}")
                    LOGGER.error(f"ファイル処理エラー ({filename}): {error}")
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
                        LOGGER.info(f"ファイル名重複を検出: {prompt['files'][0].name} → {filename}")

                    # 添付ファイルリストに追加
                    st.session_state.chat_manager.add_attachment(
                        filename=filename,
                        content=extracted_text,
                        num_pages=count_value
                    )
                    st.success(f"ファイル '{filename}' を添付しました")
                    LOGGER.info(f"ファイルを添付: {filename} ({count_value}{count_type})")

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
            # ユーザーメッセージを追加
            user_message = st.session_state.chat_manager.add_user_message(prompt.text)

            # UIに表示
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

            # この部分は不要になりました (既に上で処理されています)

            # 処理ステータスを更新
            st.session_state.status_message = "LLMにプロンプトを入力中..."
            
            # 拡張プロンプトの取得
            enhanced_prompt = None
            if st.session_state.chat_manager.attachments or (st.session_state.config["uri_processing"] and len(detects_urls) > 0):
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
                # 最新のユーザーメッセージで検索
                search_results = search_documents(prompt_content, top_k=5)
                
                if search_results:
                    # 検索結果を整形
                    search_context = "以下は検索システムから取得した関連情報です:\n\n"
                    for i, result in enumerate(search_results):
                        search_context += f"[{i+1}] {result.payload.get('filename', '文書')}:\n"
                        search_context += f"{result.payload.get('text', '')[:1000]}\n\n"
                    
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

                messages_for_api.append({"role": "user", "content": prompt_content})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

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

                    message_placeholder.markdown(full_response)

                    # 応答をメッセージ履歴に追加
                    st.session_state.chat_manager.add_assistant_message(full_response)

                    # 送信後に添付ファイルを削除
                    st.session_state.chat_manager.clear_attachments()

                except Exception as e:
                    error_message = f"APIエラー: {str(e)}"
                    message_placeholder.error(error_message)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

        st.session_state.is_sending_message = False
        st.session_state.status_message = "処理完了"
        st.rerun()

# 検索機能タブ
with tabs[1]:
    
    # 検索と文書登録のタブを作成
    search_tabs = st.tabs(["🔍 検索", "📁 文書登録"])
    
    # 検索タブ
    with search_tabs[0]:
        # 検索フィールド
        query = st.text_input("検索キーワード", "")
        
        # 詳細設定のエクスパンダー
        with st.expander("詳細設定", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider("表示件数", min_value=1, max_value=50, value=10)
            
            with col2:
                # 使用可能なソースを取得
                sources = st.session_state.manager.get_sources()
                selected_sources = st.multiselect("ソースでフィルタ", options=sources)
        
        # 検索ボタン
        search_pressed = st.button("検索", key="search_button", type="primary")
        
        # 検索実行
        if search_pressed and query:
            # フィルターの作成
            filter_params = {}
            if selected_sources:
                filter_params["source"] = selected_sources
            
            with st.spinner("検索中..."):
                results = search_documents(query, top_k=top_k, filter_params=filter_params)
            
            # 結果の表示
            if results:
                st.success(f"{len(results)}件の結果が見つかりました")
                
                for i, result in enumerate(results):
                    score = result.score
                    text = result.payload.get("text", "")
                    
                    # メタデータを表示用に整形
                    metadata = {k: v for k, v in result.payload.items() if k != "text"}
                    
                    # 結果表示
                    with st.expander(f"#{i+1}: {metadata.get('filename', 'ドキュメント')} (スコア: {score:.4f})", expanded=i==0):
                        # メタデータテーブル
                        metadata_df = pd.DataFrame([metadata])
                        st.dataframe(metadata_df, hide_index=True)
                        
                        # テキスト表示
                        st.markdown("**本文:**")
                        st.text(text[:500] + "..." if len(text) > 500 else text)
            else:
                st.info("検索結果はありません")
    
    # 文書登録タブ
    with search_tabs[1]:
        # コレクション名の入力
        collection_name = st.text_input(
            "コレクション名", 
            value=st.session_state.manager.collection_name,
            help="データを登録するコレクション名を指定します。新しいコレクション名を指定すると自動的に作成されます。"
        )
        
        # 登録方法の選択
        register_method = st.radio(
            "登録方法を選択",
            ["ファイルアップロード", "ディレクトリ指定"]
        )
        
        if register_method == "ファイルアップロード":
            # ファイルアップローダー
            uploaded_files = st.file_uploader(
                "ファイルをアップロード",
                accept_multiple_files=True,
                type=["pdf", "docx", "xlsx", "xls", "pptx", "txt", "csv", "json", "md", "html", "htm"]
            )
            
            if uploaded_files:
                if st.button("選択したファイルを登録", type="primary"):
                    with st.spinner("ファイルを処理中..."):
                        # 一時ファイルとして保存してから処理
                        texts = []
                        metadatas = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                                temp_file.write(uploaded_file.getbuffer())
                                temp_path = temp_file.name
                            
                            text, metadata = process_file(temp_path)
                            
                            if text:
                                texts.append(text)
                                metadatas.append(metadata)
                            
                            # 一時ファイルを削除
                            os.unlink(temp_path)
                            
                            # 進捗を更新
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        if texts:
                            # コレクション名を設定して処理
                            if collection_name != st.session_state.manager.collection_name:
                                # コレクションを取得または作成
                                st.session_state.manager.get_collection(collection_name)
                            
                            # Qdrantに追加
                            added_ids = add_files_to_qdrant(texts, metadatas)
                            
                            # 結果表示
                            st.success(f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")
                            
                            # 登録されたドキュメントの一覧
                            metadata_df = pd.DataFrame(metadatas)
                            st.dataframe(metadata_df)
                        else:
                            st.warning("登録できるドキュメントがありませんでした")
        
        else:  # ディレクトリ指定
            # ディレクトリパス入力
            directory_path = st.text_input("ディレクトリパスを入力", "")
            
            # 処理対象のファイル拡張子選択
            all_extensions = ['.pdf', '.docx', '.xlsx', '.xls', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']
            selected_extensions = st.multiselect(
                "処理対象の拡張子を選択",
                all_extensions,
                default=all_extensions
            )
            
            if directory_path and os.path.isdir(directory_path):
                if st.button("ディレクトリ内のファイルを登録", type="primary"):
                    with st.spinner(f"ディレクトリを処理中: {directory_path}"):
                        results = process_directory(directory_path, selected_extensions)
                        
                        if results:
                            texts = [r[0] for r in results]
                            metadatas = [r[1] for r in results]
                            
                            # コレクション名を設定して処理
                            if collection_name != st.session_state.manager.collection_name:
                                # コレクションを取得または作成
                                st.session_state.manager.get_collection(collection_name)
                            
                            # Qdrantに追加
                            added_ids = add_files_to_qdrant(texts, metadatas)
                            
                            # 結果表示
                            st.success(f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")
                            
                            # 登録されたドキュメントの一覧
                            metadata_df = pd.DataFrame(metadatas)
                            st.dataframe(metadata_df)
                        else:
                            st.warning("指定されたディレクトリに登録可能なファイルが見つかりませんでした")
            elif directory_path:
                st.error("指定されたディレクトリが存在しません")
