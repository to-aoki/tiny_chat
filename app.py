import uuid
import streamlit as st
import os
import logging

from config_manager import Config, ModelManager
from file_processor import URIProcessor, FileProcessorFactory
from chat_manager import ChatManager
from logger import get_logger
from llm_utils import get_llm_client

# ロガーを初期化
logger = get_logger(log_dir="logs", log_level=logging.INFO)

st.set_page_config(page_title="チャット", layout="wide")

# 設定ファイルのパス
CONFIG_FILE = "chat_app_config.json"


def reset_file_uploader():
    st.session_state.file_uploader_key = str(uuid.uuid4())



circle_spinner_css = """
<style>
.spinner {
  width: 40px;
  height: 40px;
  margin: 10px auto;
  border: 5px solid rgba(0, 120, 212, 0.2);
  border-top-color: rgba(0, 120, 212, 1);
  border-radius: 50%;
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.spinner-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background-color: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin: 10px 0;
}

.spinner-text {
  margin-top: 10px;
  font-weight: bold;
  color: #0078D4;
}
</style>
"""


def initialize_session_state():
    # 設定値を直接保持するconfigオブジェクトをセッション状態に追加
    if "config" not in st.session_state:
        # 外部設定ファイルから設定を読み込む
        file_config = Config.load(CONFIG_FILE)
        logger.info(f"設定ファイルを読み込みました: {CONFIG_FILE}")

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

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = str(uuid.uuid4())

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


# セッション状態を初期化
initialize_session_state()

with st.sidebar:
    st.header("API設定")

    # モデル選択または入力
    # APIに接続できる場合はドロップダウンリストを表示し、できない場合はテキスト入力欄を表示
    if st.session_state.models_api_success and st.session_state.available_models:
        model_input = st.selectbox(
            "モデル",
            st.session_state.available_models,
            index=st.session_state.available_models.index(
                st.session_state.config["selected_model"]) if st.session_state.config[
                                                                  "selected_model"] in st.session_state.available_models else 0,
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
        config.save(CONFIG_FILE)
        logger.info("設定を保存しました")

    # サーバーURL設定
    server_url = st.text_input(
        "サーバーURL",
        value=st.session_state.config["server_url"],
        help="OpenAI APIサーバーのURLを入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # APIキー設定
    api_key = st.text_input(
        "API Key",
        value=st.session_state.config["api_key"],
        type="password",
        help="APIキーを入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # Azureチェックボックス
    is_azure = st.checkbox(
        "Azure OpenAIを利用",
        value=st.session_state.config["is_azure"],
        help="APIはAzure OpenAIを利用します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # メタプロンプト設定
    meta_prompt = st.text_area(
        "メタプロンプト",
        value=st.session_state.config["meta_prompt"],
        height=150,
        help="LLMへのsystem指示を入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # メッセージ長設定
    message_length = st.number_input(
        "メッセージ長",
        min_value=1000,
        max_value=500000,
        value=st.session_state.config["message_length"],
        step=500,
        help="入力最大メッセージ長を決定します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # コンテキスト長設定
    context_length = st.number_input(
        "コンテキスト長",
        min_value=100,
        max_value=100000,
        value=st.session_state.config["context_length"],
        step=500,
        help="添付ファイルやURLからのコンテンツを切り詰める文字数を指定します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # URLチェックボックス
    uri_processing = st.checkbox(
        "メッセージURLを取得",
        value=st.session_state.config["uri_processing"],
        help="メッセージの最初のURLからコンテキストを取得し、プロンプトを拡張します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    if uri_processing != st.session_state.config["uri_processing"] and not st.session_state.is_sending_message:
        st.session_state.config["uri_processing"] = uri_processing
        logger.info(f"URI処理設定を変更: {uri_processing}")

    if server_url != st.session_state.config["server_url"] and not st.session_state.is_sending_message:
        logger.info(f"サーバーURLを変更: {server_url}")
        st.session_state.config["previous_server_url"] = st.session_state.config["server_url"]
        st.session_state.config["server_url"] = server_url
        st.session_state.config["api_key"] = api_key

        # モデルリストを更新し、必要に応じて選択モデルも更新
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
        # サーバURLは同じだがAPIキーだけ変更された場合（かつメッセージ送信中でない場合）
        if api_key != st.session_state.config["api_key"] and not st.session_state.is_sending_message:
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

    # 設定を反映ボタン
    if st.button("設定を反映", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        logger.info("設定反映ボタンがクリックされました")
        # サーバURLが変更された場合はモデルリストを更新
        if server_url != st.session_state.config["server_url"]:
            logger.info(f"サーバーURLを変更: {server_url}")
            st.session_state.config["previous_server_url"] = st.session_state.config["server_url"]
            st.session_state.config["server_url"] = server_url
            st.session_state.config["api_key"] = api_key

            # モデルリストを更新し、必要に応じて選択モデルも更新
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

        # クライアントを再初期化
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

        # その他の設定を更新
        if meta_prompt != st.session_state.config["meta_prompt"]:
            logger.info("メタプロンプトを更新しました")
        st.session_state.config["meta_prompt"] = meta_prompt

        if message_length != st.session_state.config["message_length"]:
            logger.info(f"メッセージ長を更新: {message_length}")
        st.session_state.config["message_length"] = message_length

        if context_length != st.session_state.config["context_length"]:
            logger.info(f"コンテキスト長を更新: {context_length}")
        st.session_state.config["context_length"] = context_length

        st.session_state.config["is_azure"] = is_azure

        # 設定を外部ファイルに保存
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

        if config.save(CONFIG_FILE):
            logger.info("設定をファイルに保存しました")
            st.success("設定を更新し、ファイルに保存しました")
        else:
            logger.warning("設定ファイルへの保存に失敗しました")
            st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

    # メッセージ履歴をJSONとして出力する機能
    st.markdown("---")
    if st.button("メッセージ履歴をJSONで保存", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        logger.info("JSONエクスポートボタンがクリックされました")
        if not st.session_state.chat_manager.messages:
            logger.warning("保存するメッセージ履歴がありません")
            st.warning("保存するメッセージ履歴がありません")
        else:
            # 拡張プロンプトを含むメッセージ履歴を生成
            logger.info("メッセージ履歴をJSONに変換中...")
            chat_history = st.session_state.chat_manager.to_json()

            # ダウンロードボタンを表示
            st.download_button(
                label="JSONファイルをダウンロード",
                data=chat_history,
                file_name="chat_history.json",
                mime="application/json",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )
            logger.info("メッセージ履歴のJSONエクスポートを準備しました")

    # メッセージ履歴のクリア機能
    st.markdown("---")
    if st.button("チャット履歴をクリア", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        logger.info("チャット履歴クリアボタンがクリックされました")
        st.session_state.chat_manager = ChatManager()
        reset_file_uploader()
        logger.info("チャット履歴をクリアしました")
        st.rerun()

    # メッセージ履歴のインポート機能
    st.markdown("---")
    uploaded_json = st.file_uploader(
        "メッセージ履歴をインポート (JSON)",
        type=["json"],
        help="以前に保存したチャット履歴JSONファイルをアップロードします",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    if uploaded_json is not None:
        logger.info(f"JSONファイルがアップロードされました: {uploaded_json.name}")
        # アップロードされたJSONファイルを読み込む
        content = uploaded_json.getvalue().decode("utf-8")

        # 新しいメソッドを使用して履歴を適用
        if st.button("インポートした履歴を適用", disabled=st.session_state.is_sending_message):
            logger.info("履歴インポート適用ボタンがクリックされました")
            # ChatManagerに追加した新しいメソッドを呼び出す
            if hasattr(st.session_state.chat_manager, 'apply_imported_history'):
                success = st.session_state.chat_manager.apply_imported_history(content)
            else:
                # 新しいメソッドがない場合は従来のメソッドを使用
                logger.warning("apply_imported_history メソッドが見つかりません。load_from_json を使用します")
                success = st.session_state.chat_manager.load_from_json(content)

            if success:
                logger.info("メッセージ履歴のインポートに成功しました")
                st.success("メッセージ履歴を正常にインポートしました")
                st.rerun()
            else:
                logger.error("メッセージ履歴のインポートに失敗しました: 無効なフォーマット")
                st.error("JSONのインポートに失敗しました: 無効なフォーマットです")

# 処理中ステータス表示エリア
status_area = st.empty()

# チャット履歴の表示
for message in st.session_state.chat_manager.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 添付ファイル一覧を表示
if st.session_state.chat_manager.attachments:
    with st.expander(f"添付ファイル ({len(st.session_state.chat_manager.attachments)}件)", expanded=True):
        for idx, attachment in enumerate(st.session_state.chat_manager.attachments):
            cols = st.columns([4, 1])
            with cols[0]:
                filename = attachment['filename']
                _, ext = os.path.splitext(filename)

                # ファイルタイプの判定をシンプルに
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

                # デフォルト値の設定
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

            with cols[1]:
                # メッセージ送信中はボタンを無効化
                if st.session_state.is_sending_message:
                    st.button("削除", key=f"delete_{idx}", disabled=True)
                else:
                    if st.button("削除", key=f"delete_{idx}"):
                        logger.info(f"添付ファイル削除: {attachment['filename']}")
                        st.session_state.chat_manager.attachments.pop(idx)
                        st.rerun()

# メッセージ処理中の表示を入力欄の直前に移動
if st.session_state.is_sending_message:
    with st.container():
        # CSSを挿入
        st.markdown(circle_spinner_css, unsafe_allow_html=True)

        # 円形のスピナーとメッセージを表示
        st.markdown(f"""
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="spinner-text">{st.session_state.status_message}</div>
        </div>
        """, unsafe_allow_html=True)

# サポートするファイル形式を定義
supported_extensions = ["pdf", "xlsx", "xls", "docx", "pptx", "txt", "csv", "json", "md", "html"]
uploaded_file = st.file_uploader(
    "ファイルをアップロード",
    type=supported_extensions,
    key=st.session_state.file_uploader_key,
    disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
)

# ファイル処理
if uploaded_file is not None:
    with st.status(f"ファイルを処理中...") as status:
        import os

        # ファイル拡張子の取得
        filename = uploaded_file.name
        _, file_extension = os.path.splitext(filename)

        # 適切なプロセッサを取得
        processor_class = FileProcessorFactory.get_processor(file_extension)

        if processor_class is None:
            status.update(label=f"エラー: サポートされていないファイル形式です: {file_extension}", state="error")
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
                status.update(label=f"エラー: {error}", state="error")
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

                # ファイル形式に応じたメッセージ表示
                if count_value > 1 and count_type != "":
                    status.update(label=f"'{filename}'から{count_value}{count_type}のテキストを抽出しました",
                                  state="complete")
                else:
                    status.update(label=f"'{filename}'からテキストを抽出しました", state="complete")

    # ファイルアップローダーをリセット
    reset_file_uploader()
    st.rerun()

# ユーザー入力 (メッセージ送信中は無効化)
prompt = st.chat_input(
    "メッセージを入力してください...",
    disabled=st.session_state.is_sending_message
)

# 新しいメッセージが入力された場合の処理
if prompt:
    # メッセージ長チェック
    would_exceed, estimated_length, max_length = st.session_state.chat_manager.would_exceed_message_length(
        prompt,
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
        user_message = st.session_state.chat_manager.add_user_message(prompt)

        # UIに表示
        with st.chat_message("user"):
            st.write(user_message["content"])

        # メッセージ送信中フラグをON
        st.session_state.is_sending_message = True
        st.session_state.status_message = "メッセージを処理中..."
        st.rerun()  # 状態を更新してUIを再描画

# メッセージ送信処理
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

        # 拡張プロンプトを生成
        if st.session_state.chat_manager.attachments or len(detects_urls) > 0:
            # 処理ステータスを更新
            if st.session_state.chat_manager.attachments:
                st.session_state.status_message = "添付ファイルの内容を解析中..."
            elif len(detects_urls) > 0:
                st.session_state.status_message = "URLからコンテンツを取得中..."

            # 拡張プロンプトを生成
            enhanced_prompt = st.session_state.chat_manager.get_enhanced_prompt(
                prompt_content,
                max_length=st.session_state.config["context_length"],
                uri_processor=uri_processor
            )
            if enhanced_prompt:
                # 拡張プロンプトで最後のユーザーメッセージを更新
                st.session_state.chat_manager.update_enhanced_prompt(enhanced_prompt)

        # 処理ステータスを更新
        st.session_state.status_message = "AIからの応答を生成中..."

        # APIに送信するメッセージを準備
        messages_for_api = st.session_state.chat_manager.prepare_messages_for_api(
            st.session_state.config["meta_prompt"])

        # 空のメッセージ配列を修正
        if not messages_for_api:
            # システムメッセージがあれば追加
            if st.session_state.config["meta_prompt"]:
                messages_for_api.append({"role": "system", "content": st.session_state.config["meta_prompt"]})

            # 少なくとも1つのユーザーメッセージを追加
            messages_for_api.append({"role": "user", "content": prompt_content})

        # AIからの応答を取得
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("_応答を生成中..._")

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
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_response += delta.content
                            # リアルタイムで表示を更新
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

    # 処理完了後にフラグをOFF
    st.session_state.is_sending_message = False
    st.rerun()  # 状態を更新してUIを再描画
    st.session_state.status_message = "処理完了"