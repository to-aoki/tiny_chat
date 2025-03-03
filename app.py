import uuid
import streamlit as st
import os
from openai import OpenAI

from config_manager import Config, ModelManager
from file_processor import URIProcessor, FileProcessorFactory
from chat_manager import ChatManager

st.set_page_config(page_title="チャット", layout="wide")

# 設定ファイルのパス
CONFIG_FILE = "chat_app_config.json"


def reset_file_uploader():
    st.session_state.file_uploader_key = str(uuid.uuid4())


# 円形アニメーションのCSS
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

# 起動時に設定を読み込む
config = Config.load(CONFIG_FILE)

# セッション状態の初期化
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatManager()

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = str(uuid.uuid4())

if "server_url" not in st.session_state:
    st.session_state.server_url = config.server_url

if "api_key" not in st.session_state:
    st.session_state.api_key = config.api_key

if "previous_server_url" not in st.session_state:
    st.session_state.previous_server_url = st.session_state.server_url

if "message_length" not in st.session_state:
    st.session_state.message_length = config.message_length

if "context_length" not in st.session_state:
    st.session_state.context_length = config.context_length

if "uri_processing" not in st.session_state:
    st.session_state.uri_processing = config.uri_processing

# メッセージ送信中フラグ
if "is_sending_message" not in st.session_state:
    st.session_state.is_sending_message = False

# 処理ステータスメッセージ
if "status_message" not in st.session_state:
    st.session_state.status_message = ""

# モデル情報を初期化
if "available_models" not in st.session_state:
    models, success = ModelManager.fetch_available_models(
        st.session_state.server_url,
        st.session_state.api_key
    )
    st.session_state.available_models = models
    st.session_state.models_api_success = success

if "selected_model" not in st.session_state:
    st.session_state.selected_model = config.selected_model

if "meta_prompt" not in st.session_state:
    st.session_state.meta_prompt = config.meta_prompt

if "openai_client" not in st.session_state:
    try:
        st.session_state.openai_client = OpenAI(
            base_url=st.session_state.server_url,
            api_key=st.session_state.api_key
        )
    except Exception as e:
        st.error(f"OpenAI クライアントの初期化に失敗しました: {str(e)}")
        st.session_state.openai_client = None

with st.sidebar:
    st.header("API設定")

    # モデル選択または入力
    # APIに接続できる場合はドロップダウンリストを表示し、できない場合はテキスト入力欄を表示
    if st.session_state.models_api_success and st.session_state.available_models:
        model_input = st.selectbox(
            "モデル",
            st.session_state.available_models,
            index=st.session_state.available_models.index(
                st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0,
            help="API から取得したモデル一覧から選択します",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )
    else:
        model_input = st.text_input(
            "モデル",
            value=st.session_state.selected_model,
            help="使用するモデル名を入力してください",
            disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
        )

    # 入力されたモデルを使用
    if model_input != st.session_state.selected_model and not st.session_state.is_sending_message:
        st.session_state.selected_model = model_input
        # 設定を外部ファイルに保存
        config = Config(
            server_url=st.session_state.server_url,
            api_key=st.session_state.api_key,
            selected_model=model_input,
            meta_prompt=st.session_state.meta_prompt,
            context_length=st.session_state.context_length
        )
        config.save(CONFIG_FILE)

    # サーバーURL設定
    server_url = st.text_input(
        "サーバーURL",
        value=st.session_state.server_url,
        help="OpenAI APIサーバーのURLを入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # APIキー設定
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="APIキーを入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # メタプロンプト設定
    meta_prompt = st.text_area(
        "メタプロンプト",
        value=st.session_state.meta_prompt,
        height=150,
        help="LLMへのsystem指示を入力してください",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # メッセージ長設定
    message_length = st.number_input(
        "メッセージ長",
        min_value=1000,
        max_value=500000,
        value=st.session_state.message_length,
        step=500,
        help="入力最大メッセージ長を決定します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # コンテキスト長設定
    context_length = st.number_input(
        "コンテキスト長",
        min_value=100,
        max_value=100000,
        value=st.session_state.context_length,
        step=500,
        help="添付ファイルやURLからのコンテンツを切り詰める文字数を指定します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    # URLチェックボックス
    uri_processing = st.checkbox(
        "メッセージURLを取得",
        value=st.session_state.uri_processing,
        help="メッセージの最初のURLからコンテキストを取得し、プロンプトを拡張します",
        disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
    )

    if uri_processing != st.session_state.uri_processing and not st.session_state.is_sending_message:
        st.session_state.uri_processing = uri_processing

    if server_url != st.session_state.server_url and not st.session_state.is_sending_message:
        st.session_state.previous_server_url = st.session_state.server_url
        st.session_state.server_url = server_url
        st.session_state.api_key = api_key

        # モデルリストを更新し、必要に応じて選択モデルも更新
        new_models, selected_model, api_success = ModelManager.update_models_on_server_change(
            server_url,
            api_key,
            st.session_state.selected_model
        )

        st.session_state.available_models = new_models
        st.session_state.models_api_success = api_success

        # モデルの自動変更通知 (新しいサーバーで現在のモデルが利用できない場合)
        if selected_model != st.session_state.selected_model and new_models:
            st.session_state.selected_model = selected_model
            st.info(
                f"選択したモデル '{st.session_state.selected_model}' は新しいサーバーでは利用できません。"
                f"'{selected_model}' に変更されました。")

        try:
            st.session_state.openai_client = OpenAI(
                base_url=server_url,
                api_key=api_key
            )
        except Exception as e:
            st.error(f"OpenAI クライアントの初期化に失敗しました: {str(e)}")
            st.session_state.openai_client = None

    else:
        # サーバURLは同じだがAPIキーだけ変更された場合（かつメッセージ送信中でない場合）
        if api_key != st.session_state.api_key and not st.session_state.is_sending_message:
            st.session_state.api_key = api_key
            models, api_success = ModelManager.fetch_available_models(
                server_url,
                api_key
            )
            st.session_state.available_models = models
            st.session_state.models_api_success = api_success

            try:
                st.session_state.openai_client = OpenAI(
                    base_url=server_url,
                    api_key=api_key
                )
            except Exception as e:
                st.error(f"OpenAI クライアントの初期化に失敗しました: {str(e)}")
                st.session_state.openai_client = None

    # 設定を反映ボタン
    if st.button("設定を反映", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        # サーバURLが変更された場合はモデルリストを更新
        if server_url != st.session_state.server_url:
            st.session_state.previous_server_url = st.session_state.server_url
            st.session_state.server_url = server_url
            st.session_state.api_key = api_key

            # モデルリストを更新し、必要に応じて選択モデルも更新
            new_models, selected_model, api_success = ModelManager.update_models_on_server_change(
                server_url,
                api_key,
                st.session_state.selected_model
            )

            st.session_state.available_models = new_models
            st.session_state.models_api_success = api_success

            # モデルの自動変更通知 (新しいサーバーで現在のモデルが利用できない場合)
            if selected_model != st.session_state.selected_model and new_models:
                st.session_state.selected_model = selected_model
                st.info(
                    f"選択したモデル '{st.session_state.selected_model}' は新しいサーバーでは利用できません。"
                    f"'{selected_model}' に変更されました。")
        else:
            # サーバURLは同じだがAPIキーだけ変更された場合
            st.session_state.api_key = api_key
            models, api_success = ModelManager.fetch_available_models(
                server_url,
                api_key
            )
            st.session_state.available_models = models
            st.session_state.models_api_success = api_success

        # クライアントを再初期化
        try:
            st.session_state.openai_client = OpenAI(
                base_url=server_url,
                api_key=api_key
            )
        except Exception as e:
            st.error(f"OpenAI クライアントの初期化に失敗しました: {str(e)}")
            st.session_state.openai_client = None

        st.session_state.meta_prompt = meta_prompt
        st.session_state.message_length = message_length
        st.session_state.context_length = context_length

        # 設定を外部ファイルに保存
        config = Config(
            server_url=server_url,
            api_key=api_key,
            selected_model=st.session_state.selected_model,
            meta_prompt=meta_prompt,
            message_length=message_length,
            context_length=context_length,
            uri_processing=uri_processing,
        )

        if config.save(CONFIG_FILE):
            st.success("設定を更新し、ファイルに保存しました")
        else:
            st.warning("設定は更新されましたが、ファイルへの保存に失敗しました")

    # メッセージ履歴をJSONとして出力する機能
    st.markdown("---")
    if st.button("メッセージ履歴をJSONで保存", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        if not st.session_state.chat_manager.messages:
            st.warning("保存するメッセージ履歴がありません")
        else:
            # 拡張プロンプトを含むメッセージ履歴を生成
            chat_history = st.session_state.chat_manager.to_json()

            # ダウンロードボタンを表示
            st.download_button(
                label="JSONファイルをダウンロード",
                data=chat_history,
                file_name="chat_history.json",
                mime="application/json",
                disabled=st.session_state.is_sending_message  # メッセージ送信中は無効化
            )

    # メッセージ履歴のクリア機能
    st.markdown("---")
    if st.button("チャット履歴をクリア", disabled=st.session_state.is_sending_message):  # メッセージ送信中は無効化
        st.session_state.chat_manager = ChatManager()
        reset_file_uploader()
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
        # アップロードされたJSONファイルを読み込む
        content = uploaded_json.getvalue().decode("utf-8")

        # 新しいメソッドを使用して履歴を適用
        if st.button("インポートした履歴を適用", disabled=st.session_state.is_sending_message):
            # ChatManagerに追加した新しいメソッドを呼び出す
            if hasattr(st.session_state.chat_manager, 'apply_imported_history'):
                success = st.session_state.chat_manager.apply_imported_history(content)
            else:
                # 新しいメソッドがない場合は従来のメソッドを使用
                success = st.session_state.chat_manager.load_from_json(content)

            if success:
                st.success("メッセージ履歴を正常にインポートしました")
                st.rerun()
            else:
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

            with cols[1]:
                # メッセージ送信中はボタンを無効化
                if st.session_state.is_sending_message:
                    st.button("削除", key=f"delete_{idx}", disabled=True)
                else:
                    if st.button("削除", key=f"delete_{idx}"):
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
        st.session_state.message_length,
        st.session_state.context_length,
        st.session_state.meta_prompt,
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

# メッセージ送信処理（フラグがONの場合に実行）
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
        if st.session_state.uri_processing:
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
                max_length=st.session_state.context_length,
                uri_processor=uri_processor
            )
            if enhanced_prompt:
                # 拡張プロンプトで最後のユーザーメッセージを更新
                st.session_state.chat_manager.update_enhanced_prompt(enhanced_prompt)

        # 処理ステータスを更新
        st.session_state.status_message = "AIからの応答を生成中..."

        # APIに送信するメッセージを準備
        messages_for_api = st.session_state.chat_manager.prepare_messages_for_api(st.session_state.meta_prompt)

        # 空のメッセージ配列を修正
        if not messages_for_api:
            # システムメッセージがあれば追加
            if st.session_state.meta_prompt:
                messages_for_api.append({"role": "system", "content": st.session_state.meta_prompt})

            # 少なくとも1つのユーザーメッセージを追加
            messages_for_api.append({"role": "user", "content": prompt_content})

        # AIからの応答を取得
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("_応答を生成中..._")

            try:
                # クライアントインスタンスが存在しない場合は初期化
                if "openai_client" not in st.session_state or st.session_state.openai_client is None:
                    st.session_state.openai_client = OpenAI(
                        base_url=st.session_state.server_url,
                        api_key=st.session_state.api_key
                    )

                # 既存のクライアントインスタンスを使用
                client = st.session_state.openai_client

                # ストリーミングモードでリクエスト
                response = client.chat.completions.create(
                    model=st.session_state.selected_model,
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