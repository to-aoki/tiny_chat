import os
from typing import List, Dict, Any, Tuple
import tempfile

import webbrowser
import streamlit as st
import pandas as pd
from file_processor import FileProcessorFactory


# プロセスレベルでQdrantManagerインスタンスを保持するためのグローバル変数
_qdrant_manager = None
# インスタンス生成のロックに使用
_qdrant_lock = None


def get_or_create_qdrant_manager(logger=None):
    """
    QdrantManagerを取得または初期化する共通関数
    プロセスレベルで一つのインスタンスを共有するよう修正
    スレッドセーフな実装を使用

    Args:
        logger: ロガーオブジェクト（オプション）

    Returns:
        QdrantManager: 初期化されたQdrantManagerオブジェクト
    """
    global _qdrant_manager, _qdrant_lock
    from qdrant_manager import QdrantManager
    import threading
    
    # ロックオブジェクトがなければ作成
    if _qdrant_lock is None:
        _qdrant_lock = threading.Lock()
    
    # ロックを取得して排他制御
    with _qdrant_lock:
        # プロセスレベルでQdrantManagerがまだ初期化されていない場合は初期化
        if _qdrant_manager is None:
            with st.spinner("検索データベースを初期化中..."):
                if logger:
                    logger.info("QdrantManagerを初期化しています...")
                _qdrant_manager = QdrantManager(
                    collection_name="default",
                    path="./qdrant_data"
                )
                if logger:
                    logger.info("QdrantManagerの初期化が完了しました")
    
    return _qdrant_manager


def process_file(file_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    ファイルを処理し、テキストとメタデータを抽出します
    is_page=Trueで処理するため、テキストは文字列のリストとして返されます

    Args:
        file_path: 処理するファイルのパス

    Returns:
        (extracted_text_array, metadata): 抽出されたテキスト配列とメタデータの辞書
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

    # ファイルタイプに応じた処理、is_page=Trueで文字列配列として処理
    if file_ext == '.pdf':
        text, page_count, error = processor.extract_text_from_bytes(file_bytes, is_page=True)
        if error:
            st.warning(f"PDFの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["page_count"] = page_count

    elif file_ext in ['.xlsx']:
        text, sheet_count, error = processor.extract_text_from_bytes(file_bytes, is_page=True)
        if error:
            st.warning(f"Excelの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["sheet_count"] = sheet_count

    elif file_ext == '.docx':
        text, para_count, error = processor.extract_text_from_bytes(file_bytes, is_page=True)
        if error:
            st.warning(f"Wordの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["para_count"] = para_count

    elif file_ext == '.pptx':
        text, slide_count, error = processor.extract_text_from_bytes(file_bytes, is_page=True)
        if error:
            st.warning(f"PowerPointの処理中にエラーが発生しました: {error}")
            return None, {}
        metadata["slide_count"] = slide_count

    elif file_ext in ['.txt', '.csv', '.json', '.md']:
        # テキストファイルの場合はページ分割機能がないため従来通り
        text, error = processor.extract_text_from_bytes(file_bytes)
        if error:
            st.warning(f"テキストファイルの処理中にエラーが発生しました: {error}")
            return None, {}
        # 単一のテキストを配列に変換して1ページとして扱う
        if text:
            text = [text]

    elif file_ext in ['.html', '.htm']:
        text, message = processor.extract_text_from_bytes(file_bytes)
        if not text:
            st.warning(f"HTMLの処理中にエラーが発生しました: {message}")
            return None, {}
        # 単一のテキストを配列に変換して1ページとして扱う
        if text:
            text = [text]

    else:
        st.warning(f"対応していないファイル形式です: {file_ext}")
        return None, {}

    # ファイルサイズを追加
    metadata["file_size"] = len(file_bytes)

    return text, metadata


def process_directory(directory_path: str,
    extensions: List[str] = None,
    support_extensions: List[str] = ['.pdf', '.xlsx', '.xls', '.docx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']
) -> List[Tuple[List[str], Dict]]:
    """
    ディレクトリ内のファイルを処理します

    Args:
        directory_path: 処理するディレクトリのパス
        extensions: 処理対象のファイル拡張子リスト (None の場合はすべてのサポートされる形式)

    Returns:
        [(text_array, metadata), ...]: 抽出されたテキスト配列とメタデータのリスト
    """
    results = []

    if extensions is None:
        extensions = support_extensions

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


def add_files_to_qdrant(texts: List[List[str]], metadatas: List[Dict]) -> List[str]:
    """
    テキストとメタデータをQdrantに追加します
    同じソース（ファイル名）が既に存在する場合は、削除してから追加します

    Args:
        texts: テキスト配列のリスト (is_page=True により、各ファイルのテキストは文字列の配列)
        metadatas: メタデータのリスト

    Returns:
        added_ids: 追加されたドキュメントのIDリスト
    """
    # QdrantManagerを取得（スレッドセーフな共通関数）
    _qdrant_manager = get_or_create_qdrant_manager()
    
    # ソース（ファイル名）の一覧を取得
    sources_to_add = set()
    for metadata in metadatas:
        if "source" in metadata:
            sources_to_add.add(metadata["source"])
    
    # 既存のソースと照合し、重複があれば削除
    existing_sources = _qdrant_manager.get_sources()
    for source in sources_to_add:
        if source in existing_sources:
            # ソースに関連するデータを削除
            # ソース名を配列として渡す（単一でも配列として扱う）
            filter_params = {"source": [source]}
            _qdrant_manager.delete_by_filter(filter_params)

    all_texts = []
    all_metadatas = []
    
    for i, text_array in enumerate(texts):
        base_metadata = metadatas[i].copy()
        for page_index, page_text in enumerate(text_array):
            all_texts.append(page_text)
            page_metadata = base_metadata.copy()
            page_metadata["page"] = page_index + 1  # 配列の添字 + 1 をページとして設定
            all_metadatas.append(page_metadata)
    
    # Qdrantに追加
    added_ids = _qdrant_manager.add_documents(all_texts, all_metadatas)
    return added_ids


def search_documents(
        query: str, top_k: int = 10, filter_params: Dict = None, logger=None, score_threshold=0.4) -> List:
    """
    ドキュメントを検索します

    Args:
        query: 検索クエリ
        top_k: 返す結果の数
        filter_params: 検索フィルタ（Noneの場合はフィルタを適用しない）

    Returns:
        results: 検索結果のリスト
    """
    _qdrant_manager = get_or_create_qdrant_manager(logger)
    results = _qdrant_manager.query_points(
        query, top_k=top_k, filter_params=filter_params, score_threshold=score_threshold)
    return results


@st.fragment
def show_database_component(
        logger,
        extensions=['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']):
    # 検索と文書登録のタブを作成
    search_tabs = st.tabs(["🔍 検索", "📁 登録", "🗑️ 削除"])

    # QdrantManagerを取得（必要に応じて初期化）
    _qdrant_manager = get_or_create_qdrant_manager(logger)

    # 検索タブ
    with search_tabs[0]:
        # 検索のエンターキー対応のためのコールバック関数
        if "search_query" not in st.session_state:
            st.session_state.search_query = ""
            
        def search_on_enter():
            # テキスト入力からクエリを取得し、検索実行のフラグを立てる
            st.session_state.run_search = True
            
        # 検索フィールド (Enterキーで検索実行するためにon_changeパラメータを追加)
        query = st.text_input("検索文字列", "", key="search_query_input", on_change=search_on_enter)

        # 詳細設定のエクスパンダー
        with st.expander("詳細設定", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                top_k = st.slider("表示件数", min_value=1, max_value=50, value=10)

            with col2:
                # 使用可能なソースを取得（常に最新の状態を取得）
                sources = _qdrant_manager.get_sources()
                selected_sources = st.multiselect(
                    "ソースでフィルタ", 
                    options=sources,
                    key="sources_multiselect_filter"  # 固定のキーを使用
                )

        # 検索の実行フラグをセットアップ
        if "run_search" not in st.session_state:
            st.session_state.run_search = False
        
        # 検索ボタン
        search_pressed = st.button("検索", key="search_button", type="primary")

        # 検索実行（ボタン押下またはEnterキー押下で実行）
        if (search_pressed or st.session_state.run_search) and query:
            # 検索フラグをリセット
            st.session_state.run_search = False
            # フィルターの作成
            filter_params = None
            if selected_sources:
                # 複数のソースを配列として設定
                filter_params = {"source": selected_sources}

            with st.spinner("検索中..."):
                results = search_documents(query, top_k=top_k, filter_params=filter_params, score_threshold=0.)

            # 結果の表示
            if results:
                st.success(f"{len(results)}件の結果が見つかりました")

                for i, result in enumerate(results):
                    score = result.score
                    text = result.payload.get("text", "")

                    # メタデータを表示用に整形
                    metadata = {k: v for k, v in result.payload.items() if k != "text"}

                    # ページ情報（あれば）を取得
                    page_info = ""
                    if 'page' in metadata:
                        # ファイル種類に応じて表示を変える
                        file_type = metadata.get('file_type', '').lower()
                        if file_type == 'pdf':
                            page_info = f"(ページ: {metadata['page']})"
                        elif file_type == 'xlsx':
                            page_info = f"(シート: {metadata['page']})"
                        elif file_type == 'docx':
                            page_info = f"(段落: {metadata['page']})"
                        elif file_type == 'pptx':
                            page_info = f"(スライド: {metadata['page']})"
                        else:
                            page_info = f"(記載箇所: {metadata['page']})"
                        
                    # 結果表示
                    with st.expander(
                            f"#{i + 1}: {metadata.get('filename', 'ドキュメント')} {page_info} (スコア: {score:.4f})",
                            expanded=i == 0):
                        # メタデータテーブル
                        metadata_df = pd.DataFrame([metadata])
                        st.dataframe(metadata_df, hide_index=True)
                        
                        # ソースファイルへのリンクを追加（あれば）
                        if 'source' in metadata and metadata['source']:
                            source_path = metadata['source']
                            
                            if source_path.startswith(('http://', 'https://')):
                                # 外部URLまたはファイルURLの場合は直接リンクボタンを表示（新規タブで開く）
                                st.markdown(f'<a href="{source_path}" target="_blank" rel="noopener noreferrer">リンクを新規タブで開く</a>', unsafe_allow_html=True)

                        # ローカルファイルを開くはStreamlit expander仕様およびブラウザセキュリティ都合から実装しない（煩雑で危険）

                        # テキスト表示
                        st.markdown("**本文:**")
                        st.text(text[:500] + "..." if len(text) > 500 else text)
            else:
                st.info("検索結果はありません")

    # 文書登録タブ
    with (search_tabs[1]):
        # コレクション名の入力
        collection_name = st.text_input(
            "コレクション名",
            value=_qdrant_manager.collection_name,
            help="データを登録するコレクション名を指定します。新しいコレクション名を指定すると自動的に作成されます。"
        )

        # 登録方法の選択
        register_method = st.radio(
            "登録方法を選択",
            ["ファイルアップロード", "ディレクトリ指定"]
        )

        if register_method == "ファイルアップロード":
            # ソースパスのベースディレクトリ設定
            source_base_dir = st.text_input(
                "ソースパスのベースディレクトリ（省略可）",
                "",
                help="ファイルの「source」として使用するベースディレクトリを指定できます。空の場合はファイル名のみが使用されます。"
            )
            
            # ファイルアップローダー
            uploaded_files = st.file_uploader(
                "ファイルをアップロード",
                accept_multiple_files=True,
                type=[ext.lstrip(".") for ext in extensions]
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
                            
                            # カスタムソースパスを設定（指定があれば）
                            if text and source_base_dir:
                                custom_source_path = os.path.join(source_base_dir, uploaded_file.name)
                                metadata["source"] = custom_source_path
                                metadata["original_filename"] = metadata["filename"]  # 元のファイル名を保持

                            if text:
                                texts.append(text)
                                metadatas.append(metadata)

                            # 一時ファイルを削除
                            os.unlink(temp_path)

                            # 進捗を更新
                            progress_bar.progress((i + 1) / len(uploaded_files))

                        if texts:
                            # コレクション名を設定して処理
                            if collection_name != _qdrant_manager.collection_name:
                                # コレクションを取得または作成
                                _qdrant_manager.get_collection(collection_name)

                            # Qdrantに追加
                            added_ids = add_files_to_qdrant(texts, metadatas)

                            # 結果表示
                            st.success(
                                f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")

                            # 登録されたドキュメントの一覧
                            metadata_df = pd.DataFrame(metadatas)
                            st.dataframe(metadata_df)
                        else:
                            st.warning("登録できるドキュメントがありませんでした")

        else:  # ディレクトリ指定
            # ディレクトリパス入力
            directory_path = st.text_input("ディレクトリパスを入力", "")
            
            # ソースパスのカスタマイズオプション
            source_path_option = st.radio(
                "ソースパスの設定方法",
                ["実際のファイルパスを使用", "Webパス接頭辞を設定"],
                help="ファイルの「source」として使用するパスの設定方法を選択します"
            )
            
            web_prefix_base = ""
            if source_path_option == "Webパス接頭辞を設定":
                web_prefix_base = st.text_input(
                    "Webパス接頭辞",
                    "",
                    help="ファイルの「source」として使用するWebパス接頭辞を指定します。入力ディレクトリの末尾名と相対パスがこの接頭辞に追加されます。例: http://example.com/path/to/"
                )

            # 処理対象のファイル拡張子選択
            selected_extensions = st.multiselect(
                "処理対象の拡張子を選択",
                extensions,
                default=extensions
            )

            if directory_path and os.path.isdir(directory_path):
                if st.button("ディレクトリ内のファイルを登録", type="primary"):
                    with st.spinner(f"ディレクトリを処理中: {directory_path}"):
                        results = process_directory(directory_path, selected_extensions,
                                                    support_extensions=extensions)

                        if results:
                            texts = [r[0] for r in results]
                            metadatas = [r[1] for r in results]

                            # Webパス接頭辞の設定（指定があれば）
                            if web_prefix_base:
                                for metadata in metadatas:
                                    if "rel_path" in metadata:
                                        # 元のソースパスを保持
                                        metadata["original_source"] = metadata["source"]
                                        # ディレクトリ末尾名を取得
                                        last_dir_name = os.path.basename(os.path.normpath(directory_path))
                                        # Webパス接頭辞にディレクトリ末尾名と相対パスを結合
                                        if web_prefix_base.endswith('/'):
                                            web_path = f"{web_prefix_base}{last_dir_name}/{metadata['rel_path']}"
                                        else:
                                            web_path = f"{web_prefix_base}/{last_dir_name}/{metadata['rel_path']}"
                                        # URLのパス区切り文字を統一（Windowsの場合のバックスラッシュを置換）
                                        web_path = web_path.replace('\\', '/')
                                        metadata["source"] = web_path

                            # コレクション名を設定して処理
                            if collection_name != _qdrant_manager.collection_name:
                                # コレクションを取得または作成
                                _qdrant_manager.get_collection(collection_name)

                            # Qdrantに追加
                            added_ids = add_files_to_qdrant(texts, metadatas)

                            # 結果表示
                            st.success(
                                f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")

                            # 登録されたドキュメントの一覧
                            metadata_df = pd.DataFrame(metadatas)
                            st.dataframe(metadata_df)
                        else:
                            st.warning("指定されたディレクトリに登録可能なファイルが見つかりませんでした")
            elif directory_path:
                st.error("指定されたディレクトリが存在しません")

    # データ管理タブ
    with search_tabs[2]:

        # タブを作成
        data_management_tabs = st.tabs(["ソース管理", "コレクション管理"])

        # ソース管理タブ
        with data_management_tabs[0]:

            # コレクション名の入力
            collection_name = st.text_input(
                "コレクション名",
                value=_qdrant_manager.collection_name,
                help="操作対象のコレクション名を指定します。",
                key="data_management_collection"
            )

            # 使用可能なソースを取得（常に最新の状態を取得）
            sources = _qdrant_manager.get_sources()

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
                
                # 選択したソースをセッション状態に保存（削除確認時に使用）
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
                    confirm = st.warning(
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
                                # コレクション名を設定
                                if collection_name != _qdrant_manager.collection_name:
                                    _qdrant_manager.get_collection(collection_name)

                                # ソースでフィルタリングして削除（ソース名が単一でも配列として渡す）
                                filter_params = {"source": [selected_source_to_delete]}
                                _qdrant_manager.delete_by_filter(filter_params)
                                
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

        # コレクション管理タブ
        with data_management_tabs[1]:

            # 利用可能なコレクション一覧を取得
            collections = _qdrant_manager.get_collections()

            if not collections:
                st.warning("データベースにコレクションが見つかりません。")
            else:
                # コレクション情報を表示
                st.write(f"利用可能なコレクション: {len(collections)}個")

                # 各コレクションの情報を表示
                collection_infos = []
                for col_name in collections:
                    try:
                        # 現在のコレクションを一時的に変更
                        original_collection = _qdrant_manager.collection_name
                        _qdrant_manager.collection_name = col_name

                        # 文書数を取得
                        doc_count = _qdrant_manager.count_documents()

                        # コレクションに関する情報を収集
                        collection_infos.append({
                            "name": col_name,
                            "doc_count": doc_count,
                            "is_current": col_name == original_collection
                        })

                        # 元のコレクション名に戻す
                        _qdrant_manager.collection_name = original_collection
                    except Exception as e:
                        logger.error(f"コレクション情報取得エラー ({col_name}): {str(e)}")
                        collection_infos.append({
                            "name": col_name,
                            "doc_count": "エラー",
                            "is_current": col_name == _qdrant_manager.collection_name
                        })

                # 表形式で表示
                df_collections = pd.DataFrame(collection_infos)
                st.dataframe(
                    df_collections,
                    column_config={
                        "name": "コレクション名",
                        "doc_count": "文書数",
                        "is_current": "現在使用中"
                    },
                    hide_index=True
                )

                # コレクションの選択（固定キーを使用）
                selected_collection = st.selectbox(
                    "削除するコレクションを選択",
                    options=collections,
                    help="選択したコレクションを完全に削除します。この操作は元に戻せません。",
                    key="collection_select_delete"
                )
                
                # 選択したコレクションをセッション状態に保存（削除確認時に使用）
                if "selected_collection_to_delete" not in st.session_state:
                    st.session_state.selected_collection_to_delete = None
                    
                if selected_collection:
                    st.session_state.selected_collection_to_delete = selected_collection

                # 削除ボタン
                delete_cols = st.columns([3, 3, 3])
                with delete_cols[1]:
                    delete_collection_pressed = st.button(
                        "削除実行",
                        key="delete_collection_button",
                        type="primary",
                        use_container_width=True
                    )
                    
                # 削除確認状態の管理
                if "delete_collection_confirmation_state" not in st.session_state:
                    st.session_state.delete_collection_confirmation_state = False
                
                # 削除実行ボタンが押されたら確認状態をONに
                if delete_collection_pressed and st.session_state.selected_collection_to_delete:
                    st.session_state.delete_collection_confirmation_state = True

                # 削除実行
                if st.session_state.delete_collection_confirmation_state and st.session_state.selected_collection_to_delete:
                    selected_collection_to_delete = st.session_state.selected_collection_to_delete
                    
                    # デフォルトコレクションの削除を防止
                    if selected_collection_to_delete == "default":
                        st.error("デフォルトコレクションは削除できません。")
                        st.session_state.delete_collection_confirmation_state = False
                    else:
                        # 確認ダイアログ
                        confirm = st.warning(
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
                                    # コレクションを削除
                                    _qdrant_manager.delete_collection(selected_collection_to_delete)
                                    
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


# 単独動作用
if __name__ == "__main__":
    import os
    import logging
    from logger import get_logger
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # https://discuss.streamlit.io/t/message-error-about-torch/90886/9
    # RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
    import torch
    torch.classes.__path__ = []

    st.set_page_config(page_title="データベース", layout="wide")
    # ロガーの初期化
    LOGGER = get_logger(log_dir="logs", log_level=logging.INFO)
    LOGGER.info("単独データベースアプリケーションを起動しました")

    SUPPORT_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']
    # コンポーネントの表示
    show_database_component(logger=LOGGER, extensions=SUPPORT_EXTENSIONS)
