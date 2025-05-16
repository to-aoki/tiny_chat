import os

from typing import List, Dict, Tuple
import tempfile
import functools

import streamlit as st
import pandas as pd

from tiny_chat.utils.file_processor import process_file, URIProcessor
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory


SUPPORT_EXTENSIONS = ['.pdf', '.xlsx', '.docx', '.pptx', '.txt', '.csv', '.json', '.md', '.html', '.htm']


@functools.lru_cache(maxsize=16)
def get_extensions_without_dot(extensions_tuple):
    """拡張子タプルからドットを除去して返す（キャッシュ機能付き）"""
    return [ext.lstrip(".") for ext in extensions_tuple]


def convert_extensions(extensions_list):
    """リストをタプルに変換してキャッシュ可能な関数に渡す"""
    return get_extensions_without_dot(tuple(extensions_list))


def process_directory(directory_path: str,
    extensions: List[str] = None,
    support_extensions: List[str] = SUPPORT_EXTENSIONS,
    include_subdirs: bool = True
) -> List[Tuple[List[str], Dict]]:
    """
    ディレクトリ内のファイルを処理します

    Args:
        directory_path: 処理するディレクトリのパス
        extensions: 処理対象のファイル拡張子リスト (None の場合はすべてのサポートされる形式)
        include_subdirs: サブディレクトリも走査するかどうか (True: 走査する, False: 走査しない)

    Returns:
        [(text_array, metadata), ...]: 抽出されたテキスト配列とメタデータのリスト
    """
    results = []

    if extensions is None:
        extensions = support_extensions

    # セットによる高速ルックアップ
    extensions_set = set(extensions)

    # ファイルを検索（サブディレクトリ走査オプションに基づいて処理）
    if include_subdirs:
        # サブディレクトリを含めて走査
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext in extensions_set:
                    text = None
                    try:
                        text, metadata = process_file(file_path)
                    except Exception as e:
                        st.error(str(e))
                    if text:
                        # 相対パスをメタデータに追加
                        rel_path = os.path.relpath(file_path, directory_path)
                        metadata["rel_path"] = rel_path
                        results.append((text, metadata))
    else:
        # 指定ディレクトリのみ走査（サブディレクトリは含めない）
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        for file in files:
            file_path = os.path.join(directory_path, file)
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in extensions_set:
                text = None
                try:
                    text, metadata = process_file(file_path)
                except Exception as e:
                    st.error(str(e))
                if text:
                    # 相対パスをメタデータに追加
                    metadata["rel_path"] = file
                    results.append((text, metadata))

    return results


def add_files_to_qdrant(
    texts: List[List[str]], metadatas: List[Dict],
    qdrant_manager, collection_name: str = None, strategy=None, chunk_size=None, chunk_overlap=None
) -> List[str]:
    """
    テキストとメタデータをQdrantに追加します
    同じソース（ファイル名）が既に存在する場合は、削除してから追加します
    注意: この関数を呼び出す前に、QdrantManagerを初期化する必要があります

    Args:
        texts: テキスト配列のリスト (is_page=True により、各ファイルのテキストは文字列の配列)
        metadatas: メタデータのリスト

    Returns:
        added_ids: 追加されたドキュメントのIDリスト
    """

    # ソース（ファイル名）の一覧を取得
    sources_to_add = set()
    for metadata in metadatas:
        if "source" in metadata:
            sources_to_add.add(metadata["source"])

    # 既存のソースと照合し、重複があれば削除
    existing_sources = qdrant_manager.get_sources(collection_name=collection_name)
    for source in sources_to_add:
        if source in existing_sources:
            # ソースに関連するデータを削除
            # ソース名を配列として渡す（単一でも配列として扱う）
            filter_params = {"source": [source]}
            qdrant_manager.delete_by_filter(filter_params)

    all_texts = []
    all_metadatas = []

    # ファイルごとにテキストとメタデータを処理
    for i, text_array in enumerate(texts):
        base_metadata = metadatas[i].copy()
        for page_index, page_text in enumerate(text_array):
            all_texts.append(page_text)
            page_metadata = base_metadata.copy()
            page_metadata["page"] = page_index + 1  # 配列の添字 + 1 をページとして設定
            all_metadatas.append(page_metadata)

    # Qdrantに追加
    added_ids = qdrant_manager.add_documents(
        all_texts, all_metadatas, collection_name=collection_name,
        strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return added_ids


def show_registration(
    qdrant_manager,
    extensions: List[str] = SUPPORT_EXTENSIONS
):

    # コレクション名の入力
    collection_name = st.text_input(
        "コレクション名",
        value=qdrant_manager.collection_name,
        help="データを登録するコレクション名を指定します。新しいコレクション名を指定すると自動的に作成されます。"
    )

    # 登録方法の選択
    register_method = st.radio(
        "登録方法を選択",
        ["ファイルアップロード", "ディレクトリ指定", "URI指定"]
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
            type=convert_extensions(extensions)
        )

        if uploaded_files:
            if st.button("選択したファイルを登録", type="primary"):
                with st.spinner("ファイルを処理中..."):
                    # 一時ファイルとして保存してから処理
                    texts = []
                    metadatas = []

                    progress_bar = st.progress(0)
                    total_files = len(uploaded_files)

                    for i, uploaded_file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(
                                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                            temp_file.write(uploaded_file.getbuffer())
                            temp_path = temp_file.name

                        text = None
                        try:
                            text, metadata = process_file(temp_path)
                        except Exception as e:
                            st.error(str(e))

                        # カスタムソースパスを設定（指定があれば）
                        if text and source_base_dir:
                            custom_source_path = os.path.join(source_base_dir, uploaded_file.name)
                            metadata["source"] = custom_source_path

                        if text:
                            texts.append(text)
                            metadatas.append(metadata)

                        # 一時ファイルを削除
                        os.unlink(temp_path)

                        # 進捗を更新
                        progress_bar.progress((i + 1) / total_files)

                    if texts:
                        collection = Collection.load(
                            collection_name=collection_name, qdrant_manager=qdrant_manager)
                        if collection is None:
                            collection = Collection(collection_name)
                            collection.save(qdrant_manager)

                        # Qdrantに追加
                        added_ids = add_files_to_qdrant(
                            texts, metadatas, qdrant_manager, collection_name=collection_name,
                            strategy=RagStrategyFactory.get_strategy(collection.rag_strategy, collection.use_gpu),
                            chunk_size=collection.chunk_size, chunk_overlap=collection.chunk_overlap
                        )

                        st.success(
                            f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")

                        metadata_df = pd.DataFrame(metadatas)
                        st.dataframe(metadata_df, use_container_width=True)

                    else:
                        st.warning("登録できるドキュメントがありませんでした")

    elif register_method == "ディレクトリ指定":
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
                help="ファイルの「source」として使用するWebパス接頭辞を指定します。"
                     "入力ディレクトリの末尾名と相対パスがこの接頭辞に追加されます。例: http://example.com/path/to/"
            )

        # 処理対象のファイル拡張子選択
        selected_extensions = st.multiselect(
            "処理対象の拡張子を選択",
            extensions,
            default=extensions
        )
        
        # サブディレクトリ走査オプション（ラジオボタン）
        include_subdirs = st.radio(
            "サブディレクトリの走査",
            ["含めない", "含める"],
            index=0,
            help="サブディレクトリ内のファイルも処理対象に含めるかどうかを選択します。"
        )
        # ラジオボタンの選択をbool値に変換
        include_subdirs_bool = (include_subdirs == "含める")

        if directory_path and os.path.isdir(directory_path):
            if st.button("ディレクトリ内のファイルを登録", type="primary"):
                with st.spinner(f"ディレクトリを処理中: {directory_path}"):
                    results = process_directory(
                        directory_path, 
                        selected_extensions, 
                        support_extensions=extensions,
                        include_subdirs=include_subdirs_bool
                    )

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
                        collection = Collection.load(
                            collection_name=collection_name, qdrant_manager=qdrant_manager)
                        if collection is None:
                            collection = Collection(collection_name)
                            collection.save(qdrant_manager)

                        # Qdrantに追加
                        added_ids = add_files_to_qdrant(
                            texts, metadatas, qdrant_manager, collection_name=collection_name,
                            strategy=RagStrategyFactory.get_strategy(collection.rag_strategy, collection.use_gpu),
                            chunk_size=collection.chunk_size, chunk_overlap=collection.chunk_overlap
                        )

                        # 結果表示
                        st.success(
                            f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")

                        metadata_df = pd.DataFrame(metadatas)
                        st.dataframe(metadata_df, use_container_width=True)

                    else:
                        st.warning("指定されたディレクトリに登録可能なファイルが見つかりませんでした")
        elif directory_path:
            st.error("指定されたディレクトリが存在しません")
            
    elif register_method == "URI指定":
        # URIの入力インターフェース
        uri_input = st.text_area(
            "URIを入力",
            "",
            height=150,
            help="登録するコンテンツのURIを入力してください。複数のURIを入力する場合は、1行に1つずつ入力してください。"
        )
        
        # 処理の実行
        if uri_input and st.button("URIからコンテンツを登録", type="primary"):
            # URIのリストを作成（空行を除去）
            uris = [uri.strip() for uri in uri_input.split('\n') if uri.strip()]
            
            if uris:
                with st.spinner("URIからコンテンツを取得中..."):
                    texts = []
                    metadatas = []
                    
                    progress_bar = st.progress(0)
                    total_uris = len(uris)
                    
                    for i, uri in enumerate(uris):
                        try:
                            # URIからコンテンツを取得（is_page=Trueでページ単位でテキストを取得）
                            text, meta_data = URIProcessor.process_uri(uri, is_page=True)
                            print("meta", meta_data)
                            if text:
                                # テキストが文字列の場合は配列に変換
                                if isinstance(text, str):
                                    text = [text]

                                texts.append(text)
                                meta_data["source"] = uri
                                metadatas.append(meta_data)
                            else:
                                st.warning(f"URI処理エラー: {uri}, {meta_data}")
                        except Exception as e:
                            st.error(f"URI処理エラー: {uri}, {str(e)}")
                        
                        # 進捗を更新
                        progress_bar.progress((i + 1) / total_uris)
                    
                    if texts:
                        # コレクション名を設定して処理
                        collection = Collection.load(
                            collection_name=collection_name, qdrant_manager=qdrant_manager)
                        if collection is None:
                            collection = Collection(collection_name)
                            collection.save(qdrant_manager)

                        print(texts, metadatas)

                        # Qdrantに追加
                        added_ids = add_files_to_qdrant(
                            texts, metadatas, qdrant_manager, collection_name=collection_name,
                            strategy=RagStrategyFactory.get_strategy(collection.rag_strategy, collection.use_gpu),
                            chunk_size=collection.chunk_size, chunk_overlap=collection.chunk_overlap
                        )
                        
                        # 結果表示
                        st.success(f"{len(added_ids)}件のドキュメントを「{collection_name}」コレクションに登録しました")
                        
                        metadata_df = pd.DataFrame(metadatas)
                        st.dataframe(metadata_df, use_container_width=True)
                    else:
                        st.warning("登録できるコンテンツがありませんでした")
