import json
from typing import Dict, List
import functools
import urllib.parse
import webbrowser

import pandas as pd
import streamlit as st
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory


def get_page_info_display(metadata: Dict) -> str:
    """メタデータからページ情報表示文字列を生成"""
    if 'page' not in metadata:
        return ""

    file_type = metadata.get('file_type', '').lower()
    page_num = metadata['page']

    if file_type == 'pdf':
        return f"(ページ: {page_num})"
    elif file_type in ['xlsx', 'xls']:
        return f"(シート: {page_num})"
    elif file_type == 'docx':
        return f"(段落: {page_num})"
    elif file_type == 'pptx':
        return f"(スライド: {page_num})"
    else:
        return f"(記載箇所: {page_num})"


@functools.lru_cache(maxsize=32)
def search_documents(
        query: str, qdrant_manager, top_k: int = None, filter_params_str: str = None,
        score_threshold=None, collection_name: str = None) -> List:
    """
    ドキュメントを検索します（キャッシュ機能付き）
    注意: この関数を呼び出す前に、QdrantManagerを初期化する必要があります

    Args:
        query: 検索クエリ
        top_k: 返す結果の数
        filter_params_str: 検索フィルタの文字列表現（キャッシュキーとして使用）
        score_threshold: 最小スコアしきい値

    Returns:
        results: 検索結果のリスト
    """
    # 文字列からフィルタを復元（もしあれば）
    filter_params = None
    if filter_params_str:
        import json
        filter_params = json.loads(filter_params_str)

    if collection_name is None:
        collection_name = qdrant_manager.collection_name

    collection = Collection.load(
        collection_name=collection_name, qdrant_manager=qdrant_manager)

    results = qdrant_manager.query_points(
        query, top_k=top_k, filter_params=filter_params, score_threshold=score_threshold,
        collection_name=collection_name,
        strategy=RagStrategyFactory.get_strategy(strategy_name=collection.rag_strategy, use_gpu=collection.use_gpu)
    )

    return results


def show_search_component(qdrant_manager, logger=None):
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    def search_on_enter():
        # テキスト入力からクエリを取得し、検索実行のフラグを立てる
        st.session_state.run_search = True

    # 検索フィールド
    query = st.text_input(
        "検索文字列", "", key="search_query_input", on_change=search_on_enter)

    # 詳細設定のエクスパンダー
    with st.expander("詳細設定", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider("表示件数", min_value=1, max_value=50, value=10)

        with col2:
            # 使用可能なソースを取得（常に最新の状態を取得）
            # 現在のコレクション名を明示的に使用
            current_collection = qdrant_manager.collection_name
            sources = qdrant_manager.get_sources(collection_name=current_collection)
            selected_sources = st.multiselect(
                "ソースでフィルタ",
                options=sources,
                key="sources_multiselect_filter"
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
        filter_params_str = None
        if selected_sources:
            # 複数のソースを配列として設定
            filter_params = {"source": selected_sources}
            # キャッシュ用に文字列化
            filter_params_str = json.dumps(filter_params)

        with st.spinner("検索中..."):
            # コレクション名を明示的に渡して検索（キャッシュキーに含める）
            current_collection = qdrant_manager.collection_name
            st.session_state.search_results = search_documents(
                query, qdrant_manager, top_k=top_k, filter_params_str=filter_params_str, 
                score_threshold=0., collection_name=current_collection)

    # 結果の表示
    if st.session_state.search_results:
        results = st.session_state.search_results
        result_count = len(results)
        st.success(f"{result_count}件の結果が見つかりました")

        result_grid = []
        for i, result in enumerate(results):
            score = result.score
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            page_info = get_page_info_display(metadata)

            result_grid.append({
                "index": i + 1,
                "source": metadata.get('source', 'ドキュメント'),
                "page_info": page_info,
                "score": f"{score:.4f}",
                "metadata": metadata,
                "text": result.payload.get("text", "")
            })

        # 一列表示に変更
        for idx, item in enumerate(result_grid):
            with st.expander(
                    f"#{item['index']}: {item['source']} {item['page_info']} (スコア: {item['score']})",
                    expanded=idx == 0):
                # メタデータをDataFrameとして表示
                st.dataframe(pd.DataFrame([item['metadata']]), hide_index=True, use_container_width=True)

                # ソースファイルへのリンク
                if 'source' in item['metadata'] and item['metadata']['source']:
                    source_path = item['metadata']['source']
                    if not source_path.startswith(('http://', 'https://')):
                        if st.button(f"{source_path}",
                                     key=f"open_ref_{source_path}_{idx}", use_container_width=True):
                            try:
                                webbrowser.open(source_path)
                            except Exception as e:
                                st.error(f"ファイルを開けませんでした: {str(e)}")
                    else:
                        st.markdown(f"[{source_path}]({urllib.parse.quote(source_path, safe=':/')})")

                # テキスト表示（長い場合は省略）
                text = item['text']
                st.text(text[:500] + "..." if len(text) > 500 else text)
    else:
        st.info("検索結果はありません")
