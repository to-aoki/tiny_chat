import json
from typing import Dict, List
import functools
import urllib.parse
import webbrowser
import io  # CSVデータを文字列として扱うために追加

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
        collection_name: 検索対象のコレクション名

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
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            top_k = st.slider("最大検索件数", min_value=1, max_value=10000, value=10)

        with col2:
            score_threshold = st.slider("スコアしきい値", min_value=0., max_value=1., value=0.2)

        with col3:
            # 利用可能なコレクション一覧を取得
            available_collections = qdrant_manager.get_collections()
            # コレクション_descriptionsは除外
            available_collections = [c for c in available_collections if c != Collection.STORED_COLLECTION_NAME]
            
            # 現在のコレクション名をデフォルト選択に
            current_collection = qdrant_manager.collection_name
            default_idx = available_collections.index(current_collection) if current_collection in available_collections else 0
            
            selected_collection = st.selectbox(
                "コレクションを選択",
                options=available_collections,
                index=default_idx,
                key="collection_selectbox"
            )

        with col4:
            # 選択されたコレクションに基づいてソースを取得
            sources = qdrant_manager.get_sources(collection_name=selected_collection)
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
            # 選択されたコレクション名を使用して検索
            st.session_state.search_results = search_documents(
                query, qdrant_manager, top_k=top_k, filter_params_str=filter_params_str,
                score_threshold=score_threshold, collection_name=selected_collection)

    # 結果の表示
    if st.session_state.search_results:
        results = st.session_state.search_results
        result_count = len(results)
        st.success(f"{result_count}件の結果が見つかりました")

        # 検索結果からDataFrameを作成
        csv_data = []
        for i, result in enumerate(results):
            # 各結果から必要な情報を抽出
            row = {
                "index": i + 1, # 表示上のインデックス
                "score": result.score,
                "source": str(result.payload['source']),
                "page": str(result.payload['page']),
                "file_type": str(result.payload['file_type']),
                "text": str(result.payload['text'][:500]),
            }
            csv_data.append(row)

        # DataFrameに変換
        df = pd.DataFrame(csv_data)
        cols = df.columns.tolist()
        # 指定した列をリストの最初に移動させるヘルパー関数
        def move_cols_to_front(df_cols, cols_to_move):
            new_cols = []
            for col in cols_to_move:
                if col in df_cols:
                    new_cols.append(col)
            remaining_cols = [col for col in df_cols if col not in new_cols]
            return new_cols + remaining_cols

        ordered_cols = move_cols_to_front(cols, ["index", "score", "source", "page", "file_type", "text"])
        df = df[ordered_cols]

        # DataFrameをCSV文字列に変換 (UTF-8エンコード)
        csv_string = df.to_csv(index=False).encode('utf-8')

        # ダウンロードボタンを設置
        st.download_button(
            label="検索結果をCSVでダウンロード",
            data=csv_string,
            file_name=f'search_results_{query}.csv',
            mime='text/csv',
            key='download_search_results_csv' # ボタンキー
        )

        if len(results) > 50:
            st.warning(f"上位50件を表示します")
            results = results[:50]

        for idx, result in enumerate(results): # results を直接ループ
            score = result.score
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            page_info = get_page_info_display(metadata)
            source_name = metadata.get('source', 'ドキュメント')
            text_content = result.payload.get("text", "")

            with st.expander(
                    f"#{idx + 1}: {source_name} {page_info} (スコア: {score:.4f})",
                    expanded=idx == 0): # 最初の結果を展開
                # メタデータをDataFrameとして表示
                # metadataディクショナリをそのままDataFrameの行データとして渡す
                st.dataframe(pd.DataFrame([metadata]), hide_index=True, use_container_width=True)

                # ソースファイルへのリンク/ボタン
                if source_name: # source_nameが空文字列でないことを確認
                    if not source_name.startswith(('http://', 'https://')):
                        # ローカルファイルパスの場合
                        if st.button(f"{source_name}",
                                     key=f"open_ref_local_{idx}", use_container_width=True): # キーはユニークに
                            try:
                                webbrowser.open(source_name)
                            except Exception as e:
                                st.error(f"ファイルを開けませんでした: {str(e)}")
                    else:
                        # URLの場合
                        st.markdown(f"[{source_name}]({urllib.parse.quote(source_name, safe=':/')})")

                # テキスト表示（長い場合は省略）
                st.text(text_content[:500] + "..." if len(text_content) > 500 else text_content)
    else:
        # 検索ボタンが押されたが結果が0件の場合もこのブロックに入る
        if st.session_state.run_search or search_pressed: # 検索実行試みたが結果が0件の場合
             st.info("一致する検索結果は見つかりませんでした。")
        else: # 検索前の初期状態
             st.info("検索条件を入力してください。")