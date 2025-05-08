from typing import List, Dict, Sequence
import numpy as np
import datasets


def calc_one_ndcg(
    relevant_doc_id: str,
    top_hits: List[str],
    ks: Sequence[int]
) -> Dict[int, float]:
    """
    複数のkに対してNDCG@kを計算する関数 (関連度はバイナリ、正解は1つ)。

    Args:
        relevant_doc_id: そのクエリに対する正解の文書ID。
        top_hits: モデルが出力したランキング上位の文書IDリスト。
        ks: 評価対象とするkの値のリストまたはシーケンス (例: [1, 3, 5, 10])。
            リスト内のkは正の整数である必要があります。

    Returns:
        dict: キーがk、値がNDCG@kスコアの辞書。
              例: {1: 0.0, 3: 0.5, 5: 0.5, 10: 0.5}
              もし relevant_doc_id が空文字列などの無効な値の場合や、
              ks が空の場合、空の辞書を返します。
              もし relevant_doc_id が top_hits の中に ks の最大値まで
              存在しない場合、全ての k のスコアは 0.0 になります。
    """
    ndcg_scores = {k: 0.0 for k in ks}

    if not relevant_doc_id or not ks:
        return ndcg_scores

    max_k = 0
    valid_ks = [k for k in ks if k > 0] # 0以下のkは無視
    if not valid_ks:
        return {k: 0.0 for k in ks} # 有効なkがなければ全て0
    max_k = max(valid_ks)

    dcg_value = 0.0
    found_rank = -1

    # 最大のkまで線形探索して、正解文書のランクを探す
    for rank, hit in enumerate(top_hits[:max_k], start=1):
        if hit == relevant_doc_id:
            # 正解が見つかったランクでDCGを計算
            dcg_value = 1.0 / np.log2(rank + 1)
            found_rank = rank
            break

    # 正解が見つかった場合、各kについてスコアを更新
    if found_rank != -1:
        for k in valid_ks:
            # 正解ランクが現在のk以下であれば、計算したDCG値を採用
            if found_rank <= k:
                ndcg_scores[k] = dcg_value
            # 正解ランクがkより大きい場合は、初期値の0.0のまま

    return ndcg_scores


def load_test_JaGovFags22K(seed=42, samples=-1):

    # HFよりダウンロード
    dataset = datasets.load_dataset("matsuxr/JaGovFaqs-22k", trust_remote_code=True)

    def preprocess(example: dict, idx: int) -> dict:
        example["idx"] = idx + 1
        example["Question"] = example["Question"].strip()
        example["Answer"] = example["Answer"].strip()
        return example

    dataset = dataset.map(preprocess, with_indices=True, num_proc=4)
    queries = dataset.select_columns(["Question", "idx"]).rename_columns(
        {"Question": "query", "idx": "docid"},
    )

    # query抽出
    # train:7 : dev:1.5 : test:1.5
    queries.shuffle(seed=seed)
    queries = queries["train"].train_test_split(test_size=0.3, seed=seed)
    devtest = queries.pop("test").train_test_split(
        test_size=1 - 0.15 / 0.3,
        seed=seed
    )
    queries["test"] = devtest.pop("test")
    queries = queries["test"]
    documents = dataset.select_columns(["idx", "Answer"]).rename_columns(
        {"idx": "docid", "Answer": "text"},
    )['train']

    if samples > 1: # samples が 0 以下なら全件使う
        num_test_samples = len(queries)
        actual_samples = min(samples, num_test_samples)
        if actual_samples < num_test_samples:
            sampled_indices = range(actual_samples)
            queries = queries.select(sampled_indices)

        doc_ids_raw = queries['docid']
        flat_doc_ids = []
        for item in doc_ids_raw:
            if isinstance(item, list):
                if len(item) == 1 and isinstance(item[0], int):
                    flat_doc_ids.append(item[0])
                else:
                    print(f"Warning: Unexpected list format in docid: {item}")
            elif isinstance(item, int):
                flat_doc_ids.append(item)
            else:
                 print(f"Warning: Unexpected type in docid: {type(item)}, value: {item}")

        sampled_doc_ids = set(flat_doc_ids)
        def filter_batch(batch):
            return [docid in sampled_doc_ids for docid in batch['docid']]

        documents = documents.filter(
            filter_batch,
            batched=True, # batched=True を使用
            num_proc=4
        )

    return queries, documents


if __name__ == "__main__":
    import argparse
    import time

    from tiny_chat.database.qdrant.qdrant_manager import QdrantManager

    import argparse
    import time

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="ruri_xsmall"
    )

    parser.add_argument(
        "--use_gpu", "-g",
        action='store_true',
    )

    parser.add_argument(
        "--file_path", "-f",
        type=str,
        default="./qdrant_test_data"
    )

    parser.add_argument(
        "--server_uri", "-uri",
        type=str,
        default=None
    )

    parser.add_argument(
        "--reuse_db", "-r",
        action='store_true',
    )

    parser.add_argument(
        "--samples", "-samples",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.server_uri is None:
        manager = QdrantManager(
            collection_name="test",
            rag_strategy=args.strategy, use_gpu=args.use_gpu, file_path=args.file_path)
    else:
        # 遅い (20,000件警告）
        manager = QdrantManager(
            collection_name="test",
            rag_strategy=args.strategy, use_gpu=args.use_gpu, server_url=args.server_uri)

    if not args.reuse_db:
        manager.client.delete_collection(collection_name="test")
        manager.ensure_collection_exists(collection_name="test")

    queries, documents = load_test_JaGovFags22K(samples=args.samples)

    text_documents = [d["text"] for d in documents]
    metadata_list = [{"docid": d["docid"]} for d in documents]

    if not args.reuse_db:
        print("=== 文書の追加 ===")
        start_time = time.time()
        ids = manager.add_documents(text_documents, metadata_list)
        print(f"ドキュメント追加時間: {time.time() - start_time:.4f}秒")

    print(f"コレクション内の文書数: {manager.count_documents()}")

    from collections import defaultdict

    total_ndcg_scores = defaultdict(float)

    ks = [1, 3, 5, 10]

    num_queries = len(queries['query'])

    from tqdm import tqdm

    query_processor = None
    if args.query is not None:
        from tiny_chat.api.api_util import get_llm_api
        llm_api, chat_config = get_llm_api()
        if args.query == 'hyde':
            from tiny_chat.utils.query_preprocessor import HypotheticalDocument
            query_processor = HypotheticalDocument(
                openai_client=llm_api,
                model_name=chat_config.selected_model,
                temperature=chat_config.temperature,
                top_p=chat_config.top_p,
                meta_prompt=chat_config.meta_prompt
            )
        elif args.query == 'back':
            from tiny_chat.utils.query_preprocessor import StepBackQuery
            query_processor = StepBackQuery(
                openai_client=llm_api,
                model_name=chat_config.selected_model,
                temperature=chat_config.temperature,
                top_p=chat_config.top_p,
                meta_prompt=chat_config.meta_prompt
            )
        else:
            raise ValueError('not found :' + args.query)

    for query_text, doc_id in tqdm(zip(queries['query'], queries['docid']), total=num_queries,
                                   desc="Processing queries", unit="query"):

        results = manager.query_points(
            query_text, top_k=max(ks), score_threshold=0., query_processor=query_processor)
        hit_doc_ids = [result.payload.get('docid') for result in results]
        ndcg_scores_for_query = calc_one_ndcg(relevant_doc_id=doc_id, top_hits=hit_doc_ids, ks=ks)
        for k, score in ndcg_scores_for_query.items():
            total_ndcg_scores[k] += score

    average_ndcg_scores = {}
    for k in ks:
        average_ndcg_scores[k] = total_ndcg_scores[k] / len(queries)

    print(f"\nEvaluated {len(queries)} queries.")
    print("Average NDCG Scores:")
    for k, avg_score in average_ndcg_scores.items():
        print(f"  NDCG@{k}: {avg_score:.4f}")

# $ python evaluate/eval_ja_gov_fags22k.py -uri dns://localhost:6334 -g -s ruri_xsmall
# === 文書の追加 ===
#
# ドキュメント追加時間: 682.0819秒
# コレクション内の文書数: 22892
# Processing queries: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3420/3420 [01:20<00:00, 42.34query/s]
#
# Evaluated 3420 queries.
# Average NDCG Scores:
#   NDCG@1: 0.5561
#   NDCG@3: 0.6604
#   NDCG@5: 0.6834
#   NDCG@10: 0.7033

# $ python evaluate/eval_ja_gov_fags22k.py -uri dns://localhost:6334 -g -s ruri_small
# === 文書の追加 ===
# ドキュメント追加時間: 1114.8480秒
# コレクション内の文書数: 22892
# Processing queries: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3420/3420 [01:37<00:00, 34.93query/s]
#
# Evaluated 3420 queries.
# Average NDCG Scores:
#   NDCG@1: 0.5836
#   NDCG@3: 0.6900
#   NDCG@5: 0.7111
#   NDCG@10: 0.7290

# $ python evaluate/eval_ja_gov_fags22k.py -uri dns://localhost:6334 -g -s ruri_base
# === 文書の追加 ===
# ドキュメント追加時間: 2097.0727秒
# コレクション内の文書数: 22892
# Processing queries: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3420/3420 [02:12<00:00, 25.86query/s]
#
# Evaluated 3420 queries.
# Average NDCG Scores:
#   NDCG@1: 0.6211
#   NDCG@3: 0.7256
#   NDCG@5: 0.7454
#   NDCG@10: 0.7599

# $ python evaluate/eval_ja_gov_fags22k.py -uri dns://localhost:6334 -g -s ruri_large
# === 文書の追加 ===
# ドキュメント追加時間: 4631.5279秒
# コレクション内の文書数: 22892
# Processing queries: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3420/3420 [02:12<00:00, 25.86query/s]
#
# Evaluated 3420 queries.
# Average NDCG Scores:
#   NDCG@1: 0.6088
#   NDCG@3: 0.7173
#   NDCG@5: 0.7392
#   NDCG@10: 0.7530

# $  python eval_ja_gov_fags22k.py -uri dns://localhost:6334 -s ruri_xsmall-open-vino
# === 文書の追加 ===
# ドキュメント追加時間: 804.5392秒
# コレクション内の文書数: 22892
# Processing queries: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3420/3420 [01:07<00:00, 50.51query/s]
#
# Evaluated 3420 queries.
# Average NDCG Scores:
#   NDCG@1: 0.5491
#   NDCG@3: 0.6521
#   NDCG@5: 0.6745
#   NDCG@10: 0.6941