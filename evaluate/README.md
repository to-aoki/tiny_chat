## [JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k) qdrant search (stored int8) 
| pattern                                      | NDCG@1  | NDCG@3 | NDCG@5 | NDCG@10    | 
|:---------------------------------------------|:--------|:-------|:-------|:-----------|
| bm25 (sudachi B)                             | 0.4383  | 0.5412 | 0.5637 | 0.5857     |
| bm42 (ruri-v2-small/sudachi B)               | 0.3722  | 0.4718 | 0.5011 | 0.5279     |
| cl-nagoya/ruri-v3-30m                        | 0.5561  | 0.6604 | 0.6834 | 0.7033     |
| cl-nagoya/ruri-v3-70m                        | 0.5836  | 0.6900 | 0.7111 | 0.7290     |
| cl-nagoya/ruri-v3-130m                       | 0.6211  | 0.7256 | 0.7454 | **0.7599** |
| cl-nagoya/ruri-v3-310m                       | 0.6088  | 0.7173 | 0.7392 | 0.7530     |
| hotchpotch/japanese-splade-v2                | 0.5646  | 0.6623 | 0.6858 | 0.7038     |
| hotchpotch/static-embedding-japanese (512)   | 0.3789  | 0.4778 | 0.5074 | 0.5305     |
| RFF (bm25 + ruri-v3-30m)                     | 0.5389  | 0.6498 | 0.6746 | 0.6949     |
| RFF (bm25 + static-embedding-japanese (512)) | 0.4354  | 0.5543 | 0.5800 | 0.5998     |
