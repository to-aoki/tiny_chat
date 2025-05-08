## [JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k) qdrant search (stored int8) 
| pattern                                                          | NDCG@1     | NDCG@3      | NDCG@5       | NDCG@10    | 
|:-----------------------------------------------------------------|:-----------|:------------|:-------------|:-----------|
| bm25 (sudachi B)                                                 | 0.4383     | 0.5412      | 0.5637       | 0.5857     |
| bm42 (ruri-v2-small/sudachi B)                                   | 0.3722     | 0.4718      | 0.5011       | 0.5279     |
| cl-nagoya/ruri-v3-30m                                            | 0.5561     | 0.6604      | 0.6834       | 0.7033     |
| cl-nagoya/ruri-v3-30m (openvino int8)                            | 0.5491     | 0.6521      | 0.6745       | 0.6941     |
| cl-nagoya/ruri-v3-70m                                            | 0.5836     | 0.6900      | 0.7111       | 0.7290     |
| cl-nagoya/ruri-v3-130m                                           | 0.6211     | 0.7256      | 0.7454       | 0.7599     |
| cl-nagoya/ruri-v3-310m                                           | 0.6088     | 0.7173      | 0.7392       | 0.7530     |
| cl-nagoya/ruri-v3-30m + hotchpotch/japanese-reranker-tiny-v2     | 0.5307     | 0.6441      | 0.6696       | 0.6906     |
| cl-nagoya/ruri-v3-30m + hotchpotch/japanese-reranker-xsmall-v2   | 0.6070     | 0.7119      | 0.7316       | 0.7439     |
| cl-nagoya/ruri-v3-130m + cl-nagoya/ruri-v3-reranker-310m         | **0.6482** | **0.7515**  | **0.7726**   | 0.7828     |
| hotchpotch/japanese-splade-v2                                    | 0.5646     | 0.6623      | 0.6858       | 0.7038     |
| hotchpotch/static-embedding-japanese (512)                       | 0.3789     | 0.4778      | 0.5074       | 0.5305     |
| RFF (bm25 + ruri-v3-30m)                                         | 0.5389     | 0.6498      | 0.6746       | 0.6949     |
| RFF (bm25 + ruri-v3-130m)                                        | 0.5728     | 0.6876      | 0.7113       | 0.7292     |
| RFF (bm25 + static-embedding-japanese (512))                     | 0.4354     | 0.5543      | 0.5800       | 0.5998     |
| RFF (japanese-splade-v2  + ruri-v3-30m)                          | 0.5819     | 0.6949      | 0.7168       | 0.7330     |
| RFF (japanese-splade-v2  + ruri-v3-130m)                         | 0.6108     | 0.7270      | 0.7467       | 0.7616     |
| RFF (japanese-splade-v2  + ruri-v3-130m) + ruri-v3-reranker-310m | 0.6447     | 0.7505      | 0.7713       | **0.7835** |
