# Preference-Optimized Retrieval and Ranking for Efficient Multimodal Recommendation

This repository provides the PyTorch implementation for [PRIME](https://dl.acm.org/doi/abs/10.1145/3711896.3737088). In this work, we introduce preference-optimized retrieval and ranking for efficient multimodal recommendation (PRIME). PRIME operates in two stages: (i) a lightweight retriever identifies potential candidate items; (ii) an LMM learns to rank the retrieved candidates with detailed user history and multimodal features (e.g., text and image attributes). These features are incorporated into a carefully designed prompt, facilitating finegrained transition patterns for user preference understanding. To optimize the inference efficiency of PRIME, we introduce verbalizerbased inference, which computes ranking scores for all candidate items in a single forward pass. Furthermore, we employ the LMM ranker to provide feedback on sampled candidate sets, enabling online preference optimization that refines the retriever model and improves the alignment between retrieval and ranking. As a result, PRIME can capture subtle user intentions and efficiently rank candidate items with minimal inference costs. 


## Train Retriever

Training and evaluation data is available at [here](https://github.com/jeykigung/VIP5), and the hyperparameter configuration can be adjusted in ```config.py```.

```bash
python train_retriever.py
```

## Train LMM Ranker

```bash
python train_ranker.py
```

## Optimizer Retriever

```bash
python optimizer_retriver.py
```