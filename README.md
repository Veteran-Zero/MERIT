# MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing

[![arXiv](https://img.shields.io/badge/arXiv-2603.22289-b31b1b.svg)](https://arxiv.org/abs/2603.22289)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the ACM paper:

> **MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing**  
> Anonymous Authors  
> *Under review*

---

## Overview

Knowledge Tracing (KT) models students' evolving knowledge states to predict future performance.
MERIT is a **training-free** framework that replaces expensive LLM fine-tuning with a structured,
interpretable memory bank.

| Stage | Name | What it does |
|-------|------|-------------|
| 1 | Cognitive Schema Discovery | Semantic denoising → Qwen3 Embedding → UMAP + HDBSCAN (via BERTopic) clusters students into latent cognitive schemas |
| 2 | Interpretative Memory Bank Construction | Selects representative prototypes per cluster; uses Gemini-2.5-Pro offline to generate structured CoT annotations (knowledge state, key pattern, difficulty context, causal reasoning) |
| 3 | Hierarchical Cognitive Retrieval | Routes a test student to their schema; retrieves top-K paradigms via hybrid FAISS + BM25 search |
| 4 | Logic-Augmented Inference | Injects retrieved paradigms + explicit Spike/Crash rules into a frozen LLM prompt; outputs a calibrated correctness probability |

### Key Results

| Dataset | AUC | ACC | F1 |
|---------|-----|-----|----|
| ASSISTments 2009 | **0.8244** | 0.7554 | 0.8054 |
| ASSISTments 2012 | **0.7778** | 0.7441 | 0.8122 |
| Eedi | **0.7969** | 0.7603 | 0.8139 |
| BePKT | **0.8036** | 0.7825 | 0.5991 |

> Backbone: Gemini-2.5-Flash for inference, Gemini-2.5-Pro for annotation, Qwen3-Embedding-8B for vectors.

---

## Repository Structure

```
MERIT-repo/
├── configs/
│   └── config.py          # All paths, API keys, and hyperparameters
├── src/
│   ├── stage1_clustering.py   # Stage 1: Cognitive Schema Discovery
│   ├── stage2_annotation.py   # Stage 2: Memory Bank Construction
│   ├── stage3_inference.py    # Stage 3 & 4: Retrieval + Inference
│   └── ablation.py            # Ablation study runner
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/MERIT.git
cd MERIT
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
MERIT_BASE_DIR=.
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Embedding model (OpenAI-compatible; paper uses Qwen3-Embedding-8B)
MERIT_EMBEDDING_MODEL=text-embedding-3-large

# Offline annotation model (paper uses Gemini-2.5-Pro)
MERIT_ANNOTATION_MODEL=gemini-2.5-pro

# Online inference model (paper uses Gemini-2.5-Flash)
MERIT_INFERENCE_MODEL=gemini-2.5-flash
```

All settings can also be overridden directly in `configs/config.py`.

---

## Data Format

Each dataset split (`merged_train_valid.json`, `test.json`) is a JSON array where
each element contains:

```json
{
  "input": "The student's historical response sequence: ...\nthe next question: ...",
  "output": "True"
}
```

`output` is `"True"` (correct) or `"False"` (incorrect).

Preprocessed splits for ASSISTments 2009/2012, Eedi, and BePKT follow the
paper's 80/20 temporal split with a minimum sequence length of 5 and a maximum
of 50 steps.

Place your data under `data/` (configurable via `MERIT_BASE_DIR`):

```
data/
├── merged_train_valid.json   # training + validation split
└── test.json                 # test split
```

---

## Usage

Run the four pipeline stages in order.

### Stage 1 — Cognitive Schema Discovery

Clusters the training set into latent cognitive schemas using BERTopic.
Embeddings are cached to disk on the first run.

```bash
python src/stage1_clustering.py
```

Outputs:
- `data/merged_train_valid_clustered.json` — original data + `cluster_id` / `cluster_score`
- `data/cluster_metadata.json` — cluster keywords and representative samples
- `data/embeddings_cache.npy` — cached embedding vectors
- `models/bertopic_merit/` — saved BERTopic model

### Stage 2 — Interpretative Memory Bank Construction

Selects the top-100 representative sequences per cluster and uses an LLM to
generate structured pedagogical annotations offline.

```bash
python src/stage2_annotation.py
```

Output: `data/annotated_memory_bank.json`

The script supports **resuming** — if interrupted, it skips already-annotated
entries on restart.

### Stage 3 & 4 — Retrieval + Logic-Augmented Inference

Runs inference on the test set.

```bash
python src/stage3_inference.py
```

Output: `outputs/test_results.json`

Prints AUC, Accuracy, Precision, Recall, and F1 on completion.

### Ablation Study

```bash
# Options: FULL | NO_RETRIEVAL | NO_ROUTING | NO_TRACES | NO_RULES
ABLATION_MODE=NO_RULES python src/ablation.py
```

Output: `outputs/ablation_<mode>.json`

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K_PER_CLUSTER` | 100 | Prototype samples per cognitive schema for annotation |
| `TOP_K_RETRIEVAL` | 3 | Memory entries retrieved per test query |
| `ALPHA` | 0.7 | Dense score weight in hybrid retrieval |
| `BETA` | 0.3 | Sparse (BM25) score weight in hybrid retrieval |
| `SIMILARITY_THRESHOLD` | 0.4 | Minimum hybrid score to retain a candidate |
| `BERTOPIC_NR_TOPICS` | 20 | Number of cognitive schema clusters |

---

## Citation

```bibtex
@article{anonymous2026merit,
  title   = {MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing},
  author  = {Anonymous Authors},
  journal = {arXiv preprint arXiv:2603.22289},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
