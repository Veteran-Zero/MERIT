"""
MERIT — Stage 1: Cognitive Schema Discovery

Encodes student interaction sequences using a semantic embedding model,
projects them into a lower-dimensional manifold with UMAP, and clusters
them with HDBSCAN via BERTopic.  The resulting cognitive schemas are saved
alongside the cluster-level keyword metadata used for downstream routing.

Usage:
    python src/stage1_clustering.py
"""
import os
import sys
import json
import asyncio

import numpy as np
from tqdm import tqdm
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from sklearn.feature_extraction.text import CountVectorizer
from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL,
    TRAIN_FILE, CLUSTERED_FILE, CLUSTER_META_FILE,
    EMBEDDING_CACHE_FILE, BERTOPIC_MODEL_PATH, IMAGE_DIR,
    BERTOPIC_NR_TOPICS, BERTOPIC_MIN_TOPIC,
    EMBEDDING_BATCH_SIZE, EMBEDDING_CONCURRENCY,
)

for d in [os.path.dirname(CLUSTERED_FILE), os.path.dirname(BERTOPIC_MODEL_PATH), IMAGE_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── Semantic Denoising ────────────────────────────────────────────────────────

_STOP_WORDS = [
    "true", "false", "student", "response", "sequence", "historical",
    "question", "next", "input", "output", "given", "predict",
    "and", "the", "of", "with", "in", "to", "for", "from", "by",
    "on", "at", "is", "are", "was", "were", "be", "this", "that",
]

def semantic_denoise(text: str) -> str:
    """Remove numeric/statistical tokens so the embedder focuses on concepts."""
    text = text.replace("The student's historical response sequence:", "")
    text = text.replace("the next question:", "")
    return text.strip()


# ─── Async Embedding Backend ──────────────────────────────────────────────────

class AsyncEmbeddingBackend(BaseEmbedder):
    """
    OpenAI-compatible embedding backend for BERTopic.

    Batches requests and runs them concurrently, writing progress to a tqdm bar.
    Compatible with any model that follows the OpenAI embeddings API, including
    Qwen3-Embedding and text-embedding-3-large.
    """

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 max_concurrency: int = 50):
        super().__init__()
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        async with self.semaphore:
            try:
                cleaned = [t.replace("\n", " ") for t in texts]
                resp = await self.client.embeddings.create(
                    input=cleaned, model=self.model_name
                )
                return [d.embedding for d in resp.data]
            except Exception as exc:
                print(f"[Embedding] API error: {exc}")
                dim = 4096  # Qwen3-8B dimension; adjust if using a different model
                return [[0.0] * dim for _ in texts]

    async def _embed_all_async(self, documents: list[str],
                               batch_size: int = 100) -> list[list[float]]:
        batches = [
            documents[i: i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]
        pbar = tqdm(total=len(batches), desc="Embedding (API)")

        async def _with_progress(coro):
            try:
                return await coro
            finally:
                pbar.update(1)

        results = await asyncio.gather(
            *[_with_progress(self._embed_batch_async(b)) for b in batches]
        )
        pbar.close()
        return [emb for batch in results for emb in batch]

    def embed(self, documents, verbose=False):
        """Synchronous wrapper required by BERTopic."""
        loop = asyncio.get_event_loop()
        batches = loop.run_until_complete(
            self._embed_all_async(documents, EMBEDDING_BATCH_SIZE)
        )
        return np.array(batches)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print(f"Loading data: {TRAIN_FILE}")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [semantic_denoise(item["input"]) for item in data]
    print(f"  {len(docs)} student sequences loaded.")

    # 2. Embedding (with disk cache)
    embedding_backend = AsyncEmbeddingBackend(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=EMBEDDING_MODEL,
        max_concurrency=EMBEDDING_CONCURRENCY,
    )

    embeddings = None
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            embeddings = np.load(EMBEDDING_CACHE_FILE)
            if len(embeddings) != len(docs):
                print("Cache size mismatch — recomputing.")
                embeddings = None
            else:
                print(f"Loaded cached embeddings {embeddings.shape}.")
        except Exception:
            embeddings = None

    if embeddings is None:
        print("Computing embeddings via API …")
        embeddings = embedding_backend.embed(docs)
        np.save(EMBEDDING_CACHE_FILE, embeddings)
        print(f"Embeddings saved → {EMBEDDING_CACHE_FILE}")

    # 3. Topic model (BERTopic wraps UMAP + HDBSCAN)
    vectorizer = CountVectorizer(
        stop_words=_STOP_WORDS,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # alphabetic tokens only
    )
    topic_model = BERTopic(
        embedding_model=embedding_backend,
        vectorizer_model=vectorizer,
        nr_topics=BERTOPIC_NR_TOPICS,
        min_topic_size=BERTOPIC_MIN_TOPIC,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # 4. Re-assign noise points (-1) to nearest topic
    if isinstance(probs, list):
        probs = np.array(probs)

    final_topics, final_scores = [], []
    for idx, t in enumerate(topics):
        if t == -1 and probs.shape[1] > 0:
            best = int(np.argmax(probs[idx]))
            final_topics.append(best)
            final_scores.append(float(probs[idx][best]))
        else:
            final_topics.append(int(t))
            score = float(np.max(probs[idx])) if len(probs.shape) > 1 else 1.0
            final_scores.append(score)

    # 5. Save BERTopic model
    topic_model.save(
        BERTOPIC_MODEL_PATH,
        serialization="safetensors",
        save_embedding_model=False,
    )
    print(f"BERTopic model saved → {BERTOPIC_MODEL_PATH}")

    # 6. Extract cluster metadata (keywords, representative docs)
    cluster_metadata = {}
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        keywords = [kw for kw, _ in (topic_model.get_topic(tid) or [])[:10]]
        cluster_metadata[int(tid)] = {
            "name": row["Name"],
            "count": int(row["Count"]),
            "keywords": keywords,
            "representative_samples": topic_model.get_representative_docs(tid),
            "description": f"Focuses on {', '.join(keywords[:5])}",
        }
    with open(CLUSTER_META_FILE, "w", encoding="utf-8") as f:
        json.dump(cluster_metadata, f, indent=4)
    print(f"Cluster metadata saved → {CLUSTER_META_FILE}")

    # 7. Annotate original data with cluster assignments
    for idx, item in enumerate(data):
        item["cluster_id"] = final_topics[idx]
        item["cluster_score"] = final_scores[idx]
    with open(CLUSTERED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Clustered data saved → {CLUSTERED_FILE}")


if __name__ == "__main__":
    main()
