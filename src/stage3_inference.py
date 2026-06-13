"""
MERIT — Stage 3 & 4: Hierarchical Cognitive Retrieval + Logic-Augmented Inference

For each test student the pipeline:
  1. Embeds their interaction history.
  2. Routes them to the nearest cognitive schema via BERTopic.
  3. Retrieves the top-K most similar memory entries with hybrid search
     (dense FAISS + sparse BM25).
  4. Filters out low-quality candidates (similarity threshold, length ratio).
  5. Calls an LLM with the retrieved paradigms and explicit logic constraints
     (Spike Rule, Crash Rule) to predict next-step correctness.

Usage:
    python src/stage3_inference.py
"""
import os
import sys
import json
import asyncio
import random
import re

import numpy as np
import faiss
import openai
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from rank_bm25 import BM25Okapi
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    API_KEY, BASE_URL, EMBEDDING_MODEL, INFERENCE_MODEL,
    MEMORY_BANK_FILE, TEST_FILE, RESULT_FILE,
    BERTOPIC_MODEL_PATH, EMBEDDING_CACHE_FILE,
    TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, LENGTH_DIFF_TOLERANCE,
    ALPHA, BETA, INFERENCE_MAX_CONC,
)

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)


# ─── Embedding Backend ────────────────────────────────────────────────────────

class AsyncEmbeddingBackend(BaseEmbedder):
    """Async embedding backend (same as Stage 1) reused here for online encoding."""

    def __init__(self, api_key, base_url, model_name, max_concurrency=50):
        super().__init__()
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_single_async(self, text: str) -> list[float] | None:
        async with self.semaphore:
            try:
                resp = await self.client.embeddings.create(
                    input=[text.replace("\n", " ")], model=self.model_name
                )
                return resp.data[0].embedding
            except Exception:
                return None

    def embed(self, documents, verbose=False):
        """Synchronous wrapper for BERTopic compatibility."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("Cannot call sync embed() from an async context.")
        tasks = [self._embed_single_async(d) for d in documents]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        fallback_dim = 4096
        return np.array([r if r is not None else [0.0] * fallback_dim for r in results])


# ─── Difficulty Labelling ─────────────────────────────────────────────────────

_DIFF_PATTERN = re.compile(r"\b(0\.\d+|1\.0)\b")

def _label_difficulty(val: float) -> str:
    if val < 0.3:
        return "[EASY]"
    if val < 0.7:
        return "[MEDIUM]"
    return "[HARD]"

def inject_difficulty_labels(text: str) -> str:
    """Replace raw difficulty floats with labelled tokens, e.g. 0.85 → 0.85 [HARD]."""
    def _replace(m):
        v = float(m.group(1))
        return f"{v} {_label_difficulty(v)}"
    return _DIFF_PATTERN.sub(_replace, text)


# ─── Prompt Construction ──────────────────────────────────────────────────────

_LOGIC_CONSTRAINTS = """\
**DECISION LOGIC (Follow Strictly):**

**A. THE "SPIKE" RULES (Winning → Harder):**
* [EASY] Wins → [HARD] Next: **PREDICT FAIL**.
* [MEDIUM] Wins → [HARD] Next: **CAUTIOUS**.  Only predict PASS if the student
  has a Long Streak (>4 Correct).  If streak is short (<3), predict FAIL (0.45).

**B. THE "CRASH" RULES (Failing → Next):**
* Failed [EASY] → Any: **PREDICT FAIL** — failing an easy question is a red flag.
* Failed [MEDIUM] → [MEDIUM/HARD]: **PREDICT FAIL** — negative momentum persists.
* Failed [HARD] → [HARD]: **PREDICT FAIL**.

**C. THE "SLIP" RULE (Winning → Easier):**
* [HARD/MEDIUM] Wins → [EASY] Next: **PASS**, cap probability at 0.8 if the
  history shows carelessness.

**D. UNCERTAINTY:**
* If in doubt, **PREDICT FAIL (<0.5)**.  It is safer to assume the student
  needs help than to assume mastery.
"""

def build_inference_prompt(
    student_input: str,
    retrieved_memories: list[dict],
    cluster_id: int = -1,
) -> str:
    cluster_note = (
        "WARNING: This student belongs to a General/Unclassified group. "
        "Prioritise the student's recent history over reference cases.\n"
        if cluster_id == 0 else ""
    )

    if not retrieved_memories:
        context_block = "No specific reference cases available."
    else:
        lines = ["### Reference Cases (Cognitive Peers)"]
        for i, res in enumerate(retrieved_memories, 1):
            item = res["item"]
            score = res["score"]
            outcome = "CORRECT" if str(item.get("output", "")).lower() == "true" else "INCORRECT"
            analysis = item.get("analysis", {})
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except Exception:
                    analysis = {}
            lines.append(
                f"--- [Case #{i}] (Sim: {score:.2f}) ---\n"
                f"[Outcome]: {outcome}\n"
                f"[State]: {analysis.get('knowledge_state', 'N/A')}\n"
                f"[Pattern]: {analysis.get('key_pattern', 'N/A')}\n"
                f"[Analysis]: {analysis.get('reasoning', 'N/A')}"
            )
        context_block = "\n".join(lines)

    labeled_input = inject_difficulty_labels(student_input)

    return f"""\
### SYSTEM INSTRUCTIONS
You are an advanced Knowledge Tracing Expert.  Predict the student's next response.

**CRITICAL DEFINITIONS:**
1. Labels: [EASY] (0–0.3), [MEDIUM] (0.3–0.7), [HARD] (>0.7).
2. Negative Momentum: a recent failure strongly predicts future failure unless
   difficulty drops significantly.

{_LOGIC_CONSTRAINTS}

{cluster_note}

{context_block}

### Target Student Profile (difficulty-labelled)
{labeled_input}

### Prediction Task
1. Check Last Outcome — Pass or Fail?  If Fail, apply Crash Rule.
2. Check Difficulty Delta — is there a spike?  Apply Spike Rule.
3. Output JSON ONLY:

{{
    "probability": 0.5,
    "confidence_reasoning": "Explain using Crash/Spike Rule.",
    "risk_factor": "e.g. Negative Momentum, Difficulty Spike."
}}
"""


# ─── Inference Engine ─────────────────────────────────────────────────────────

class MERITInferenceEngine:
    """Full MERIT pipeline: BERTopic routing + hybrid FAISS/BM25 retrieval + LLM inference."""

    def __init__(self):
        print("Initialising MERIT Inference Engine …")

        self.embed_backend = AsyncEmbeddingBackend(
            api_key=API_KEY, base_url=BASE_URL,
            model_name=EMBEDDING_MODEL, max_concurrency=100,
        )

        print("  Loading BERTopic router …")
        self.topic_model = BERTopic.load(
            BERTOPIC_MODEL_PATH, embedding_model=self.embed_backend
        )

        print(f"  Loading memory bank: {MEMORY_BANK_FILE}")
        with open(MEMORY_BANK_FILE, "r", encoding="utf-8") as f:
            self.memory_bank: list[dict] = json.load(f)

        self.cluster_indices: dict = {}
        self._build_indices()

        self.client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90.0)

    def _build_indices(self):
        """Build per-cluster FAISS + BM25 indices from the memory bank."""
        print("  Building FAISS + BM25 indices …")

        cached_embs = None
        if os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                cached_embs = np.load(EMBEDDING_CACHE_FILE)
            except Exception:
                pass

        grouped: dict[int, dict] = {}
        for idx, item in enumerate(self.memory_bank):
            cid = item.get("cluster_id", -1)
            if cid == -1:
                continue
            if cid not in grouped:
                grouped[cid] = {"items": [], "vectors": []}
            grouped[cid]["items"].append(item)
            if cached_embs is not None:
                grouped[cid]["vectors"].append(cached_embs[idx])

        for cid, group in tqdm(grouped.items(), desc="Indexing clusters"):
            items = group["items"]
            if not group["vectors"]:
                continue
            vecs = np.array(group["vectors"], dtype="float32")
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            corpus = [item["input"] for item in items]
            bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
            self.cluster_indices[cid] = {
                "faiss_index": index, "bm25_index": bm25, "items": items
            }

    async def retrieve_async(
        self,
        query_text: str,
        cluster_id: int,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        if cluster_id not in self.cluster_indices:
            return []

        idx_data = self.cluster_indices[cluster_id]
        items = idx_data["items"]

        if query_vec is None:
            query_vec = await self.embed_backend._embed_single_async(query_text)
            if query_vec is None:
                return []

        q_np = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(q_np)
        D, I = idx_data["faiss_index"].search(q_np, TOP_K_RETRIEVAL * 5)

        bm25_scores = idx_data["bm25_index"].get_scores(query_text.lower().split())
        max_bm25 = max(bm25_scores) if bm25_scores.max() > 0 else 1.0

        query_len = len(query_text)
        candidates = []
        for rank, item_idx in enumerate(I[0]):
            if item_idx == -1:
                continue
            candidate = items[item_idx]
            if abs(query_len - len(candidate["input"])) > LENGTH_DIFF_TOLERANCE:
                continue
            score = ALPHA * D[0][rank] + BETA * (bm25_scores[item_idx] / max_bm25)
            if score < SIMILARITY_THRESHOLD:
                continue
            candidates.append({"item": candidate, "score": score})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:TOP_K_RETRIEVAL]

    async def predict_async(self, prompt: str, retries: int = 5) -> tuple[float, dict]:
        backoff = 1
        for attempt in range(retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=INFERENCE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content
                clean = content.replace("```json", "").replace("```", "").strip()
                if not clean.endswith("}"):
                    clean += "}"
                obj = json.loads(clean)
                prob = max(0.0, min(1.0, float(obj.get("probability", 0.5))))
                return prob, obj
            except Exception as exc:
                if attempt < retries - 1:
                    delay = backoff + random.uniform(0, 1)
                    if "429" in str(exc):
                        backoff *= 2
                    await asyncio.sleep(delay)
                else:
                    # Fallback: try regex extraction
                    m = re.search(r'"probability"\s*:\s*(\d+\.?\d*)', str(exc))
                    if m:
                        return float(m.group(1)), {}
                    return 0.5, {"error": str(exc)}
        return 0.5, {"error": "Max retries exceeded"}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(results: list[dict]) -> None:
    y_true = [1 if str(r["ground_truth"]).lower() == "true" else 0 for r in results]
    y_scores = [r["pred_prob"] for r in results]
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    print("\n" + "=" * 50)
    print("MERIT Performance Report")
    print("=" * 50)
    print(f"AUC       : {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("=" * 50)


# ─── Per-Sample Worker ────────────────────────────────────────────────────────

async def process_sample(
    engine: MERITInferenceEngine,
    semaphore: asyncio.Semaphore,
    sample: dict,
    idx: int,
) -> dict:
    async with semaphore:
        text = sample["input"]

        # 1. Embed
        vec = await engine.embed_backend._embed_single_async(text)
        fallback_vec = vec if vec is not None else [0.0] * 4096

        # 2. Route to cognitive schema
        q_np = np.array([fallback_vec])
        topics, _ = engine.topic_model.transform([text], embeddings=q_np)
        cid = int(topics[0])

        # 3. Retrieve relevant memory paradigms
        retrieved = await engine.retrieve_async(text, cid, query_vec=fallback_vec)

        # 4. Build prompt & predict
        prompt = build_inference_prompt(text, retrieved, cluster_id=cid)
        prob, response_obj = await engine.predict_async(prompt)

        return {
            "id": idx,
            "cluster_id": cid,
            "rag_count": len(retrieved),
            "ground_truth": sample.get("output", "False"),
            "pred_prob": prob,
            "pred_bool": prob >= 0.5,
            "confidence_reasoning": response_obj.get("confidence_reasoning", ""),
            "risk_factor": response_obj.get("risk_factor", ""),
        }


# ─── Main Pipeline ────────────────────────────────────────────────────────────

async def main_async():
    engine = MERITInferenceEngine()

    print(f"Loading test data: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Starting inference | samples={len(test_data)} | concurrency={INFERENCE_MAX_CONC}")
    semaphore = asyncio.Semaphore(INFERENCE_MAX_CONC)
    tasks = [process_sample(engine, semaphore, s, i) for i, s in enumerate(test_data)]
    results = await tqdm_asyncio.gather(*tasks, desc="Inference")
    results.sort(key=lambda x: x["id"])

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved → {RESULT_FILE}")

    evaluate(results)


if __name__ == "__main__":
    asyncio.run(main_async())
