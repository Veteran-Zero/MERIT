"""
MERIT — Ablation Study

Runs the four ablation variants reported in the paper by toggling ABLATION_MODE:

  FULL          Full MERIT framework (default)
  NO_RETRIEVAL  Base LLM with no external memory (tests parametric knowledge ceiling)
  NO_ROUTING    Flat retrieval over entire memory bank (disables cognitive schema routing)
  NO_TRACES     Retrieves raw interaction history instead of annotated paradigms
  NO_RULES      Dense-only retrieval (disables BM25 hybrid + logic constraints)

Usage:
    ABLATION_MODE=NO_RULES python src/ablation.py
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
    MEMORY_BANK_FILE, TEST_FILE, OUTPUT_DIR,
    BERTOPIC_MODEL_PATH, EMBEDDING_CACHE_FILE,
    TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, LENGTH_DIFF_TOLERANCE,
    ALPHA, BETA, INFERENCE_MAX_CONC,
)

# ─── Ablation Mode ────────────────────────────────────────────────────────────
ABLATION_MODE = os.environ.get("ABLATION_MODE", "FULL")
assert ABLATION_MODE in {"FULL", "NO_RETRIEVAL", "NO_ROUTING", "NO_TRACES", "NO_RULES"}, \
    f"Unknown ABLATION_MODE '{ABLATION_MODE}'"
print(f"[Ablation] Mode: {ABLATION_MODE}")

RESULT_FILE = os.path.join(OUTPUT_DIR, f"ablation_{ABLATION_MODE.lower()}.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Embedding Backend (shared with stage3) ──────────────────────────────────

class AsyncEmbeddingBackend(BaseEmbedder):
    def __init__(self, api_key, base_url, model_name, max_concurrency=50):
        super().__init__()
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_single_async(self, text: str):
        async with self.semaphore:
            try:
                resp = await self.client.embeddings.create(
                    input=[text.replace("\n", " ")], model=self.model_name
                )
                return resp.data[0].embedding
            except Exception:
                return None

    def embed(self, documents, verbose=False):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("Cannot call sync embed() from async context.")
        results = loop.run_until_complete(
            asyncio.gather(*[self._embed_single_async(d) for d in documents])
        )
        fallback_dim = 4096
        return np.array([r if r is not None else [0.0] * fallback_dim for r in results])


# ─── Difficulty Labelling ─────────────────────────────────────────────────────

_DIFF_RE = re.compile(r"\b(0\.\d+|1\.0)\b")

def inject_difficulty_labels(text: str) -> str:
    def _replace(m):
        v = float(m.group(1))
        tag = "[EASY]" if v < 0.3 else ("[MEDIUM]" if v < 0.7 else "[HARD]")
        return f"{v} {tag}"
    return _DIFF_RE.sub(_replace, text)


# ─── Logic Constraints Prompt Block ──────────────────────────────────────────

_SPIKE_RULE = """\
**DECISION LOGIC:**
* [EASY] Wins → [HARD]: PREDICT FAIL.
* [MEDIUM] Wins → [HARD]: cautious — FAIL unless streak >4 correct.
* Failed [EASY] → Any: PREDICT FAIL.
* Failed [MEDIUM/HARD] → Same/Harder: PREDICT FAIL.
* [HARD] Wins → [EASY]: PASS (cap at 0.8 for careless students).
* Uncertain: PREDICT FAIL (<0.5).
"""


def build_prompt(student_input: str, retrieved: list[dict], cluster_id: int) -> str:
    labeled = inject_difficulty_labels(student_input)

    # Context block — varies by ablation mode
    if ABLATION_MODE == "NO_RETRIEVAL" or not retrieved:
        context = "No reference cases available."
    elif ABLATION_MODE == "NO_TRACES":
        lines = ["### Reference Cases (raw history only)"]
        for i, res in enumerate(retrieved, 1):
            item = res["item"]
            outcome = "CORRECT" if str(item.get("output", "")).lower() == "true" else "INCORRECT"
            lines.append(
                f"--- [Case #{i}] (Sim: {res['score']:.2f}) ---\n"
                f"History: {item['input'][:300]}…\nOutcome: {outcome}"
            )
        context = "\n".join(lines)
    else:
        lines = ["### Reference Cases (Cognitive Peers)"]
        for i, res in enumerate(retrieved, 1):
            item = res["item"]
            outcome = "CORRECT" if str(item.get("output", "")).lower() == "true" else "INCORRECT"
            analysis = item.get("analysis", {})
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except Exception:
                    analysis = {}
            lines.append(
                f"--- [Case #{i}] (Sim: {res['score']:.2f}) ---\n"
                f"[Outcome]: {outcome}\n"
                f"[State]: {analysis.get('knowledge_state', 'N/A')}\n"
                f"[Pattern]: {analysis.get('key_pattern', 'N/A')}\n"
                f"[Analysis]: {analysis.get('reasoning', 'N/A')}"
            )
        context = "\n".join(lines)

    # Logic block — omitted if NO_RULES
    logic = "" if ABLATION_MODE == "NO_RULES" else _SPIKE_RULE

    return f"""\
You are an advanced Knowledge Tracing Expert.  Predict the student's next response.
Labels: [EASY] (0-0.3), [MEDIUM] (0.3-0.7), [HARD] (>0.7).
{logic}

{context}

### Target Student
{labeled}

Output JSON ONLY:
{{
    "probability": 0.5,
    "confidence_reasoning": "brief explanation"
}}
"""


# ─── Ablation Inference Engine ────────────────────────────────────────────────

class AblationEngine:
    def __init__(self):
        self.embed_backend = AsyncEmbeddingBackend(
            api_key=API_KEY, base_url=BASE_URL,
            model_name=EMBEDDING_MODEL, max_concurrency=100,
        )
        self.topic_model = None
        if ABLATION_MODE not in {"NO_RETRIEVAL", "NO_ROUTING"}:
            self.topic_model = BERTopic.load(
                BERTOPIC_MODEL_PATH, embedding_model=self.embed_backend
            )
        with open(MEMORY_BANK_FILE, "r") as f:
            self.memory_bank: list[dict] = json.load(f)
        self.cluster_indices: dict = {}
        if ABLATION_MODE != "NO_RETRIEVAL":
            self._build_indices()
        self.client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90.0)

    def _build_indices(self):
        cached_embs = None
        if os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                cached_embs = np.load(EMBEDDING_CACHE_FILE)
            except Exception:
                pass

        grouped: dict[int, dict] = {}
        for idx, item in enumerate(self.memory_bank):
            # NO_ROUTING: flatten everything into one global pool
            cid = 0 if ABLATION_MODE == "NO_ROUTING" else item.get("cluster_id", -1)
            if cid == -1:
                continue
            grouped.setdefault(cid, {"items": [], "vectors": []})
            grouped[cid]["items"].append(item)
            if cached_embs is not None:
                grouped[cid]["vectors"].append(cached_embs[idx])

        for cid, group in tqdm(grouped.items(), desc="Indexing"):
            if not group["vectors"]:
                continue
            vecs = np.array(group["vectors"], dtype="float32")
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            bm25 = None
            if ABLATION_MODE not in {"NO_RULES"}:
                corpus = [it["input"] for it in group["items"]]
                bm25 = BM25Okapi([d.lower().split() for d in corpus])
            self.cluster_indices[cid] = {
                "faiss_index": index, "bm25_index": bm25, "items": group["items"]
            }

    async def retrieve(self, text: str, cid: int, vec=None) -> list[dict]:
        if ABLATION_MODE == "NO_RETRIEVAL":
            return []
        target_cid = 0 if ABLATION_MODE == "NO_ROUTING" else cid
        if target_cid not in self.cluster_indices:
            return []
        idx_data = self.cluster_indices[target_cid]
        if vec is None:
            vec = await self.embed_backend._embed_single_async(text)
            if vec is None:
                return []
        q_np = np.array([vec], dtype="float32")
        faiss.normalize_L2(q_np)
        D, I = idx_data["faiss_index"].search(q_np, TOP_K_RETRIEVAL * 5)
        bm25_scores = None
        max_bm25 = 1.0
        if idx_data["bm25_index"] is not None:
            bm25_scores = idx_data["bm25_index"].get_scores(text.lower().split())
            max_bm25 = max(bm25_scores) if bm25_scores.max() > 0 else 1.0
        candidates = []
        for rank, item_idx in enumerate(I[0]):
            if item_idx == -1:
                continue
            candidate = idx_data["items"][item_idx]
            if abs(len(text) - len(candidate["input"])) > LENGTH_DIFF_TOLERANCE:
                continue
            if bm25_scores is not None:
                score = ALPHA * D[0][rank] + BETA * (bm25_scores[item_idx] / max_bm25)
            else:
                score = float(D[0][rank])
            if score < SIMILARITY_THRESHOLD:
                continue
            candidates.append({"item": candidate, "score": score})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:TOP_K_RETRIEVAL]

    async def predict(self, prompt: str, retries: int = 5) -> tuple[float, dict]:
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
                return max(0.0, min(1.0, float(obj.get("probability", 0.5)))), obj
            except Exception as exc:
                if attempt < retries - 1:
                    delay = backoff + random.uniform(0, 1)
                    if "429" in str(exc):
                        backoff *= 2
                    await asyncio.sleep(delay)
                else:
                    return 0.5, {"error": str(exc)}
        return 0.5, {}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(results: list[dict]) -> None:
    y_true = [1 if str(r["ground_truth"]).lower() == "true" else 0 for r in results]
    y_scores = [r["pred_prob"] for r in results]
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
    print(f"\n[Ablation: {ABLATION_MODE}]")
    print(f"AUC={roc_auc_score(y_true, y_scores):.4f}  "
          f"ACC={accuracy_score(y_true, y_pred):.4f}  "
          f"F1={f1_score(y_true, y_pred, zero_division=0):.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main_async():
    engine = AblationEngine()
    with open(TEST_FILE, "r") as f:
        test_data = json.load(f)

    semaphore = asyncio.Semaphore(INFERENCE_MAX_CONC)

    async def run_one(sample, idx):
        async with semaphore:
            text = sample["input"]
            vec = await engine.embed_backend._embed_single_async(text)
            fallback = vec if vec is not None else [0.0] * 4096
            cid = -1
            if engine.topic_model is not None:
                topics, _ = engine.topic_model.transform(
                    [text], embeddings=np.array([fallback])
                )
                cid = int(topics[0])
            retrieved = await engine.retrieve(text, cid, vec=fallback)
            prompt = build_prompt(text, retrieved, cluster_id=cid)
            prob, obj = await engine.predict(prompt)
            return {
                "id": idx,
                "cluster_id": cid,
                "ground_truth": sample.get("output", "False"),
                "pred_prob": prob,
                "pred_bool": prob >= 0.5,
                "reasoning": obj.get("confidence_reasoning", ""),
            }

    tasks = [run_one(s, i) for i, s in enumerate(test_data)]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Ablation [{ABLATION_MODE}]")
    results.sort(key=lambda x: x["id"])

    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results → {RESULT_FILE}")
    evaluate(results)


if __name__ == "__main__":
    asyncio.run(main_async())
