"""
MERIT — Stage 2: Interpretative Memory Bank Construction

Selects the top-K most representative student sequences from each cognitive
schema (cluster) and uses an LLM to generate structured "Annotated Cognitive
Paradigm" entries (knowledge state, key pattern, difficulty context, and
causal reasoning).  The resulting memory bank is saved as a JSON file and
forms the external knowledge store for online inference in Stage 3.

Usage:
    python src/stage2_annotation.py
"""
import os
import sys
import json
import asyncio
import random

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    API_KEY, BASE_URL, ANNOTATION_MODEL,
    CLUSTERED_FILE, MEMORY_BANK_FILE,
    TOP_K_PER_CLUSTER, ANNOTATION_MAX_CONC,
)

os.makedirs(os.path.dirname(MEMORY_BANK_FILE), exist_ok=True)


# ─── Prompt Construction ──────────────────────────────────────────────────────

_ANNOTATION_SYSTEM = """\
You are a pedagogical expert tasked with creating an "Annotated Cognitive Paradigm"
for a student's learning trajectory.  This paradigm will be stored in a memory bank
to help interpret future students' behaviours.
"""

def build_annotation_prompt(student_input: str, ground_truth) -> str:
    outcome = "CORRECT" if str(ground_truth).lower() == "true" else "INCORRECT"
    return f"""\
### Data Legend
Each interaction: `(['Concept'], difficulty 0.0-1.0, Correct/Incorrect)`
  * 0.0 – 0.3  → Easy (Foundational)
  * 0.3 – 0.7  → Medium (Standard)
  * 0.7 – 1.0  → Hard (Advanced)

### Student History
{student_input}

### Target Outcome
The student answered the next question: **{outcome}**

### Annotation Task
Analyse **WHY** this outcome happened.  Dig into cognitive causes:
* Knowledge State — does the student understand this domain or is it guessing?
* Difficulty Delta — did the difficulty spike (e.g. 0.2 → 0.8)?  Careless slip?
* Momentum — winning streak carried over, or crashed?
* Concept Gap — consistent failure on specific topics?

### Output (JSON only)
{{
    "knowledge_state": "e.g. 'Solid on basics (<0.4 difficulty), fragile on advanced.'",
    "reasoning": "Causal explanation, e.g. 'Difficulty spike (0.2→0.8) exposed lack of deep mastery despite streak.'",
    "key_pattern": "ONE of: Difficulty Spike Failure | Careless Slip | Solid Mastery | Knowledge Gap | Lucky Guess | Gradual Improvement | Inconsistent Performance",
    "difficulty_context": "e.g. 'Significantly harder than previous questions.'"
}}
"""


# ─── Annotation Agent ─────────────────────────────────────────────────────────

class AnnotationAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90.0)
        self.semaphore = asyncio.Semaphore(ANNOTATION_MAX_CONC)

    async def annotate(self, item: dict, retries: int = 5) -> dict | None:
        async with self.semaphore:
            prompt = build_annotation_prompt(
                item["input"], item.get("output", "False")
            )
            backoff = 1
            for attempt in range(retries):
                try:
                    resp = await self.client.chat.completions.create(
                        model=ANNOTATION_MODEL,
                        messages=[
                            {"role": "system", "content": _ANNOTATION_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.0,
                    )
                    raw = resp.choices[0].message.content
                    clean = raw.replace("```json", "").replace("```", "").strip()
                    if not clean.endswith("}"):
                        clean += "}"
                    item["analysis"] = json.loads(clean)
                    return item
                except Exception as exc:
                    if attempt < retries - 1:
                        delay = backoff + random.uniform(0, 1)
                        if "429" in str(exc) or "Rate limit" in str(exc):
                            backoff *= 2
                        await asyncio.sleep(delay)
                    else:
                        print(f"[Annotation] Failed for item: {exc}")
            return None


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_representative_samples(path: str) -> list[dict]:
    """Select the top-K most representative sequences from each cluster."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters: dict[int, list[dict]] = {}
    for item in data:
        cid = item.get("cluster_id", -1)
        if cid == -1:
            continue
        clusters.setdefault(cid, []).append(item)

    selected = []
    for cid in sorted(clusters):
        ranked = sorted(clusters[cid], key=lambda x: x.get("cluster_score", 0), reverse=True)
        selected.extend(ranked[:TOP_K_PER_CLUSTER])

    print(f"  {len(data)} total → {len(selected)} representative samples selected.")
    return selected


# ─── Main Pipeline ────────────────────────────────────────────────────────────

async def main_async():
    agent = AnnotationAgent()

    print(f"Loading clustered data: {CLUSTERED_FILE}")
    targets = load_representative_samples(CLUSTERED_FILE)

    # Resume support: skip already-annotated inputs
    done_inputs: set[str] = set()
    completed: list[dict] = []
    if os.path.exists(MEMORY_BANK_FILE):
        try:
            with open(MEMORY_BANK_FILE, "r", encoding="utf-8") as f:
                completed = json.load(f)
            done_inputs = {item["input"] for item in completed}
            print(f"  Resuming: {len(completed)} done, {len(targets) - len(done_inputs)} remaining.")
        except Exception:
            print("  Existing file unreadable — starting fresh.")
            completed = []

    todo = [item for item in targets if item["input"] not in done_inputs]
    if not todo:
        print("All tasks already completed.")
        return

    print(f"\nStarting annotation | tasks={len(todo)} | concurrency={ANNOTATION_MAX_CONC}")

    SAVE_INTERVAL = 50
    buffer: list[dict] = []

    for coro in tqdm_asyncio.as_completed(
        [agent.annotate(item) for item in todo], desc="Annotating"
    ):
        result = await coro
        if result:
            completed.append(result)
            buffer.append(result)
        if len(buffer) >= SAVE_INTERVAL:
            with open(MEMORY_BANK_FILE, "w", encoding="utf-8") as f:
                json.dump(completed, f, indent=4, ensure_ascii=False)
            buffer.clear()

    with open(MEMORY_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(completed, f, indent=4, ensure_ascii=False)
    print(f"\nMemory bank saved → {MEMORY_BANK_FILE}  ({len(completed)} entries)")


if __name__ == "__main__":
    asyncio.run(main_async())
